"""
Graph Coarsening + GNN→MLP Distillation for GLUE acceleration.

Two-stage approach:
  Stage 1: Bipartite graph coarsening
    - Gene nodes stay as-is
    - ATAC peaks grouped by connected gene → super-peak nodes
    - 12k→4k nodes (3x smaller graph)

  Stage 2: GNN→MLP distillation
    - Extract GNN feature embeddings from trained GLUE
    - Train MLP to approximate: node_features → GNN_embedding
    - Replace GNN with MLP in subsequent training (no message passing)

Reference: GraphECL (ICML 2024), Graph Batch Coarsening (Neural Networks 2024)
"""

import numpy as np
import torch
import torch.nn as nn
import networkx as nx
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)


def coarsen_bipartite_graph(guidance_hvf, st_var_names, atac_var_names):
    """
    Coarsen bipartite guidance graph by grouping ATAC peaks per gene.

    Args:
        guidance_hvf: NetworkX graph (HVF subgraph from GLUE)
        st_var_names: gene names (ST features)
        atac_var_names: peak names (ATAC features)

    Returns:
        coarse_graph: Coarsened NetworkX graph
        super_peak_map: dict of {super_peak_name: [original_peak_names]}
        peak_to_super: dict of {original_peak: super_peak_name}
    """
    st_set = set(st_var_names)
    atac_set = set(atac_var_names)

    # Find cross-edges: gene ↔ peak
    peak_to_genes = defaultdict(list)
    gene_degree = defaultdict(int)

    for u, v in guidance_hvf.edges():
        if u == v:
            continue
        gene = u if u in st_set else (v if v in st_set else None)
        peak = u if u in atac_set else (v if v in atac_set else None)
        if gene and peak:
            peak_to_genes[peak].append(gene)
            gene_degree[gene] += 1

    # Assign each peak to its primary gene (most connected)
    peak_assignment = {}
    for peak, genes in peak_to_genes.items():
        best_gene = max(genes, key=lambda g: gene_degree[g])
        peak_assignment[peak] = best_gene

    # Group peaks by gene
    gene_to_peaks = defaultdict(list)
    for peak, gene in peak_assignment.items():
        gene_to_peaks[gene].append(peak)

    # Orphan peaks (not connected to any gene)
    orphans = [p for p in atac_set if p in guidance_hvf and p not in peak_assignment]

    # Build coarsened graph
    coarse_graph = nx.DiGraph()
    super_peak_map = {}
    peak_to_super = {}

    # Gene nodes: unchanged
    for gene in st_set:
        if gene in guidance_hvf:
            coarse_graph.add_edge(gene, gene, weight=1.0, sign=1)

    # Super-peak nodes: one per gene
    for gene, peaks in gene_to_peaks.items():
        sp_name = f"SP_{gene}"
        super_peak_map[sp_name] = peaks
        for p in peaks:
            peak_to_super[p] = sp_name
        coarse_graph.add_edge(gene, sp_name, weight=1.0, sign=1)
        coarse_graph.add_edge(sp_name, gene, weight=1.0, sign=1)
        coarse_graph.add_edge(sp_name, sp_name, weight=1.0, sign=1)

    # Add orphans as individual nodes
    for p in orphans:
        coarse_graph.add_edge(p, p, weight=1.0, sign=1)
        peak_to_super[p] = p

    logger.info(f"Graph coarsening: {guidance_hvf.number_of_nodes()} → {coarse_graph.number_of_nodes()} nodes "
                f"({len(super_peak_map)} super-peaks, {len(orphans)} orphans)")

    return coarse_graph, super_peak_map, peak_to_super


def aggregate_super_peak_data(adata_atac, super_peak_map):
    """
    Aggregate ATAC data for super-peaks (mean of member peaks).

    Args:
        adata_atac: AnnData with ATAC peaks
        super_peak_map: {super_peak_name: [peak_names]}

    Returns:
        AnnData with super-peaks as features (aggregated)
    """
    import anndata as ad
    import pandas as pd
    from scipy.sparse import issparse

    X = adata_atac.X
    if issparse(X):
        X = X.toarray()

    peak_to_idx = {p: i for i, p in enumerate(adata_atac.var_names)}

    sp_names = sorted(super_peak_map.keys())
    X_super = np.zeros((adata_atac.n_obs, len(sp_names)), dtype=np.float32)

    for j, sp in enumerate(sp_names):
        peaks = super_peak_map[sp]
        valid_idx = [peak_to_idx[p] for p in peaks if p in peak_to_idx]
        if valid_idx:
            X_super[:, j] = X[:, valid_idx].mean(axis=1)

    adata_super = ad.AnnData(
        X=X_super,
        obs=adata_atac.obs.copy(),
        var=pd.DataFrame(index=sp_names),
    )

    # Copy obsm
    for key in adata_atac.obsm:
        adata_super.obsm[key] = adata_atac.obsm[key].copy()

    logger.info(f"Super-peak aggregation: {adata_atac.n_vars} peaks → {len(sp_names)} super-peaks")
    return adata_super


class DistillMLP(nn.Module):
    """MLP to approximate GNN feature embeddings."""

    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.net(x)


def distill_gnn_to_mlp(glue_model, guidance_hvf, node_feature_dim=64,
                        hidden_dim=256, n_epochs=1000, lr=1e-3, device='cuda'):
    """
    Distill GLUE's GNN graph encoder to an MLP.

    Args:
        glue_model: Trained GLUE model
        guidance_hvf: HVF subgraph
        node_feature_dim: Dimension of synthetic node features
        hidden_dim: MLP hidden dimension
        n_epochs: Training epochs
        lr: Learning rate
        device: 'cuda' or 'cpu'

    Returns:
        DistillMLP model, node_features tensor
    """
    # Extract GNN embeddings
    feature_embeddings = glue_model.encode_graph(guidance_hvf)
    nodes = list(feature_embeddings.index)
    gnn_emb = torch.tensor(feature_embeddings.values, dtype=torch.float32, device=device)
    latent_dim = gnn_emb.shape[1]

    # Build node features: type + degree + position encoding
    node_features = []
    for node in nodes:
        is_gene = ':' not in node
        degree = guidance_hvf.degree(node) if node in guidance_hvf else 0
        # Simple features: [is_gene, degree_normalized, positional_hash]
        feat = np.zeros(node_feature_dim)
        feat[0] = 1.0 if is_gene else 0.0
        feat[1] = np.log1p(degree) / 10.0
        # Hash of node name for positional encoding
        h = hash(node) % (node_feature_dim - 2)
        feat[2 + h] = 1.0
        node_features.append(feat)

    node_features = torch.tensor(np.array(node_features), dtype=torch.float32, device=device)

    # Train MLP
    mlp = DistillMLP(node_feature_dim, hidden_dim, latent_dim).to(device)
    optimizer = torch.optim.AdamW(mlp.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, n_epochs)

    best_loss = float('inf')
    for epoch in range(n_epochs):
        pred = mlp(node_features)
        loss = nn.MSELoss()(pred, gnn_emb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        if loss.item() < best_loss:
            best_loss = loss.item()
            best_state = {k: v.clone() for k, v in mlp.state_dict().items()}

    mlp.load_state_dict(best_state)

    # Evaluate
    with torch.no_grad():
        pred = mlp(node_features)
        cos_sim = nn.CosineSimilarity(dim=1)(pred, gnn_emb).mean().item()
        mse = nn.MSELoss()(pred, gnn_emb).item()

    logger.info(f"GNN→MLP distillation: MSE={mse:.4f}, cosine_sim={cos_sim:.4f}")

    return mlp, node_features, nodes
