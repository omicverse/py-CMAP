"""
Independent guidance graph construction.

Builds a bipartite graph linking ATAC peaks to RNA genes based on
genomic proximity. No dependency on scglue.

Algorithm:
  1. For each gene, define regulatory region = gene body + 2kb upstream
  2. For each peak, check overlap with gene regions (same chromosome)
  3. Edge weight = distance_decay(distance_to_gene)
  4. Add reverse edges + self-loops → symmetric bipartite graph
"""

import numpy as np
import pandas as pd
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)


def dist_power_decay(d, offset=1000, power=-0.75):
    """Distance-to-weight decay: w = ((d + offset) / offset) ^ power."""
    return ((d + offset) / offset) ** power


def parse_peak_names(peak_names):
    """Parse peak names like 'chr1:1000-2000' into DataFrame."""
    records = []
    for p in peak_names:
        try:
            chrom, se = p.split(':')
            s, e = se.split('-')
            records.append({'name': p, 'chrom': chrom,
                           'start': int(s), 'end': int(e)})
        except (ValueError, AttributeError):
            continue
    return pd.DataFrame(records)


def build_guidance_graph(adata_rna, adata_atac, gtf_path=None,
                         promoter_len=2000, extend_range=0,
                         decay_fn=dist_power_decay):
    """
    Build genomic guidance graph connecting RNA genes and ATAC peaks.

    For each gene, its regulatory window = [start - promoter_len, end].
    Peaks overlapping this window (within extend_range) are connected.

    Args:
        adata_rna: AnnData with gene expression.
            Needs var columns: chrom, chromStart, chromEnd.
            If missing, provide gtf_path.
        adata_atac: AnnData with ATAC peaks.
            var_names should be 'chrN:start-end' format.
        gtf_path: GTF file for gene coordinate annotation.
        promoter_len: bp upstream of TSS to include in gene region.
        extend_range: Additional bp beyond gene region for connections.
        decay_fn: Function(distance_bp) → weight.

    Returns:
        dict with:
          - 'edge_index': (2, n_edges) np.int64 array [source, target]
          - 'edge_weight': (n_edges,) float32
          - 'edge_sign': (n_edges,) int8 (+1 or -1)
          - 'n_rna': number of RNA features (indices 0..n_rna-1)
          - 'n_atac': number of ATAC features (indices n_rna..n_rna+n_atac-1)
          - 'rna_names': list of RNA feature names
          - 'atac_names': list of ATAC feature names
          - 'rna_hvg_mask': bool array, which RNA genes are highly variable
          - 'atac_hvg_mask': bool array, which ATAC peaks are connected to HVGs
    """
    # --- Annotate RNA genes ---
    rna_var = adata_rna.var.copy()
    if 'chrom' not in rna_var.columns:
        if gtf_path is None:
            raise ValueError("RNA var needs chrom/chromStart/chromEnd, or provide gtf_path")
        rna_var = _annotate_from_gtf(rna_var, gtf_path)

    # Filter genes with valid annotations
    has_ann = rna_var[['chrom', 'chromStart', 'chromEnd']].notna().all(axis=1)
    rna_var = rna_var[has_ann].copy()
    rna_names = list(rna_var.index)
    n_rna = len(rna_names)

    # --- Parse ATAC peaks ---
    peak_df = parse_peak_names(adata_atac.var_names)
    atac_names = list(peak_df['name'])
    n_atac = len(atac_names)

    logger.info(f"Building guidance graph: {n_rna} genes, {n_atac} peaks")

    # --- Normalize chromosome naming ---
    # Detect: do peaks use "chr" prefix? Do genes use "chr" prefix?
    peak_chroms = set(peak_df['chrom'].values[:100])
    gene_chroms = set(str(c) for c in rna_var['chrom'].dropna().values[:100])
    peaks_have_chr = any(c.startswith('chr') for c in peak_chroms)
    genes_have_chr = any(c.startswith('chr') for c in gene_chroms)

    if peaks_have_chr and not genes_have_chr:
        # Add "chr" prefix to gene chromosomes
        rna_var['chrom'] = rna_var['chrom'].apply(
            lambda x: f'chr{x}' if pd.notna(x) and not str(x).startswith('chr') else x)
        logger.info("Added 'chr' prefix to gene chromosomes to match peaks")
    elif genes_have_chr and not peaks_have_chr:
        # Strip "chr" prefix from gene chromosomes
        rna_var['chrom'] = rna_var['chrom'].apply(
            lambda x: str(x).replace('chr', '') if pd.notna(x) else x)
        logger.info("Stripped 'chr' prefix from gene chromosomes to match peaks")

    # --- Build gene regulatory regions ---
    # Combined region: gene body + promoter_len upstream
    gene_regions = []
    for _, row in rna_var.iterrows():
        chrom = str(row['chrom'])
        start = int(row['chromStart']) - promoter_len
        end = int(row['chromEnd'])
        if start < 0:
            start = 0
        gene_regions.append((chrom, start, end))

    # --- Index peaks by chromosome for fast lookup ---
    peak_by_chrom = defaultdict(list)
    for i, row in peak_df.iterrows():
        peak_by_chrom[row['chrom']].append((i, row['start'], row['end']))

    # Sort peaks by start position per chromosome
    for chrom in peak_by_chrom:
        peak_by_chrom[chrom].sort(key=lambda x: x[1])

    # --- Find overlapping peak-gene pairs ---
    edges_src = []  # gene index (in 0..n_rna-1)
    edges_tgt = []  # peak index (offset by n_rna)
    edges_wt = []
    edges_dist = []

    for gi, (chrom, g_start, g_end) in enumerate(gene_regions):
        # Extend gene region
        query_start = g_start - extend_range
        query_end = g_end + extend_range

        for pi, p_start, p_end in peak_by_chrom.get(chrom, []):
            # Check overlap: gene_region overlaps peak?
            if p_start >= query_end:
                break  # Sorted: all subsequent peaks are further right
            if p_end <= query_start:
                continue  # Peak is before gene region

            # Compute distance (0 if overlapping with gene body+promoter)
            if p_start < g_end and p_end > g_start:
                d = 0  # Overlap
            elif p_end <= g_start:
                d = g_start - p_end + 1
            else:
                d = p_start - g_end + 1

            if d <= extend_range:
                w = decay_fn(d)
                edges_src.append(gi)
                edges_tgt.append(n_rna + pi)  # Offset by n_rna
                edges_wt.append(w)
                edges_dist.append(d)

    n_edges_fwd = len(edges_src)
    logger.info(f"Forward edges: {n_edges_fwd}")

    if n_edges_fwd == 0:
        logger.warning("No edges found! Check chromosome naming (chr1 vs 1)")
        # Return empty graph
        return {
            'edge_index': np.zeros((2, 0), dtype=np.int64),
            'edge_weight': np.zeros(0, dtype=np.float32),
            'edge_sign': np.zeros(0, dtype=np.int8),
            'n_rna': n_rna, 'n_atac': n_atac,
            'rna_names': rna_names, 'atac_names': atac_names,
            'rna_hvg_mask': np.ones(n_rna, dtype=bool),
            'atac_hvg_mask': np.ones(n_atac, dtype=bool),
        }

    # --- Build full graph: forward + reverse + self-loops ---
    # Forward: gene → peak (sign=+1)
    # Reverse: peak → gene (sign=-1)
    # Self-loops: all nodes (sign=+1, weight=1.0)

    total_nodes = n_rna + n_atac

    # Forward edges
    fwd_src = np.array(edges_src, dtype=np.int64)
    fwd_tgt = np.array(edges_tgt, dtype=np.int64)
    fwd_wt = np.array(edges_wt, dtype=np.float32)
    fwd_sign = np.ones(n_edges_fwd, dtype=np.int8)

    # Reverse edges
    rev_src = fwd_tgt.copy()
    rev_tgt = fwd_src.copy()
    rev_wt = fwd_wt.copy()
    rev_sign = -np.ones(n_edges_fwd, dtype=np.int8)

    # Self-loops
    loop_nodes = np.arange(total_nodes, dtype=np.int64)
    loop_wt = np.ones(total_nodes, dtype=np.float32)
    loop_sign = np.ones(total_nodes, dtype=np.int8)

    # Concatenate
    edge_index = np.stack([
        np.concatenate([fwd_src, rev_src, loop_nodes]),
        np.concatenate([fwd_tgt, rev_tgt, loop_nodes]),
    ])
    edge_weight = np.concatenate([fwd_wt, rev_wt, loop_wt])
    edge_sign = np.concatenate([fwd_sign, rev_sign, loop_sign])

    # --- Normalize edges (symmetric) ---
    edge_weight = _normalize_edges_sym(edge_index, edge_weight, total_nodes)

    # --- HVG propagation ---
    rna_hvg = np.zeros(n_rna, dtype=bool)
    if 'highly_variable' in adata_rna.var.columns:
        for i, name in enumerate(rna_names):
            if name in adata_rna.var.index:
                rna_hvg[i] = bool(adata_rna.var.loc[name, 'highly_variable'])
    else:
        rna_hvg[:] = True

    # Propagate: peaks connected to HVGs are also "highly variable"
    atac_hvg = np.zeros(n_atac, dtype=bool)
    hvg_gene_set = set(np.where(rna_hvg)[0])
    for i in range(n_edges_fwd):
        if edges_src[i] in hvg_gene_set:
            atac_hvg[edges_tgt[i] - n_rna] = True

    n_fwd = n_edges_fwd
    n_total = len(edge_weight)
    logger.info(f"Guidance graph: {total_nodes} nodes, {n_total} edges "
                f"({n_fwd} fwd + {n_fwd} rev + {total_nodes} self-loops)")
    logger.info(f"HVG: {rna_hvg.sum()} RNA genes → {atac_hvg.sum()} ATAC peaks")

    return {
        'edge_index': edge_index,
        'edge_weight': edge_weight,
        'edge_sign': edge_sign,
        'n_rna': n_rna,
        'n_atac': n_atac,
        'rna_names': rna_names,
        'atac_names': atac_names,
        'rna_hvg_mask': rna_hvg,
        'atac_hvg_mask': atac_hvg,
    }


def _normalize_edges_sym(edge_index, edge_weight, n_nodes):
    """Symmetric normalization: w_norm = w / sqrt(deg_in[tgt]) / sqrt(deg_out[src])."""
    from scipy.sparse import coo_matrix

    src, tgt = edge_index[0], edge_index[1]
    adj = coo_matrix((np.abs(edge_weight), (src, tgt)), shape=(n_nodes, n_nodes))

    out_deg = np.array(adj.sum(axis=1)).ravel()
    in_deg = np.array(adj.sum(axis=0)).ravel()

    out_norm = np.where(out_deg > 0, out_deg ** (-0.5), 0.0)
    in_norm = np.where(in_deg > 0, in_deg ** (-0.5), 0.0)

    normed = edge_weight * out_norm[src] * in_norm[tgt]
    return normed.astype(np.float32)


def _annotate_from_gtf(var_df, gtf_path):
    """Annotate gene coordinates from GTF file."""
    import gzip

    gene_coords = {}
    opener = gzip.open if gtf_path.endswith('.gz') else open

    with opener(gtf_path, 'rt') as f:
        for line in f:
            if line.startswith('#'):
                continue
            fields = line.strip().split('\t')
            if len(fields) < 9 or fields[2] != 'gene':
                continue

            chrom = fields[0]
            start = int(fields[3]) - 1  # GTF is 1-based
            end = int(fields[4])

            # Extract gene_name from attributes
            attrs = fields[8]
            gene_name = None
            for attr in attrs.split(';'):
                attr = attr.strip()
                if attr.startswith('gene_name'):
                    gene_name = attr.split('"')[1]
                    break

            if gene_name:
                gene_coords[gene_name] = (chrom, start, end)

    # Apply to var
    for col, idx in [('chrom', 0), ('chromStart', 1), ('chromEnd', 2)]:
        var_df[col] = [gene_coords.get(g, (None, None, None))[idx]
                       for g in var_df.index]

    annotated = var_df['chrom'].notna().sum()
    logger.info(f"GTF annotation: {annotated}/{len(var_df)} genes annotated")
    return var_df
