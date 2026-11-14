# CMAP_multi: Cross-Modal Spatial Mapping (scATAC-seq → Spatial)

## Critical Notes (from implementation experience)

### GLUE Preprocessing Checklist
1. **Chromosome naming must match**: RNA (from GTF) uses `1,2,3`, ATAC (from 10x) uses `chr1,chr2,chr3` → add `chr` prefix to RNA: `rna.var['chrom'] = 'chr' + rna.var['chrom'].astype(str)`
2. **ATAC uses NB model** (not Bernoulli): `configure_dataset(atac, "NB", use_rep="X_lsi")`
3. **RNA raw counts in layer**: `rna.layers["counts"] = rna.X.copy()` before normalization, then `use_layer="counts"`
4. **HVG propagation**: Mark ATAC peaks reachable from RNA HVGs in guidance graph, then `use_highly_variable=True`
5. **HVF subgraph**: `guidance.subgraph(chain(rna_hvg, atac_hvg))` — reduces graph from 164k to ~25k nodes
6. **Verify graph**: Cross-edges should be >>1000 (not 26!)

### Results Comparison
| Version | Same-cell corr | Top-10 match | Training | Error |
|---------|---------------|-------------|----------|-------|
| v1 (Bernoulli, no binarize) | 0.054 | - | 100 min | Embedding explosion |
| v2 (Bernoulli, binarized) | 0.346 | 3.8% | 60 min | No guidance graph (chr mismatch) |
| **v3 (NB, chr fix, HVF)** | **0.804** | **25.7%** | **36 min** | ✓ |

### Architecture Evolution

**v1-v3**: 间接架构 (kNN投射)
```
ATAC → GLUE → embedding → kNN找RNA邻居 → 估算ST embedding → CMAP
                                ↑
                          引入大量误差!
```

**v4 (current)**: 直接架构 (ST参与GLUE训练)
```
ATAC ──┐                    
       ├── GLUE共同训练 ──→ 共享嵌入 ──→ CMAP映射
ST ────┘
  ↑
  直接编码，无kNN投射
```

关键改进:
1. **数据匹配**: Adult ATAC + Adult Visium (同组织同时期)
2. **ST直接参与GLUE**: Visium spots作为第二个模态训练 → 直接得到嵌入
3. **消除kNN投射**: ATAC和ST在同一GLUE空间直接比较

## GLUE算法原理 (Gao Lab, Nature Biotechnology 2022)

### 核心思想：引导图 (Guidance Graph)

GLUE的关键创新是用一个**基因组先验知识图**来桥接不同模态的特征空间：

```
引导图 G = (V, E)
  节点 V: 所有模态的特征 (RNA基因 + ATAC peaks + ...)
  边 E:  基于基因组位置的调控关系
         - 当ATAC peak在基因TSS上游2kb内 → 连边 (promoter)
         - 当ATAC peak在基因±150kb内 → 连边 (distal enhancer)
         - 边权重: w = ((d + 1000)/1000)^{-0.75} (距离衰减)
```

**作用**: 引导图告诉模型哪些ATAC peaks应该和哪些RNA基因关联，提供了跨模态的"翻译词典"。

### 模型架构：双VAE + 图神经网络 + 对抗训练

```
                    引导图 (基因组先验)
                         │
              ┌──────────┼──────────┐
              │    Graph Encoder    │  ← GNN: 学习每个特征(基因/peak)的向量表示
              │    (图卷积)         │
              └──────────┬──────────┘
                   特征嵌入 v
                    ╱          ╲
         ┌─────────┐          ┌─────────┐
         │RNA VAE  │          │ATAC VAE │  ← 每个模态独立的编码器/解码器
         │Encoder  │          │Encoder  │
         └────┬────┘          └────┬────┘
           细胞嵌入 u_rna       细胞嵌入 u_atac
              ╲                    ╱
               ╲                  ╱
            ┌───────────────────────┐
            │   Discriminator      │  ← 对抗训练: 分辨不出模态来源
            │   (对抗对齐)          │     → 强制 u_rna 和 u_atac 在同一空间
            └───────────────────────┘
                     │
               共享细胞嵌入空间
```

### 损失函数

```
L = L_data + λ_graph × L_graph - λ_align × L_adversarial

L_data  = Σ_k [NLL(x_k | decoder_k(u_k, v)) + λ_kl × KL(q(u_k|x_k) || p(u))]
          ↑ 每个模态的VAE重构损失

L_graph = NLL(edges | decoder_graph(v)) + λ_kl × KL(q(v|G) || p(v))
          ↑ 图重构损失: 确保特征嵌入保持基因组结构

L_adversarial = CE(discriminator(u), modality_label)
          ↑ 对抗损失: 对齐不同模态的细胞嵌入
```

**关键**: 图重构损失强制**相邻的基因和peaks获得相似的嵌入向量**，从而自然地桥接了两个模态。

### GLUE的启发：引导图思想在CMAP_multi中的应用

GLUE的引导图给了一个很好的先验：**不需要gene activity score这种粗糙的信号聚合，
而是让模型自己学习peaks和genes之间的对应关系**。

---

## 核心问题

```
scATAC-seq: 染色质可及性 (100k-500k peaks)
Spatial ST: 基因表达      (20k genes)
                ↓ 不同模态，不同特征空间
需要: 桥接模态差异 → 映射ATAC细胞到空间位置
```

## 方案4 (推荐): GLUE引导图嵌入 + CMAP空间映射

```
Step 1: 构建引导图 (基因组先验)
   RNA基因 ←→ ATAC peaks (基于TSS距离, 距离衰减权重)

Step 2: GLUE训练 (在scATAC + scRNA/ST上联合训练)
   → 得到共享的细胞嵌入空间
   → ATAC细胞和ST spots在同一个嵌入空间里

Step 3: 在共享嵌入空间中运行CMAP的3级映射
   Level 1: SVM (在嵌入空间上训练, 不在基因表达空间)
   Level 2: SSIM优化器 (在嵌入空间上计算相似度)
   Level 3: 相关性插值 (在嵌入空间上计算Spearman)
```

### 为什么GLUE + CMAP是最优组合

| 问题 | 纯GLUE | 纯CMAP | GLUE + CMAP |
|------|--------|--------|-------------|
| 跨模态对齐 | 强 (引导图+对抗) | 不能 | GLUE负责 |
| 空间映射精度 | 无 (GLUE不做空间) | 强 (3级精细化) | CMAP负责 |
| 需要gene activity | 不需要 | 需要 | 不需要 |
| 保留ATAC特异信息 | 保留 | 丢失 | 保留 |

### 实现架构

```python
# Step 1: 引导图
import scglue
graph = scglue.genomics.rna_anchored_guidance_graph(adata_rna, adata_atac)

# Step 2: GLUE训练 → 共享嵌入
scglue.models.fit_SCGLUE(
    {"rna": adata_st, "atac": adata_atac},  # ST数据作为RNA模态
    graph, fit_kws={"directory": "glue_model"}
)
# 提取嵌入
st_emb = adata_st.obsm["X_glue"]   # spots在共享空间
atac_emb = adata_atac.obsm["X_glue"]  # ATAC细胞在共享空间

# Step 3: CMAP映射 (在嵌入空间上)
# 直接把GLUE嵌入作为"表达矩阵"传给CMAP
# 跳过quantile_norm和harmony (因为GLUE已经对齐了)
```

### vs 方案1-3的优势

- **比Gene Activity精确**: 不做粗糙的peak求和，而是学习peak-gene关系
- **比Multiome训练灵活**: 不需要配对数据，引导图提供基因组先验
- **比FGW稳定**: GLUE的VAE+对抗训练比纯最优传输更鲁棒
- **最大创新点**: 引导图 + CMAP空间精细化 的组合是全新的

---

## 其他三种备选方案

### 方案1: Gene Activity Score + 标准CMAP (最简单，无需改核心代码)

```
scATAC peaks → Gene Activity Score (peak-to-gene聚合) → 伪表达矩阵 → 标准CMAP
```

**原理**: 将每个基因附近的ATAC peaks信号求和，得到"基因活性分数"，作为表达量的代理。

**优点**: 零代码改动，直接复用CMAP pipeline
**缺点**: Gene activity ≠ 真实表达 (per-gene相关性 r≈0.3-0.6)，远端增强子信息丢失
**适用**: 快速验证、细胞类型差异明显的场景

```python
# 使用方法
import muon as mu
mu.atac.tl.gene_activity(adata_atac)  # 计算gene activity
CMAP_py.run_cmap_fast(adata_atac, adata_st, spatial_genes)  # 直接调用
```

---

### 方案2: Multiome训练 + RNA表达预估 + CMAP (最高精度，需配对数据)

```
10x Multiome (配对RNA+ATAC) → 训练 ATAC→RNA 映射模型
未配对scATAC → 模型预测 → 预估RNA表达 → 标准CMAP  
```

**原理**: 在配对的Multiome数据上学习ATAC→RNA的非线性映射，然后对未配对ATAC数据预测表达。

**数学模型**:
- 线性: Ridge回归 `RNA = W @ ATAC_peaks + b` (per-gene)
- 非线性: 浅层MLP `RNA = f(ATAC_embedding)`
- 先验知识: 只用基因周围±500kb的peaks作为该基因的预测特征

**优点**: 预估表达质量高 (r≈0.5-0.8)，捕获远端调控关系
**缺点**: 需要同组织/同物种的Multiome配对数据
**适用**: 有Multiome参考数据的场景

---

### 方案3: Gene Activity + Fused Gromov-Wasserstein (创新算法，最有潜力)

```
Level 1: Gene Activity → Harmony → SVM (域预测，与CMAP相同)
Level 2: 不用SSIM，改用 Fused Gromov-Wasserstein:
         - Wasserstein项: gene activity相似性 (跨模态特征匹配)
         - GW项: ATAC空间 vs ST空间的拓扑结构保持 (模态内结构保持)
Level 3: 坐标插值 (与CMAP相同)
```

**原理**: 最优传输(OT)在不需要共享特征空间的情况下找到两个分布之间的对应关系。
Fused GW结合了：
- **Wasserstein距离**: 在gene activity共享特征空间中匹配（类似SSIM）
- **Gromov-Wasserstein距离**: 保持两个空间各自的拓扑结构（即使特征不同也能对齐）

**数学公式**:

```
min_T  α * Σ_ijkl |C_atac(i,k) - C_st(j,l)|² * T_ij * T_kl   (GW: 结构保持)
     + (1-α) * Σ_ij d(x_i, y_j) * T_ij                        (W: 特征匹配)  
     - ε * H(T)                                                 (熵正则化)

其中:
  T: 传输计划 (cells × spots)，即映射矩阵
  C_atac: ATAC细胞间的距离矩阵 (kNN图)
  C_st: Spot间的距离矩阵 (空间+表达)
  d(x,y): gene activity embedding距离
  α: GW vs W的权重 (0=纯Wasserstein, 1=纯GW)
  ε: 熵正则化参数 (控制映射的模糊度)
```

**优点**: 
- 即使gene activity质量差，GW项通过结构保持补偿
- 天然产出概率映射矩阵（与CMAP Level 2输出格式一致）
- 可用POT库GPU加速

**缺点**: 计算量大 O(n²) 内存，需要逐域计算
**适用**: 数据质量参差不齐、需要高鲁棒性的场景

---

## 方案对比

| 维度 | 方案1 Gene Activity | 方案2 Multiome训练 | 方案3 FGW |
|------|-------------------|--------------------|-----------|
| 需要配对数据 | 否 | **是** | 否 |
| 代码改动量 | 无 | 新增预处理模块 | 替换Level 2优化器 |
| 预期精度 | 中 | **高** | 中-高 |
| 计算成本 | 低 | 中(训练)+低(推理) | 高 |
| 创新性 | 低 | 中 | **高** |
| Python依赖 | muon | torch | POT |

## 推荐实现路线

```
Phase 1 (立即可做):
  方案1 — Gene Activity Score + 现有CMAP
  → 零代码改动，验证可行性
  → 建立baseline准确率

Phase 2 (核心创新):
  方案4 — GLUE引导图嵌入 + CMAP空间映射
  → GLUE做跨模态对齐，CMAP做空间精细化
  → 保留ATAC特异信息，不丢失为gene activity
  → 论文核心创新点

Phase 3 (增强):
  方案4 + FGW — GLUE嵌入 + Fused GW优化器
  → 用GLUE嵌入替代gene activity做FGW的特征匹配项
  → 用GW项保持ATAC拓扑结构
  → 进一步提升映射精度

Phase 4 (有配对数据时):
  方案2 — Multiome训练 + CMAP
  → 作为精度上限参考
```

## 关键依赖包

```bash
pip install muon          # gene activity score
pip install POT           # optimal transport (Gromov-Wasserstein)
pip install lamin         # (optional) multiome数据管理
```

## 文件结构

```
CMAP_multi/
├── __init__.py
├── gene_activity.py       # 方案1: ATAC → gene activity
├── glue_bridge.py         # 方案4: GLUE引导图嵌入 + CMAP
├── atac_to_rna.py         # 方案2: Multiome训练 → RNA预估
├── fgw_optimizer.py       # 方案3: Fused Gromov-Wasserstein
├── pipeline.py            # 统一入口: run_cmap_multi()
└── notebooks/
    ├── 01_gene_activity_mapping.ipynb
    ├── 02_glue_spatial_mapping.ipynb
    ├── 03_multiome_imputation.ipynb
    └── 04_fgw_mapping.ipynb
```

## GLUE + CMAP 详细算法流程

```
输入:
  adata_atac: scATAC-seq (cells × peaks)
  adata_st:   Spatial ST  (spots × genes)
  genome:     基因组注释 (GTF)

Step 1: 构建引导图
  ┌──────────────────────────────────┐
  │ 节点: RNA基因 ∪ ATAC peaks       │
  │ 边:   peak在gene TSS ±150kb内   │
  │ 权重: w = ((d+1000)/1000)^-0.75 │
  └──────────────────────────────────┘

Step 2: GLUE训练
  ┌──────────────────────────────────┐
  │ 图编码器: 学习每个基因/peak的      │
  │          向量表示 (引导图约束)     │
  │ RNA VAE:  ST spots → 嵌入 u_st   │
  │ ATAC VAE: ATAC cells → 嵌入 u_atac│
  │ 对抗器:   对齐 u_st 和 u_atac     │
  └──────────────────────────────────┘
       ↓
  共享嵌入: u_st (spots × d), u_atac (cells × d)

Step 3: CMAP Level 1 (在嵌入空间)
  ┌──────────────────────────────────┐
  │ 训练集: u_st + HMRF标签           │
  │ 测试集: u_atac                    │
  │ SVM → 每个ATAC细胞 → 域标签      │
  │ 注: 跳过quantile_norm + harmony  │
  │     (GLUE已经对齐了)             │
  └──────────────────────────────────┘

Step 4: CMAP Level 2 (在嵌入空间)
  ┌──────────────────────────────────┐
  │ 每个域:                           │
  │   cell_emb (cells_d × d)         │
  │   spot_emb (spots_d × d)         │
  │   SSIM优化器 → cell-spot映射     │
  └──────────────────────────────────┘

Step 5: CMAP Level 3 (在嵌入空间)
  ┌──────────────────────────────────┐
  │ Spearman相关性 (嵌入空间)         │
  │ 邻居加权插值 → (x, y) 坐标       │
  └──────────────────────────────────┘

输出:
  adata_atac.obsm['spatial_CMAP']  # ATAC细胞的空间坐标
  adata_atac.obs['CMAP_quality']   # 映射质量分数
```
