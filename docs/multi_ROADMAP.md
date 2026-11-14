# CMAP_multi Improvement Roadmap

## Current Status (v4)
- GLUE引导图 + CMAP/FGW空间映射
- Cell-type specificity: 52% (ATAC+ST only)
- Cell-type centroid correlation: 0.87 (三模态RNA vs ATAC)
- GLUE训练: ~30min (9k cells, 240k features)
- 瓶颈: GLUE嵌入质量决定了52%上限，不是下游OT

## Phase 1: 加速训练 (当前)

### 1a. Graph Coarsening
- 164k节点 → ~10-20k节点
- 保持Laplacian谱性质
- 参考: Graph Batch Coarsening (Neural Networks, 2024)
- 预期效果: 图编码器速度提升5-10x

### 1b. GNN → MLP蒸馏
- 训练一次完整GNN，然后蒸馏到MLP
- MLP在VAE训练循环中替代GNN (无message passing)
- 参考: GraphECL (ICML 2024)
- 预期效果: 每epoch速度提升10x+

### 1c. ATAC HVG预筛选
- 当前用24万peaks → decoder输出层巨大
- 只用引导图可达的HVG peaks (~15-25k)
- 已在v3实现 (145k → 15k), 效果显著

## Phase 2: 改善嵌入质量

### 2a. Gromov-Monge Gap正则化 (ICLR 2025)
- 在VAE encoder中加GMG正则项
- L_gmg = GW(f(X), X) - W(f(X), f(X))
- 强制encoder保持成对距离关系
- 参考: Apple Research, ICLR 2025
- 预期效果: 提升cell-level精度, 突破52%上限

### 2b. VICReg替代Adversarial Alignment
- 去掉discriminator
- 用Variance + Invariance + Covariance三项loss
- 避免"过度混合"问题
- 参考: VICReg (ICLR 2022), Barlow Twins (ICML 2021)
- 预期效果: 更稳定训练, 更好的细粒度对齐

### 2c. Sliced Wasserstein Discrepancy
- 用SWD替代或补充对齐loss
- 保持分布几何结构 (adversarial只匹配marginal)
- O(n log n)每次slice, 可微分
- 参考: SWD for Domain Adaptation (CVPR 2019+)

## Phase 3: 算法创新 (论文级)

### 3a. Graph-guided Contrastive Alignment
- 引导图提供正样本对 (gene ↔ peak)
- PolyGCL的频谱滤波器生成对比视图
- 异质图对比学习 (HGCLADA)
- 替代VAE+GAN为contrastive+graph框架

### 3b. FGW Flow Matching (GENOT-style)
- 学习连续的transport map而非离散plan
- 直接输出(x,y)坐标而非spot分配
- 引导图结构作为GW cost
- 参考: GENOT (NeurIPS 2024)

### 3c. Partial/Unbalanced FGW
- 不强制所有ATAC cells都映射
- 没有空间对应的cell type保持unmatched
- 参考: R-FPGWOT (arXiv 2025)

## Benchmark Plan
- 对比: Tangram, CytoSPACE, novoSpaRc
- 数据: Chen-2019 (cortex), 10x Brain, PBMC Multiome
- 指标: cell-type specificity, per-cell距离, 运行时间
- 消融: 每个组件的贡献
