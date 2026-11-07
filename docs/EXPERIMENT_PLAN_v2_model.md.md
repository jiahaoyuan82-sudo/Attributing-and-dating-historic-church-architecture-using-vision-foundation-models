# 补充实验方案 v2（强化版）：提出模型 ChronoArch 与消融实验

> 本文件是 `EXPERIMENT_PLAN.md` 的**补充**。数据集、划分、已有结果全部沿用，不得改动。交给 Codex 执行。
> 目标：在**公平比较**的前提下，让提出的模型取得尽可能强的结果。
> 红线（违反任何一条都会让论文不可信）：
> 1. 超参数、早停、模型选择只用验证集或训练折内部的验证集；**测试集对每个最终配置只评估一次**；
> 2. 基线使用与原文献一致或更好的标准设置，**不得故意削弱**；
> 3. 所有结果（包括没赢的）如实写入结果文件；不得编造或挑选数字。

---

## Part A — 设计说明（给作者）

### A1. 三个改进点（论文中的方法创新）及其依据

| 已有发现 | 改进 |
|---|---|
| 诊断特征很小（专家框并集面积中位数 5.2%），Renaissance 最难，原论文报告高分辨率有帮助 | **改进 1：多尺度局部特征注意力池化**（224 + 448 两个分辨率的 patch token，可学习查询注意力聚合） |
| DINOv2（视觉结构）与 CLIP/SigLIP（语义）机制不同，错误可能互补 | **改进 2：门控多编码器融合**（DINOv2 局部表示 + CLIP + SigLIP 2 全局表示，逐维门控） |
| 一级风格标签混合了原作与复兴（Gothic 年份中位数 1837）；最细标签含大量年代信息（断代 MAE 81 年 vs 194 年） | **改进 3：层级与年代感知的多任务学习**（主风格 + 最细风格 + 有序年代分箱，测试时只输入图像） |

### A2. 为了“最强”加入的训练配方（不算创新点，但全部做消融）

| 配方 | 作用 |
|---|---|
| **更强骨干**：DINOv2 ViT-L/14（替代 ViT-B/14） | 更强的视觉特征；8 GB 显存可推理 |
| **LoRA 轻量适配**（ChronoArch-LoRA 版本）：DINOv2 最后 4 个 block 的 qkv 加 LoRA（rank 8） | 在冻结版之外提供更强版本，可训练参数仍很少 |
| **长尾损失**：logit adjustment（按训练类频率调整 logit） | 针对 WC14 macro-F1 偏低（少数类被误判）的问题 |
| **特征级数据增强**：训练时使用预先提取的水平翻转视图与 2 个随机裁剪视图 | 冻结骨干下也能做增强 |
| **5 个 seed 集成 + 测试时翻转增强** | 降低方差、稳定提升 |

论文中提供两个版本：
- **ChronoArch-F**（骨干完全冻结，强调低成本）；
- **ChronoArch-L**（LoRA 轻量适配，追求最佳性能）。

### A3. 文章的三个创新点（建议写法）

1. **方法**：提出 ChronoArch，集成多尺度局部注意力池化、门控多编码器融合、层级与年代感知多任务学习，可在冻结或轻量适配条件下同时完成风格归属、原作/复兴辨识与断代。
2. **问题与评估**：首次在公开数据上系统检验视觉模型区分中世纪原作与 19 世纪复兴建筑的能力，并配合三项混杂控制与“超越风格标签”的断代分析。
3. **可解释性**：“去除/仅保留”专家诊断区域的双向遮挡实验，区分诊断特征的充分性与必要性。

### A4. 为什么基线必须这样设
审稿人一定会问：“提升是不是只来自更大的骨干、更多的编码器或者集成？”所以基线里包括**同骨干、同编码器组合、同集成方式**的简单方法。ChronoArch 赢过这些，才说明三个模块本身有用。这样的结果最有说服力。

---

## Part B — 执行规范（交给 Codex）

### B0. 总原则
1. **存放位置（必须严格遵守）**：所有新产生的文件都放在桌面的 `Attributing and dating historic church architecture using vision foundation models_实验数据` 文件夹（下称“实验数据文件夹”）内，与第一轮的 `results/`、`figures/`、`features/` 并列：
   - 新代码：`实验数据文件夹/code/src/`（与已有代码放在一起）；
   - 新结果：`实验数据文件夹/results_v2/`；
   - 新图：`实验数据文件夹/figures_v2/`；
   - 新特征：`实验数据文件夹/features_v2/`；
   - `RESULTS_v2.md` 放在实验数据文件夹根目录，与 `RESULTS.md` 并列；运行日志继续追加到同一个 `RUN_LOG.md`；
   - 原始数据集只从桌面的 `…_实验材料` 文件夹读取，**不修改、不移动、不复制**其中任何文件；
   - **不覆盖、不删除**第一轮的任何文件。
2. 全部深度模型计算在 GPU（RTX 5060 Laptop，8 GB）上进行；显存不足时用较小 batch、梯度累积、梯度检查点，并记录。
3. 数据划分完全沿用：WC6 官方划分；WC14 `splits/wc14.csv`；RQ3a/RQ3b 已有 5 折教堂分组划分。RQ3a/RQ3b 在每个外层训练折内再按教堂切出 15% 作为内部验证集。
4. **选择规则**：所有超参数、早停、版本选择只看验证集；`results_v2/selection_log.csv` 记录全部网格结果与最终选择。测试集与外层测试折只在配置确定后评估一次。
5. 每个可训练配置跑 5 个 seed（0–4）；报告单 seed 均值 ± SD，以及 5 seed 集成结果。置信区间用教堂 bootstrap 1,000 次。
6. 记录每个配置的可训练参数量、训练时间、推理时间、峰值显存。
7. 改动设计前先停下来问用户；不得编造数字。

### B1. 特征提取（新增）
1. **DINOv2 ViT-L/14**（`facebook/dinov2-large`）：
   - 224：16×16 patch token + CLS；
   - 448（从 `…_实验材料` 中的原图重新缩放，短边 448、中心裁剪）：32×32 token，2×2 平均池化到 16×16；
   - 每个分辨率另存**水平翻转**视图；训练集另存 2 个随机裁剪视图（RandomResizedCrop scale 0.6–1.0，seed 固定）；
   - fp16、memmap 存储于 `features_v2/`；预计每个视图约 5 GB，先检查桌面磁盘剩余空间并报告。
2. **DINOv2 ViT-B/14** 448 分辨率 token（用于消融“骨干大小”）。
3. CLIP ViT-L/14、SigLIP 2 ViT-B/16 全局嵌入：复用已有文件，另提取翻转视图。
4. RQ4 遮挡实验的各遮挡条件图，同样提取上述特征。

### B2. 模型定义（`src/chronoarch.py`）
```
输入：T224, T448 (各 256×1024，ViT-L)，cls224, cls448, z_clip, z_siglip
改进 1 多尺度注意力池化：
  tokens = concat(T224 + scale_emb0, T448 + scale_emb1) → 512×1024
  LayerNorm → Linear → d = 384
  K 个可学习查询，2 层交叉注意力（6 头，带前馈层与残差） → K×384 → 展平 → Linear → h_dino (384)
  h_dino += Linear(concat(cls224, cls448))
改进 2 门控多编码器融合：
  h_clip = Linear(LN(z_clip)) → 384；h_sig = Linear(LN(z_siglip)) → 384
  [g_d, g_c, g_s] = softmax(Linear([h_dino; h_clip; h_sig]) 按每一维重塑为 3 路)
  h = g_d*h_dino + g_c*h_clip + g_s*h_sig
改进 3 多任务头：
  style_head（主任务，logit adjustment τ ∈ {0, 1}）
  fine_head（最细风格标签，辅助）
  year_head（50 年分箱，800–2050，高斯有序标签平滑 σ=1 箱；仅有年份样本计入）
损失：L = CE_style + λ_fine·CE_fine + λ_year·KL_year
优化：AdamW，lr ∈ {5e-4, 1e-3}，wd 0.05，cosine，warmup 3 epoch，最多 80 epoch，验证集早停（patience 10），dropout 0.1–0.2
网格（只在验证集选）：K ∈ {4, 8}，λ_fine ∈ {0.3, 1}，λ_year ∈ {0.3, 1}，τ ∈ {0, 1}，lr 两档
```
**ChronoArch-L**：在 ChronoArch-F 最优配置基础上，对 DINOv2 ViT-L 最后 4 个 block 的 qkv 加 LoRA（rank 8，alpha 16），224 分辨率在线前向（448 分支仍用缓存 token）；lr（LoRA）1e-4；梯度检查点；batch 16（不足用梯度累积到 64）；最多 25 epoch。

任务读出：
- WC6/WC14：style_head；
- RQ3a：主任务换为原作 vs 复兴；辅助最细标签中**删除 Gothic Revival / Romanesque Revival 等直接定义复兴组的标签**（防止目标泄漏）；
- RQ3b：主任务为年份，预测值 = year_head 分箱中心的概率加权平均；style/fine 头为辅助。
- 推理：原图 + 翻转视图概率平均；最终结果为 5 seed 集成。

### B3. 基线（同样的选择规则与集成方式）
| 编号 | 基线 |
|---|---|
| B-1 | 已有 4 个线性探针结果（复用） |
| B-2 | DINOv2 ViT-L/14 CLS 线性探针（224 与 448） |
| B-3 | DINOv2 ViT-L CLS + MLP（2 层，隐藏 1024；同增强、同 5 seed 集成） |
| B-4 | DINOv2-L + CLIP-L + SigLIP 2 特征拼接 + 线性探针 |
| B-5 | 同 B-4 拼接 + MLP（同增强、同集成） |
| B-6 | 三个线性探针概率平均（后期集成） |
| B-7 | DINOv2 ViT-L 全局注意力池化单尺度（224 token + 单个查询，常见的 attentive probing） |
| B-8 | DINOv2 ViT-B/14 部分微调（最后 2 个 block，224，20 epoch，验证集早停，5 seed）——“微调”强基线 |
| B-9 | DINOv2 ViT-L LoRA + 线性头（与 ChronoArch-L 相同 LoRA 设置，但没有三个模块）——用来对照 ChronoArch-L |

### B4. 消融（WC6、WC14、RQ3a Gothic、RQ3b，5 seed）
| 编号 | 配置 |
|---|---|
| A0 | 完整 ChronoArch-F |
| A1 | 去掉改进 1（只用 cls224） |
| A1a | 只用 224 token 注意力池化（去掉 448） |
| A1b | 只用 448（去掉 224） |
| A2 | 去掉改进 2（只用 DINOv2 分支） |
| A2a | 门控改为简单拼接 |
| A2b | 只融合 CLIP（去掉 SigLIP 2） |
| A3 | 去掉改进 3（只训练主任务） |
| A3a | 只去掉 fine_head |
| A3b | 只去掉 year_head |
| A4 | 三个改进全部去掉 |
| R1 | 骨干换为 ViT-B/14 |
| R2 | 去掉 logit adjustment |
| R3 | 去掉特征级增强 |
| R4 | 单 seed（不集成）、不做测试时翻转 |
| L0 | ChronoArch-L（完整） vs B-9 |

### B5. 评估与统计
1. 指标：WC6/WC14 平衡准确率与 macro-F1；RQ3a AUROC；RQ3b MAE、median AE、Spearman ρ。
2. ChronoArch（F 与 L）vs **验证集选出的最强基线**：配对教堂 bootstrap 差值 95% CI；分类再加 McNemar 精确检验；RQ3a 用 DeLong 检验；Holm 校正。
3. 消融：相对 A0 的差值与 95% CI。
4. 分析：WC6/WC14 逐类召回（重点 Renaissance、少数类）；按风格的平均门控权重 g；注意力图（描述性）。
5. RQ4：用 ChronoArch-F 重跑 E、R、K、R_keep 四个遮挡条件，与 DINO-B 线性探针对比。

### B6. 预计耗时（单张 RTX 5060 Laptop）
| 步骤 | 预计 |
|---|---|
| ViT-L 多视图 token 提取 + 遮挡条件 | 2–3 h |
| ChronoArch-F 网格搜索（验证集） | 2–3 h |
| 基线 B-2 至 B-9 | 4–6 h |
| ChronoArch-L | 2–3 h |
| 消融（16 组 × 5 seed × 4 任务） | 3–4 h |
| 统计、作图、汇总 | 1 h |
| **合计** | **约 14–20 h**（可分多次运行，按原方案的中断恢复方式继续） |

### B7. 输出
- `results_v2/main_comparison.csv`、`ablation.csv`、`significance.csv`、`selection_log.csv`、`per_class_recall.csv`、`gate_by_style.csv`、`rq4_chronoarch.csv`、`compute_cost.csv`；
- `figures_v2/model_architecture.pdf`（模型结构示意图）；
- `RESULTS_v2.md`：汇总全部结果，并单列“ChronoArch 未超过最强基线的任务/指标”一节（如有）。

完成后把 `RESULTS_v2.md` 的完整内容发给用户。
