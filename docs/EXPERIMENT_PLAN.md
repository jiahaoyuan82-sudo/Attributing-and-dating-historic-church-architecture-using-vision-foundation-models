# 实验方案：Attributing and dating historic church architecture using vision foundation models
## —— 基于 WikiChurches 公开数据集的文化遗产图像识别研究（目标期刊：npj Heritage Science）
> 论文工作标题：**Attributing and dating historic church architecture using vision foundation models**（利用视觉基础模型进行历史教堂建筑的风格归属与断代）

> **Part A** 给作者本人看（研究定位、科学问题、论文结构）；
> **Part B** 交给 Codex 执行（环境、数据、每个实验的步骤、预计耗时、输出、验收）。
> 预计总耗时：**约 10–11 小时**（含下载、调试、作图与桌面整理，略超出没有关系）。全程**不做任何模型训练/微调**，只用冻结特征 + 轻量分类器，单张 GPU 即可（无 GPU 也能跑，但特征提取会慢 3–5 倍）。

---

# Part A — 研究设计（给作者）

## A1. 数据集：WikiChurches

| 项目 | 内容 |
|---|---|
| 名称 | WikiChurches: A Fine-Grained Dataset of Architectural Styles with Real-World Challenges |
| 作者 | Björn Barz, Joachim Denzler（Friedrich Schiller University Jena） |
| 发表 | NeurIPS 2021 Datasets and Benchmarks Track；arXiv:2108.06959 |
| 下载 | Zenodo，v1.0 版本 DOI 10.5281/zenodo.5166987（https://zenodo.org/records/5166987）；原论文中引用的是 10.5281/zenodo.5166986，两者均指向该数据集 |
| 许可 | CC BY-SA 4.0（单张图像作者/许可见 `image_meta.json`） |
| 规模 | 9,485 张图像，9,346 座欧洲教堂；117 个风格标签，三级层级 |
| 专家标注 | 139 座教堂、631 个“风格诊断特征”边界框（Romanesque / Gothic / Renaissance / Baroque） |
| 元数据 | 原论文称建库时从 Wikidata 查询了风格、国家，以及“可得时”的坐标与建造年份；这些字段具体存放在哪个文件、覆盖率多少，**需以数据集 README 为准**（S1 会统计） |
| 官方子集 | WikiChurches-14 / -6 / -4 / -H；-6 与 -4 有官方 train/val/test 划分 |
| 已知基线 | WikiChurches-6 平衡准确率：ResNet-50 (IN-1k) 60.2%；TResNet-M (IN-21k) 69.5% |

## A2. 科学问题

1. **RQ1 基准**：冻结的视觉基础模型（DINOv2、CLIP、SigLIP 2）不经任何微调，能否在建筑风格识别上达到或超过原论文微调 CNN 的水平？——“冻结特征 + 线性分类器”几乎零训练成本，对资源有限的遗产机构最实用，这本身就是卖点。
2. **RQ2 错误的艺术史结构**：模型的混淆是否集中在年代相邻的风格之间？（Mantel 检验）
3. **RQ3 年代与真伪（核心）**：
   - 3a 能否区分中世纪原作哥特式与 19 世纪哥特复兴式？（含混杂因素控制）
   - 3b 能否仅凭照片估计建造年代？图像是否含有“风格标签之外”的年代信息？
4. **RQ4 是否依赖专家诊断特征**：遮挡专家标注区域 vs 遮挡等面积随机区域，比较置信度下降（因果检验）。

> 相关工作提醒：Choy & Leong（NUS-ISS 项目报告）已在 WikiChurches 上做过注意力与专家框的对齐分析；本文 RQ4 的区别在于因果遮挡并结合 RQ2/RQ3 解释，写作时需引用并说明差异。投稿前请再检索一次同类期刊论文。

## A3. 精简版的取舍（写进论文 Limitations）

- 不做全量微调 → 论文定位为“冻结基础模型的遗产学评估”，而非刷 SOTA；可在 Discussion 中指出微调是未来工作。
- 单次运行 + bootstrap 置信区间（线性探针本身方差很小，可接受）。
- 删去少样本曲线与跨国家泛化（可作为审稿返修时的补充实验）。

## A4. 建议图表

- **Fig. 1** 数据集概览：样例、类别分布、建造年份分布、专家框示例、研究流程图
- **Fig. 2** RQ1：4 个模型 × 3 种协议的平衡准确率（含原论文基线参考线）
- **Fig. 3** RQ2：按年代排序的混淆矩阵 + 混淆度 vs 年代差散点图
- **Fig. 4** RQ3：原作 vs 复兴 ROC；断代预测-真实散点图
- **Fig. 5** RQ4：专家框遮挡 vs 随机遮挡的概率下降分布 + 案例
- **Table 1** 主结果；**Table 2** 统计检验汇总

---

# Part B — 执行规范（交给 Codex）

## B0. 总原则

1. **耗时参考**：B2 给出每步的预计耗时，仅作规划参考，不是硬性上限；某步明显超出（如 2 倍以上）时在 `RUN_LOG.md` 说明原因即可，保证实验做完整、做正确优先。
2. **不训练深度网络**：所有 backbone 冻结；只训练 sklearn 分类器/回归器。
3. **防泄漏**：所有自建划分以教堂 ID（Wikidata QID）为单位；同一教堂的图像不得跨 train/test。写断言 `tests/test_no_leakage.py`。
4. **主指标**：平衡准确率；辅以 macro-F1。所有主结果给 95% 置信区间（按教堂 ID bootstrap，1,000 次）。
5. **固定 seed = 0**；超参数写入 `configs/config.yaml`；`pip freeze > env/requirements.lock.txt`。
6. **以数据为准**：先读数据集自带 `README.md`，实际格式与本文假设不符时按实际适配并记录；不得静默修改科学设计。
7. **如实报告负结果**。

## B1. 环境

```
Python 3.10+
torch>=2.2, torchvision, timm>=1.0, open_clip_torch, transformers（安装最新稳定版）
scikit-learn, scipy, pandas, numpy, imagehash, Pillow, matplotlib, seaborn, pyyaml, tqdm
```

> **GPU 要求（强制）**：所有深度模型的前向计算（S2 特征提取、S5 重压缩后重新提特征、S6 遮挡实验）**必须在 NVIDIA GPU 上运行**，使用 CUDA 版 PyTorch（Windows 上 `pip install torch` 默认是 CPU 版，必须按 pytorch.org 的说明安装对应 CUDA 版本）。开始 S2 前运行 `nvidia-smi` 与 `python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"`，结果写入 `env/gpu_info.txt`。`torch.cuda.is_available()` 为 False 时**立即停止并告知用户，禁止退回 CPU 运行**。代码中显式使用 `device="cuda"`，不要写成“有 GPU 用 GPU、否则用 CPU”。sklearn 的分类器/回归器与统计检验在 CPU 上运行属正常。

> 注：SigLIP 2 最初以 transformers 的 `v4.49.0-SigLIP-2` 标签单独发布。若当前版本加载 `google/siglip2-base-patch16-224` 时报错，按 transformers 官方说明安装该标签版本，并在日志中记录。

## B2. 预计耗时

| 步骤 | 内容 | 预计 |
|---|---|---|
| S0 | 环境 + 下载 + 解压 | 1.5 h |
| S1 | 解析元数据、划分、去重、数据统计 | 1.5 h |
| S2 | 4 个模型特征提取（含 RQ4 遮挡图的前向） | 1.5 h |
| S3 | RQ1 基准 | 0.5 h |
| S4 | RQ2 错误结构 | 0.5 h |
| S5 | RQ3 年代与真伪 | 1.5 h |
| S6 | RQ4 因果遮挡 | 1.0 h |
| S7 | 作图 + `RESULTS.md` | 1.0 h |
| S8 | 整理到桌面（实验数据、实验材料两个文件夹） | 0.5 h |
| — | 调试余量 | 1.0 h |
| **合计** | | **约 10.5 h** |

## B3. 仓库结构

```
project/
├── configs/config.yaml
├── data/raw/  data/processed/
├── splits/  features/  results/  figures/  tests/
├── src/ data.py models.py extract.py rq1.py rq2.py rq3.py rq4.py stats.py plots.py
├── run_all.sh
├── RUN_LOG.md
└── RESULTS.md
```

## S0. 下载（约 1.5 h）

从 https://zenodo.org/records/5166987 下载：`images.zip`(12.4 GB)、`churches.json`、`image_meta.json`、`labels.zip`、`building_parts.json`、`style_names.txt`、`parent_child_rel.txt`、`README.md`、`datasheet.pdf`、`LICENSE`。校验 md5。**不需要** `models.zip`。
下载期间可并行完成 S1 的元数据解析（元数据文件很小）。

**解压位置**：`images.zip` 解压到 `data/raw/images/`。解压前先检查所在磁盘剩余空间（解压后约再占 12–13 GB），不足时停下并告知用户。解压并核对图像数量无误后，`images.zip` 保留不删。

## S1. 数据准备（约 1.5 h）

1. 生成 `data/processed/metadata.parquet`，字段至少：
   `image_id, church_id, image_path, level1_style, level2_style(可空), all_styles, country, construction_year(可空), split_wc6, split_wc4`
2. 建造年份/国家：优先用 `churches.json` 自带字段。
   若缺失：用 Wikidata SPARQL 按 QID 补全（P571 建造/成立时间、P17 国家），结果缓存到 `data/processed/wikidata_cache.json` 并记录查询日期；若 Wikidata 无法访问，则只用已有字段，年份覆盖不足的部分按 S5 的备用规则处理。
3. WikiChurches-14 划分：按教堂 ID 分层，train 65% / val 10% / test 25%（每类 test ≥ 5），保存到 `splits/wc14.csv`。
4. 去重：pHash，汉明距离 ≤ 4 视为近重复；移除 test 中与 train 重复的图像，报告数量。
5. 输出 `results/dataset_stats.csv`：各子集各类数量、年份覆盖率（按类别）、国家 Top-10。

## S2. 特征提取（约 1.5 h）

| 代号 | 模型 | 来源 |
|---|---|---|
| R50 | ResNet-50 ImageNet-1k（基线） | timm `resnet50.a1_in1k` |
| DINO-B | DINOv2 ViT-B/14 | HF `facebook/dinov2-base` |
| CLIP-L | CLIP ViT-L/14 (OpenAI) | open_clip `ViT-L-14`, `openai` |
| SIG2-B | SigLIP 2 ViT-B/16 | HF `google/siglip2-base-patch16-224` |

- 输入 224×224（短边缩放到 224 后中心裁剪）；fp16；batch 64；DataLoader `num_workers≥8`（大图解码是瓶颈）。
- **提速**：先把全部图像一次性缩放到短边 256 并缓存为 JPEG（`data/processed/img256/`），之后所有模型都读缓存。
- 特征：CNN 用全局平均池化；ViT 用 CLS；CLIP/SigLIP 用投影后的图像嵌入；全部 L2 归一化，存 `features/{model}.npy` 与对应 `image_id` 索引。
- **备用方案**：若某权重下载失败，CLIP-L 换 `ViT-B-16`，SigLIP 2 换 `google/siglip-base-patch16-224`，并记录。

## S3. RQ1 基准（约 0.5 h）

任务：WikiChurches-6（官方划分，主）与 WikiChurches-14（S1 划分）。

1. **零样本**（CLIP-L、SIG2-B）：5 个提示模板取平均文本嵌入，例如
   `"a photo of a {style} church"`, `"a church built in {style} style"`, `"{style} architecture"`, `"the facade of a {style} church"`, `"a historic {style} church building"`。
2. **kNN**：余弦距离，k ∈ {5, 10, 20, 50}，val 选 k。
3. **线性探针**：`LogisticRegression(class_weight="balanced", max_iter=3000)`，C ∈ {0.01, 0.1, 1, 10}，val 选 C。
4. 输出 `results/rq1_main.csv`（model, protocol, task, bal_acc, macro_f1, ci_low, ci_high）与逐类召回；保存每个模型线性探针在两个任务 test 上的预测概率（后续 RQ2 用）。
5. 统计：最佳模型 vs R50（同为线性探针）做配对 bootstrap 差值 CI + McNemar 检验。
6. 注意：原论文 60.2% 是**微调**结果，在表中作为“参考线”列出，不与线性探针做显著性检验。

## S4. RQ2 错误结构（约 0.5 h）

在 WikiChurches-14 test 上，使用最佳线性探针模型：

1. 每个风格的代表年份 = 该类有年份教堂的**中位数**（需 ≥ 10 个有效年份，否则剔除该类并记录）。**不要手写年代。**
2. 行归一化混淆矩阵 → 对称化 `(C + Cᵀ)/2` → 取非对角元素。
3. Mantel 检验：混淆度 vs 年代距离 `|year_i − year_j|`，Spearman，10,000 次置换。
4. 同样检验：类中心余弦距离 vs 年代距离（表征空间是否含“时间轴”）。
5. 输出 `results/rq2_mantel.csv`；混淆矩阵按代表年份排序作图；导出 Renaissance→Baroque 误判缩略图拼图（≤ 24 张）。

## S5. RQ3 年代与真伪（约 1.5 h）

### 3a 原作 vs 复兴
1. 构造数据：
   - Gothic 原作：level1=Gothic、非 Gothic Revival、年份 ≤ 1550；
   - Gothic 复兴：标签含 “Gothic Revival”（或数据中等价名称），或 level1=Gothic 且年份 ≥ 1800；
   - 标签与年份冲突者剔除并计数。
   - **备用方案**：若任一组 < 50 张，则只用标签（不用年份）定义两组；若仍 < 50，则改做 Romanesque vs Romanesque Revival；都不行就记录并跳过 3a。
2. 按教堂 ID 分层 5 折 CV；4 个模型的特征 + 线性探针（同 S3）。
3. 指标：AUROC（主）、平衡准确率；均值 ± 折间标准差。
4. **混杂控制（必须做，三项都便宜）**：
   - (i) 仅用“图像宽、高、文件大小、EXIF 拍摄年份（若有）”训练逻辑回归，报告其 AUROC，作为“非建筑线索”上限；
   - (ii) 只对最佳模型：把这部分图像统一缩放到 224 并以 JPEG quality=75 重压缩后**重新提特征**，重复主实验；
   - (iii) 按国家分层：报告样本数 ≥ 20 的国家内 AUROC 的加权平均。
   - 结论只有在三项控制后仍成立时才可写“模型能辨识复兴式”。

### 3b 视觉断代
1. 数据：WikiChurches-14 中年份在 800–2020 之间的图像。**备用方案**：若有效样本 < 500，只报告描述性结果并说明。
2. 按教堂 ID 5 折 CV；Ridge 回归，alpha ∈ {0.1, 1, 10, 100}（内层 3 折选择）。
3. 基线：B0 全体中位数；B1 已知真实风格时预测该风格中位年份；主模型（仅图像）；组合（图像 + 风格 one-hot）。
4. 关键分析：**风格内偏相关**——控制风格后，预测年份与真实年份的 Spearman 相关；显著 > 0 说明图像含有风格之外的年代信息。
5. 指标：MAE（年）、中位绝对误差、Spearman ρ，bootstrap 95% CI。

输出：`results/rq3a_authenticity.csv`、`results/rq3a_confound_controls.csv`、`results/rq3b_dating.csv`。

## S6. RQ4 因果遮挡（约 1.0 h）

数据：WikiChurches-4 中有专家框的 139 张图（`building_parts.json`）。分类器：在 WikiChurches-4 官方 train 上训练的线性探针（**若这 139 张出现在 train 中，先移除再训练**，记录数量）。模型：最佳模型 + R50。

1. 条件 **E**：所有专家框用图像均值色填充并高斯模糊；
2. 条件 **R**：放置与专家框数量、尺寸完全相同的随机框（与专家框重叠 ≤ 10%，位于图像中心 80% 区域内），每图 **20 次**随机；
3. 条件 **K**：只保留专家框区域，其余遮挡；
4. 每个条件重新前向提特征 → 线性探针 → 真实类别概率；
5. 指标：Δp = p(原图) − p(遮挡后)；top-1 翻转率；
6. 检验：每图 Δp_E vs 平均 Δp_R 的配对 Wilcoxon 符号秩检验，报告效应量 r；按风格、按特征描述标签分组报告（原论文说明每个框带有描述其结构的标签，如 “Tracery”、“Round Arch Window”；只统计样本 ≥ 10 的标签）。
7. 额外前向次数约 139 × 22 × 2 ≈ 6,000 次，应在 15 分钟内完成。
8. 不做 Grad-CAM / attention rollout（节省时间）；挑 8 个案例图（原图 + 专家框 + Δp）用于 Fig. 5。

输出：`results/rq4_occlusion.csv`、`results/rq4_by_group.csv`。

## S7. 作图与汇总（约 1.0 h）

图（pdf + png, 300 dpi）：`fig1_dataset`、`fig2_rq1`、`fig3_confusion_time`、`fig3_mantel_scatter`、`fig4_roc`、`fig4_dating`、`fig5_occlusion`、`fig5_cases`。

`RESULTS.md` 必须包含：
- [ ] 数据统计与剔除记录
- [ ] Table 1：RQ1 主结果（± 95% CI），附原论文 60.2% / 69.5% 参考
- [ ] RQ2：Mantel r、p（混淆度与表征空间两个版本）
- [ ] RQ3a：主结果 + 三项混杂控制，并明确写出结论是否在控制后成立
- [ ] RQ3b：MAE 与各基线、风格内偏相关
- [ ] RQ4：Δp_E vs Δp_R 检验、分组结果
- [ ] 统计检验汇总表（检验、统计量、p、效应量）
- [ ] 每步实际耗时、启用了哪些备用方案
- [ ] 负结果/异常一节
- [ ] 桌面两个文件夹（`…_实验数据`、`…_实验材料`）的路径、大小与文件数量（见 S8）

最后检查：`run_all.sh` 能一键复现；`tests/test_no_leakage.py` 通过；图中数字与 CSV 一致。

## S8. 整理到桌面（全部实验完成后执行，约 0.5 h）

实验全部完成、`RESULTS.md` 写好之后，在**当前用户的桌面**上创建以下两个文件夹（名称必须完全一致）：

```
Attributing and dating historic church architecture using vision foundation models_实验数据
Attributing and dating historic church architecture using vision foundation models_实验材料
```

**桌面路径确定方法**：
- macOS / Linux：`~/Desktop`；若不存在，再检查 `~/桌面`（中文 Linux 系统）；
- Windows：`%USERPROFILE%\Desktop`；若桌面被 OneDrive 接管，则为 `%USERPROFILE%\OneDrive\Desktop`（或 `OneDrive\桌面`）；
- 以上都找不到时，不要自行猜测或新建“Desktop”，在 `RUN_LOG.md` 中记录并告知用户。

### 文件夹 1：`…_实验数据`（实验产生的所有结果）

```
…_实验数据/
├── README.md            # 本文件夹说明：各子文件夹内容、生成时间、对应的实验（RQ1–RQ4）
├── RESULTS.md           # 最终结果汇总
├── RUN_LOG.md           # 完整运行日志
├── results/             # 全部 CSV / JSON 结果
├── figures/             # 全部图（pdf + png）
├── splits/              # 全部数据划分文件
├── features/            # 缓存的冻结特征（.npy）及 image_id 索引
├── configs/             # 实验配置
├── env/                 # requirements.lock.txt、GPU/CUDA 信息
└── code/                # src/、tests/、run_all.sh 的副本（保证可复现）
```

### 文件夹 2：`…_实验材料`（实验使用的数据集）

```
…_实验材料/
├── README.md            # 数据集名称、作者、来源 DOI、许可证、下载日期、md5 校验结果、引用格式
├── WikiChurches_raw/    # 从 Zenodo 下载的原始文件（images.zip、churches.json、image_meta.json、
│                        #   labels.zip、building_parts.json、style_names.txt、parent_child_rel.txt、
│                        #   README.md、datasheet.pdf、LICENSE）
└── processed/           # metadata.parquet、wikidata_cache.json（如有）、去重报告、img256 缩放缓存
```

`…_实验材料/README.md` 必须写明：
- 数据集：WikiChurches（Barz & Denzler, NeurIPS 2021 Track on Datasets and Benchmarks, arXiv:2108.06959）；
- 来源：https://zenodo.org/records/5166987（DOI 10.5281/zenodo.5166987）；
- 许可：CC BY-SA 4.0，单张图像的作者与许可见 `image_meta.json`；再分发时需署名并以相同许可共享；
- 下载日期、每个文件的 md5 及是否校验通过。

**执行要求**（目标：实验结束后，所有文件都在桌面这两个文件夹里，不在别处留副本）：
1. **全部用移动，不用复制**：实验数据类文件（`RESULTS.md`、`RUN_LOG.md`、`results/`、`figures/`、`splits/`、`features/`、`configs/`、`env/`、`src/`、`tests/`、`run_all.sh`）移动到 `…_实验数据/`（代码放在其中的 `code/` 下）；数据集类文件（`data/raw/` 含 `images.zip` 与解压后的 `images/`，以及 `data/processed/`）移动到 `…_实验材料/`。移动前先确认所有实验已完成、`RESULTS.md` 已写好。
2. 移动后更新 `…_实验数据/configs/config.yaml` 中的数据路径，指向 `…_实验材料/` 中的新位置，保证在 `…_实验数据/code/` 中运行 `run_all.sh` 仍可复现；并在 `…_实验数据/README.md` 中写明如何复现。
3. 原项目文件夹中只留一个 `MOVED_TO_DESKTOP.txt`，写明两个桌面文件夹的完整路径。
4. 移动前检查桌面所在磁盘剩余空间；若项目与桌面不在同一磁盘，移动等于跨盘复制，需确认空间足够，不足则停下并告知用户，**不要删除任何文件**。
5. 完成后逐个比对文件数量与总大小；对原始数据文件重新校验 md5，结果写进 `…_实验材料/README.md`。
6. 在 `RUN_LOG.md`（已移入 `…_实验数据/`）最后记录两个文件夹的完整路径、大小和文件数量。

## B4. 数据与代码可用性（供论文使用）

- 数据：WikiChurches，Zenodo DOI 10.5281/zenodo.5166987，CC BY-SA 4.0。
- 代码与划分文件：建议公开到 GitHub 并在 Zenodo 存档获得 DOI。

## 参考文献（核心；1、2、3、5、6 已核实来源，4 为广为人知的文献，投稿前请用 Google Scholar 导出标准格式）

1. Barz, B. & Denzler, J. WikiChurches: A fine-grained dataset of architectural styles with real-world challenges. NeurIPS 2021 Track on Datasets and Benchmarks. arXiv:2108.06959.
2. Llamas, J. et al. Classification of architectural heritage images using deep learning techniques. Applied Sciences 7, 992 (2017).
3. Oquab, M. et al. DINOv2: Learning robust visual features without supervision. Transactions on Machine Learning Research (2024). arXiv:2304.07193.
4. Radford, A. et al. Learning transferable visual models from natural language supervision. ICML (2021).
5. Tschannen, M. et al. SigLIP 2: Multilingual vision-language encoders with improved semantic understanding, localization, and dense features. arXiv:2502.14786 (2025).
6. Choy, D. & Leong, K. M. Do self-supervised vision models learn what experts see? NUS-ISS 项目报告（非同行评审），https://desmondchoy.github.io/projects/ssl-vit-models
