# WikiChurches 实验结果

本报告由 `code/src/write_results.py` 从保存的 CSV、JSON、特征提取记录和元数据生成。四个冻结模型均使用原方案指定权重；所有深度模型前向记录为 CUDA。固定 seed=0；自建划分按教堂 ID 隔离。原论文指标只作为外部参考。

## 数据与排除记录

| 子集 | 图像 | 教堂 | 有建造年份图像 | 年份覆盖率 |
| --- | --- | --- | --- | --- |
| all | 9485 | 9346 | 4086 | 43.0785% |
| wc14 | 8931 | 8800 | 3713 | 41.5743% |
| wc6 | 8488 | 8362 | 3473 | 40.9166% |
| wc4 | 7526 | 7418 | 3050 | 40.5262% |

原始压缩包经 MD5 核验并解压出 9,502 张 JPEG；其中 17 张没有教堂元数据，分析元数据含 9,485 张。WC14 训练集与测试集之间 pHash 距离 ≤4 的近重复配对为 0，测试图剔除 0 张。另有 28 张压缩包图像没有 `image_meta.json` 记录。

RQ3a：Gothic 原作 700 张，混合规则复兴 1637 张，严格年份复兴 869 张；Gothic 标签–年份冲突 49 张。Romanesque 原作 392 张，混合规则复兴 510 张，严格年份复兴 187 张；Romanesque 标签–年份冲突 3 张。Romanesque 混合复兴中的 494 张带 Revival 标签，另 16 张符合 level1 与 ≥1800 年的年份条件；此前预检的 494 是仅标签计数。

RQ3b：WC14 中 800–2020 年的主样本 3,668 张、3,605 座教堂；冲突剔除敏感性分析移除 81 张，留下 3,587 张、3,525 座。全部元数据上的排除清单有 155 张；只有落在 RQ3b 主样本中的 81 张影响本项比较。排除规则与每类阈值见 `RUN_LOG.md`、`results/rq3b_early_year_thresholds.json` 和逐图清单。

## Table 1 · RQ1 风格归属

平衡准确率为主指标；95% CI 以教堂 ID 分层 bootstrap 1,000 次。macro-F1 为辅助指标。

| 任务 | 模型 | 协议 | 平衡准确率 [95% CI] | macro-F1 |
| --- | --- | --- | --- | --- |
| WC6 | R50 | knn | 43.46% [41.51%, 45.52%] | 44.22% |
| WC6 | R50 | linear_probe | 59.67% [56.80%, 62.70%] | 56.68% |
| WC6 | DINO-B | knn | 63.60% [61.08%, 66.11%] | 66.53% |
| WC6 | DINO-B | linear_probe | 74.58% [72.11%, 77.27%] | 72.57% |
| WC6 | CLIP-L | knn | 52.93% [50.49%, 55.40%] | 56.22% |
| WC6 | CLIP-L | linear_probe | 69.16% [66.67%, 71.91%] | 67.39% |
| WC6 | CLIP-L | zero_shot | 60.04% [57.38%, 62.83%] | 59.36% |
| WC6 | SIG2-B | knn | 51.52% [48.75%, 54.36%] | 55.49% |
| WC6 | SIG2-B | linear_probe | 67.71% [65.27%, 70.38%] | 65.02% |
| WC6 | SIG2-B | zero_shot | 54.08% [51.59%, 56.85%] | 53.32% |
| WC14 | R50 | knn | 23.43% [20.89%, 26.00%] | 24.85% |
| WC14 | R50 | linear_probe | 40.40% [35.20%, 45.83%] | 15.58% |
| WC14 | DINO-B | knn | 51.73% [47.72%, 56.14%] | 53.38% |
| WC14 | DINO-B | linear_probe | 73.96% [69.15%, 78.77%] | 56.41% |
| WC14 | CLIP-L | knn | 43.53% [39.27%, 47.76%] | 45.62% |
| WC14 | CLIP-L | linear_probe | 68.62% [64.14%, 73.41%] | 48.27% |
| WC14 | CLIP-L | zero_shot | 47.33% [42.67%, 51.95%] | 35.11% |
| WC14 | SIG2-B | knn | 36.56% [32.15%, 41.22%] | 41.06% |
| WC14 | SIG2-B | linear_probe | 64.31% [59.16%, 70.16%] | 50.90% |
| WC14 | SIG2-B | zero_shot | 40.32% [35.04%, 45.75%] | 30.51% |

原论文 WC6 的微调参考线：ResNet-50 **60.2%**、TResNet-M **69.5%**。这些是已发表数值，不是本次冻结特征的运行结果，也未用于显著性检验。

以 WC6 验证集平衡准确率选择的最佳线性探针为 **DINO-B**（验证集 76.82%）。与 R50 同协议的 WC6 测试集配对平衡准确率差为 14.91%，95% CI [12.10%, 17.63%]；McNemar 精确 p=7.64e-38，仅最佳模型正确/仅 R50 正确的图像数分别为 349/88。逐类召回与测试概率见 `results/rq1_per_class.csv` 及 `rq1_prob_*.npz`。

## RQ2 · 错误与年代结构

WC14 中 12 个风格达到至少 10 座有建造年份教堂的门槛；排除 2 类。代表年份从教堂记录的中位数计算；混淆矩阵先按全部 14 类行归一化，再选取可检验类别。

| Mantel 检验 | Spearman r | 双侧置换 p | 置换次数 |
| --- | --- | --- | --- |
| confusion_vs_year_distance | 0.0504 | 0.787221 | 10000 |
| center_distance_vs_year_distance | -0.0184 | 0.906209 | 10000 |

两项检验均未显示显著年代结构。Renaissance→Baroque 误判缩略图拼图包含 7 张测试图；完整图像 ID 与成对风格数据分别见 `results/rq2_renaissance_to_baroque_images.csv` 和 `results/rq2_pairs.csv`。

## RQ3a · 原作与复兴

两组风格均按混合规则作主分析，并按严格年份规则作敏感性分析。表内 AUROC、平衡准确率为按教堂分组的 5 折均值 ± 折间标准差；CI 是全部折外预测按教堂 ID bootstrap 1,000 次所得。

| 风格 | 规则 | 模型 | 图像 | AUROC 均值±SD | 折外 AUROC 95% CI | 平衡准确率均值±SD |
| --- | --- | --- | --- | --- | --- | --- |
| Gothic | mixed | R50 | 2337 | 0.814 ± 0.026 | [0.794, 0.834] | 73.54% ± 3.36% |
| Gothic | mixed | DINO-B | 2337 | 0.949 ± 0.009 | [0.938, 0.955] | 88.21% ± 1.42% |
| Gothic | mixed | CLIP-L | 2337 | 0.936 ± 0.012 | [0.924, 0.945] | 85.65% ± 1.64% |
| Gothic | mixed | SIG2-B | 2337 | 0.912 ± 0.016 | [0.897, 0.922] | 82.12% ± 2.78% |
| Gothic | strict_year | R50 | 1569 | 0.805 ± 0.035 | [0.769, 0.814] | 72.75% ± 4.13% |
| Gothic | strict_year | DINO-B | 1569 | 0.945 ± 0.005 | [0.932, 0.954] | 87.24% ± 1.39% |
| Gothic | strict_year | CLIP-L | 1569 | 0.939 ± 0.014 | [0.926, 0.948] | 86.59% ± 0.57% |
| Gothic | strict_year | SIG2-B | 1569 | 0.917 ± 0.011 | [0.900, 0.928] | 83.80% ± 1.20% |
| Romanesque | mixed | R50 | 902 | 0.862 ± 0.029 | [0.836, 0.884] | 77.81% ± 2.67% |
| Romanesque | mixed | DINO-B | 902 | 0.963 ± 0.016 | [0.946, 0.972] | 89.81% ± 3.33% |
| Romanesque | mixed | CLIP-L | 902 | 0.954 ± 0.012 | [0.942, 0.967] | 90.01% ± 1.11% |
| Romanesque | mixed | SIG2-B | 902 | 0.946 ± 0.016 | [0.928, 0.957] | 87.12% ± 2.24% |
| Romanesque | strict_year | R50 | 579 | 0.851 ± 0.028 | [0.806, 0.874] | 73.43% ± 3.71% |
| Romanesque | strict_year | DINO-B | 579 | 0.951 ± 0.014 | [0.927, 0.968] | 89.92% ± 2.59% |
| Romanesque | strict_year | CLIP-L | 579 | 0.944 ± 0.019 | [0.919, 0.963] | 88.51% ± 3.14% |
| Romanesque | strict_year | SIG2-B | 579 | 0.935 ± 0.022 | [0.904, 0.953] | 87.19% ± 1.60% |

三项混杂控制均按原方案执行：仅图像属性（宽、高、文件大小、可得的 EXIF 拍摄年份）、统一 224 像素后 JPEG quality=75 并重新 GPU 提特征、以及每国至少 20 张且两类均存在的国家内 AUROC 加权平均。

| 风格 | 控制 | AUROC | 折间 SD | 覆盖图像 |
| --- | --- | --- | --- | --- |
| Gothic | image_properties_only | 0.599 | 0.031 | 2337 |
| Gothic | recompressed_q75 | 0.942 | 0.011 | 2337 |
| Romanesque | image_properties_only | 0.658 | 0.029 | 902 |
| Romanesque | recompressed_q75 | 0.956 | 0.020 | 902 |
| Gothic | country_weighted_auroc | 0.934 | NA | 2218 |
| Romanesque | country_weighted_auroc | 0.905 | NA | 775 |

DINO-B 的两组主分析 AUROC 在重压缩控制后仍保持较高，且高于图像属性控制；两组国家内加权 AUROC 也保持较高。因此在这些已执行的控制范围内，模型辨识复兴式的结论成立。国家细项见 `results/rq3a_countries_*.csv`；这些控制不能排除所有未测量混杂。

## RQ3b · 视觉断代与冲突敏感性

5 折教堂分组 CV；Ridge alpha 在每个训练折内用 3 折选择。B0 为训练折全体中位数，B1 为训练折对应真实风格的中位数。MAE、median AE、Spearman ρ 的 95% CI 均按教堂 ID bootstrap 1,000 次。

| 样本 | 模型 | 方法 | MAE 年 [95% CI] | median AE 年 | Spearman ρ |
| --- | --- | --- | --- | --- | --- |
| main | baseline | B0_global_median | 249.00 [242.62, 255.66] | 172.00 | -0.013 |
| main | baseline | B1_true_style_median | 193.82 [186.30, 201.73] | 70.00 | 0.368 |
| main | R50 | image_only | 205.46 [200.65, 210.28] | 178.09 | 0.518 |
| main | R50 | image_plus_style | 184.93 [180.07, 189.46] | 153.77 | 0.565 |
| main | DINO-B | image_only | 161.50 [157.11, 165.81] | 125.39 | 0.715 |
| main | DINO-B | image_plus_style | 149.32 [144.75, 153.55] | 115.11 | 0.728 |
| main | CLIP-L | image_only | 162.29 [157.50, 166.77] | 126.45 | 0.710 |
| main | CLIP-L | image_plus_style | 149.57 [144.88, 153.95] | 113.95 | 0.717 |
| main | SIG2-B | image_only | 171.81 [167.15, 176.40] | 137.26 | 0.670 |
| main | SIG2-B | image_plus_style | 157.70 [153.09, 162.07] | 121.88 | 0.683 |
| conflict_excluded | baseline | B0_global_median | 242.16 [236.22, 249.02] | 162.00 | -0.022 |
| conflict_excluded | baseline | B1_true_style_median | 185.43 [178.19, 192.77] | 64.00 | 0.376 |
| conflict_excluded | R50 | image_only | 198.59 [193.88, 203.73] | 168.82 | 0.519 |
| conflict_excluded | R50 | image_plus_style | 176.62 [171.75, 181.37] | 145.92 | 0.570 |
| conflict_excluded | DINO-B | image_only | 152.42 [148.07, 156.77] | 118.67 | 0.725 |
| conflict_excluded | DINO-B | image_plus_style | 139.67 [135.79, 143.60] | 108.73 | 0.736 |
| conflict_excluded | CLIP-L | image_only | 154.67 [150.25, 159.00] | 120.68 | 0.717 |
| conflict_excluded | CLIP-L | image_plus_style | 141.12 [137.22, 145.15] | 108.56 | 0.723 |
| conflict_excluded | SIG2-B | image_only | 164.76 [160.35, 169.24] | 129.68 | 0.676 |
| conflict_excluded | SIG2-B | image_plus_style | 149.80 [145.61, 154.01] | 116.12 | 0.688 |

| 样本 | 模型 | 控制风格后的偏 Spearman ρ | 教堂 bootstrap 95% CI | 近似 p |
| --- | --- | --- | --- | --- |
| main | DINO-B | 0.653 | [0.632, 0.674] | p < 1e-16 |
| conflict_excluded | DINO-B | 0.665 | [0.645, 0.686] | p < 1e-16 |

DINO-B 仅图像模型的 MAE 从主样本 161.50 年变为冲突剔除样本 152.42 年，样本集合不同，差值不作配对因果解释。两套样本的风格内偏相关均为正，教堂 bootstrap CI 均未跨 0。原始折外预测见 `results/rq3b_oof.csv`。

## RQ4 · 专家诊断区域遮挡

共 139 张专家标注图；其中 8 张没有官方 WC4 单标签，但有专家给出的 WC4 风格标签，真实类别概率按专家标签集合求和。为避免泄漏，训练集中移除 90 张、验证集中移除 8 张来自专家标注教堂的图。每图做 20 次等数量、等尺寸随机框，共 5560 条模型×图像×随机重复记录。

| 模型 | Δp_E | 平均 Δp_R | E−R [95% CI] | Wilcoxon p | 效应量 r | E/R/K top-1 翻转率 |
| --- | --- | --- | --- | --- | --- | --- |
| DINO-B | 0.0008 | 0.0010 | -0.0003 [-0.0109, 0.0090] | 0.1812 | 0.113 | 3.60%/1.87%/23.74% |
| R50 | 0.0257 | -0.0056 | 0.0314 [0.0155, 0.0478] | 0.0003513 | 0.303 | 15.83%/8.63%/64.03% |

DINO-B 的专家框优势不显著，且平均差接近零；R50 的专家框遮挡导致更大的概率下降。R50 与 DINO-B 的差异提示两者利用诊断区域的方式不同，不能把 R50 的显著结果套用到最佳模型。

| 模型 | 风格 | 图像 | 平均 Δp_E−Δp_R |
| --- | --- | --- | --- |
| DINO-B | Gothic | 49 | 0.0092 |
| DINO-B | Renaissance | 22 | -0.0687 |
| DINO-B | Romanesque | 54 | 0.0182 |
| DINO-B | Baroque | 17 | 0.0036 |
| R50 | Gothic | 49 | 0.0651 |
| R50 | Renaissance | 22 | -0.0103 |
| R50 | Romanesque | 54 | 0.0228 |
| R50 | Baroque | 17 | 0.0263 |

按结构描述标签分组只保留至少 10 张图的标签；完整的 22 个合格标签×两模型结果见 `results/rq4_by_group.csv`。这些分组为描述性分析，未另行做多重检验。

## Table 2 · 统计检验汇总

| 比较 | 检验 | 统计量/效应量 | 95% CI | p |
| --- | --- | --- | --- | --- |
| RQ1 最佳模型−R50 | 配对教堂 bootstrap | 差 14.91% | [12.10%, 17.63%] | — |
| RQ1 最佳模型 vs R50 | McNemar 精确检验 | 349/88 | — | 7.64e-38 |
| RQ2 confusion_vs_year_distance | Mantel, 10,000 置换 | 0.0504 | — | 0.787221 |
| RQ2 center_distance_vs_year_distance | Mantel, 10,000 置换 | -0.0184 | — | 0.906209 |
| RQ3b 偏相关 main | 控制风格后 Spearman | 0.653 | [0.632, 0.674] | p < 1e-16 |
| RQ3b 偏相关 conflict_excluded | 控制风格后 Spearman | 0.665 | [0.645, 0.686] | p < 1e-16 |
| RQ4 DINO-B E vs R | 配对 Wilcoxon | r=0.113 | 均值差 [-0.0109, 0.0090] | 0.1812 |
| RQ4 R50 E vs R | 配对 Wilcoxon | r=0.303 | 均值差 [0.0155, 0.0478] | 0.0003513 |

## 运行耗时与备用方案

| 步骤 | 有记录的实际耗时 | 说明 |
| --- | --- | --- |
| S0 | 下载约 1 小时 25 分 | 原日志约 20:15–21:40，解压另计；初次会话未记录完整墙钟时间 |
| S1 | 未单独记录 | 初次会话完成，不能补造耗时 |
| S2 R50 | 138.36 秒 | CUDA |
| S2 DINO-B | 234.33 秒 | CUDA |
| S2 CLIP-L | 172.59 秒 | CUDA；QuickGELU 配置修正后重提 |
| S2 SIG2-B | 90.76 秒 | CUDA；权重下载与续传另计 |
| S3 | 121.61 秒 | RQ1 |
| S4 | 6.36 秒 | RQ2 |
| S5 | 464.1 秒 | 含两组重压缩重新提特征及敏感性分析 |
| S6 | 72.5 秒 | 修正效应量后的完整重跑；首次运行 71.7 秒 |
| S7 图 | 15.62 秒 | 最终九组 PNG/PDF；报告生成另计 |

没有启用模型替换、实验删减或主指标更换。RQ3a 两组风格均达到样本门槛，混合规则及严格年份规则均执行；RQ3b 冲突剔除分析也已执行。

## 负结果、异常与限制

- RQ2 的混淆度和类中心距离都未显示显著的年代距离关系。
- 最佳模型 DINO-B 的 RQ4 专家框相对随机框效果未达显著；其平均差接近零。
- 建造年份仅覆盖部分图像，且可能代表重建、扩建或 Wikidata 记录口径；视觉断代结果需按这个标签含义解释。
- RQ3a 的图像属性控制与重压缩、国家分层控制不能排除全部地理、摄影和建筑状态混杂。
- SigLIP 2 权重下载受到网络沙箱、Xet 停滞与 Windows 长路径影响；最终文件通过大小和 SHA-256 校验。CLIP-L 的 OpenAI 权重第一次加载出现 QuickGELU 配置警告，相关输出仅保留审计，正式结果使用匹配激活函数的重提特征。
- RQ4 真实类别概率对 8 张无官方 WC4 单标签图使用专家标签集合求和；详情见 `RUN_LOG.md`。

## 图和复现

`figures/` 含计划规定的八组图和一组 Renaissance→Baroque 误判拼图，每组 PNG 与 PDF。`configs/config.yaml` 记录超参数；`env/requirements.lock.txt` 和 `env/gpu_info.txt` 记录环境。`code/run_all.sh`（Windows 另有 `code/run_all.ps1`）与数据文件夹一起用于复现。`tests/test_no_leakage.py` 已通过。

## 补充分析

### RQ3b · 最细风格标签基线 B2

B2 使用每图唯一的 `all_styles` 最细标签 ID；Gothic Revival 等子类单独求训练折中位年份。沿用原有 5 折教堂分组划分，测试折中未见的子类回退到该训练折全体中位数。95% CI 均为教堂 ID bootstrap 1,000 次。

| 样本 | 模型 | 方法 | 图像 | MAE 年 [95% CI] | median AE 年 [95% CI] | Spearman ρ [95% CI] |
| --- | --- | --- | --- | --- | --- | --- |
| main | baseline | B1_true_style_median | 3668 | 193.82 [186.30, 201.73] | 70.00 [66.00, 73.00] | 0.368 [0.334, 0.400] |
| main | baseline | B2_fine_style_median | 3668 | 81.49 [77.25, 85.82] | 37.00 [34.99, 40.00] | 0.857 [0.840, 0.872] |
| main | DINO-B | image_only | 3668 | 161.50 [157.11, 165.81] | 125.39 [121.04, 130.74] | 0.715 [0.696, 0.732] |
| conflict_excluded | baseline | B1_true_style_median | 3587 | 185.43 [178.19, 192.77] | 64.00 [62.00, 68.00] | 0.376 [0.340, 0.410] |
| conflict_excluded | baseline | B2_fine_style_median | 3587 | 70.87 [67.63, 74.33] | 35.00 [32.00, 37.00] | 0.881 [0.866, 0.896] |
| conflict_excluded | DINO-B | image_only | 3587 | 152.42 [148.07, 156.77] | 118.67 [112.93, 122.99] | 0.725 [0.708, 0.742] |

B2 的最细标签数与未见标签回退数：main 52 类、6 张回退；conflict_excluded 52 类、10 张回退。逐图折外预测见 `results/rq3b_b2_oof.csv`。
B2 使用真实的细粒度风格标签信息，DINO-B image_only 仅使用图像；两者的输入信息不同。

### RQ4 · 专家框面积及随机区域保留 R_keep

对 139 张标注图，在 256 像素图上按原专家框坐标绘制并集掩膜；并集面积占图像比例的均值 7.69%、中位数 5.18%、最小值 0.46%、最大值 31.67%；四分位数 2.29%–10.65%。逐图面积及未去重的框面积之和见 `results/rq4_expert_box_area.csv`。

| 并集面积比例区间 | 图像数 |
| --- | --- |
| 0.00%–5.00% | 68 |
| 5.00%–10.00% | 30 |
| 10.00%–20.00% | 31 |
| 20.00%–30.00% | 7 |
| 30.00%–40.00% | 3 |
| 40.00%–50.00% | 0 |
| 50.00%–60.00% | 0 |
| 60.00%–70.00% | 0 |
| 70.00%–80.00% | 0 |
| 80.00%–90.00% | 0 |
| 90.00%–100.00% | 0 |

R_keep 每图随机放置与专家框等数量、等尺寸的区域 20 次，保留框内像素并遮挡其余区域；位置仍限中心 80%，与专家框 IoU ≤0.1。K 保留专家框内像素。真实类别概率按专家风格集合求和。top-1 保持正确率以原图预测正确的图为分母；R_keep 为每图 20 次的正确比例后再按图平均。配对 Wilcoxon 比较逐图 K 与 R_keep 均值，差值 CI 按教堂 bootstrap 1,000 次。

| 模型 | 指标 | 图像 | K | R_keep | K−R_keep [95% CI] | 配对 Wilcoxon p |
| --- | --- | --- | --- | --- | --- | --- |
| DINO-B | true_class_probability | 139 | 0.6094 | 0.5144 | 0.0951 [0.0631, 0.1281] | 8.57e-07 |
| DINO-B | top1_retention | 120 | 79.17% | 64.67% | 14.50% [8.58%, 20.12%] | 3.539e-06 |
| R50 | true_class_probability | 139 | 0.3153 | 0.3244 | -0.0091 [-0.0333, 0.0165] | 0.1019 |
| R50 | top1_retention | 104 | 36.54% | 42.07% | -5.53% [-12.98%, 1.68%] | 0.06037 |

逐图和 20 次随机重复结果分别见 `results/rq4_r_keep_per_image.csv`、`results/rq4_r_keep_draws.csv`。

## S8 桌面归档

- `C:\Users\Administrator\Desktop\Attributing and dating historic church architecture using vision foundation models_实验数据`：43,817 个文件，逻辑总大小 9,708,845,913 字节（9.042 GiB）。
- `C:\Users\Administrator\Desktop\Attributing and dating historic church architecture using vision foundation models_实验材料`：22,255 个文件，逻辑总大小 25,210,544,743 字节（23.479 GiB）。

大小为逐文件逻辑字节数之和；硬链接按目录条目计数。原项目文件夹只保留 `MOVED_TO_DESKTOP.txt`。
