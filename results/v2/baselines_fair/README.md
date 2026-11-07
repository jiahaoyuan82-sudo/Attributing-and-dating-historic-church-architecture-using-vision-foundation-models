# 公平基线重跑文件清单

本目录只包含 B-2 至 B-7 的新训练、验证选择、一次性测试预测和统计。原 `results_v2/` 的基线文件及 ChronoArch-F 文件未覆盖。

| 文件 | 用途 |
|---|---|
| `selection_log_linear.csv` | 线性模型全部 C / Ridge alpha 验证搜索，包括边缘扩展。 |
| `selection_log_neural.csv` | B-3/B-5/B-7 的 seed 0 八组验证网格及选中配置的额外 seed 训练。 |
| `selected_*.json` | 每任务、模型、折的验证选模结果；RQ3a/RQ3b 后四折引用首折配置。 |
| `pred_*.csv` | 272 份最终测试预测；线性模型单次，神经模型三 seed。 |
| `pred_*.test_started` | 与成功预测文件配对的一次性测试启动标记。 |
| `validation_best.csv` | 仅按首折或固定验证集确定的五任务最强公平基线。 |
| `main_comparison_fair.csv` | 七种公平基线在五任务上的主指标、seed 统计、集成和教堂级 95% CI。 |
| `significance_fair.csv` | ChronoArch-F 与验证选出基线、第一轮 B-1 DINO-B 的配对差值、CI、p 与 Holm p。 |
| `code_src_sha256_before.csv`、`code_src_sha256_after.csv` | 修改代码前后 `code/src/*.py` 的完整 SHA-256。原 46 个文件哈希不变，只新增两个独立脚本。 |
| `checkpoints/` | 公平基线验证搜索及最终配置的模型权重，供追溯。 |

报告正文见实验数据目录的 `RESULTS_v2.md` 新增“公平基线重跑”一节。所有 272 份测试 CSV 的行数、`image_id` 顺序及预测有限性已核对。
