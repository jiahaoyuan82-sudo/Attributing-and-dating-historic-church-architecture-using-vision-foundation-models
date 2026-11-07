# results_v2 文件清单

本目录是压缩版 ChronoArch v2 实验的结果与过程记录。相邻目录的 `RESULTS_v2.md` 是正式报告，`figures_v2/` 是图。以下清单只说明用途；未移动或删除任何原有文件。

## 正式结果

| 文件 | 内容 |
|---|---|
| `main_comparison_all.csv` | 60 行主总表：ChronoArch-F、B-1 的四种既有单 seed 探针、B-2 至 B-7；含主指标、seed 均值与 SD、集成或单次分数、教堂 bootstrap 95% CI 和验证集基线标记。 |
| `main_comparison.csv` | 本轮 ChronoArch-F 与 B-2 至 B-7 的 40 行主结果。 |
| `b1_reused_summary.csv`、`b1_reused_oof.csv` | 第一轮 B-1 已保存预测的只读复用及单 seed 汇总；未重训。 |
| `significance.csv` | ChronoArch-F 与验证集选出的本轮基线，以及预先指定的第一轮 DINO-B 的配对比较、区间、原始及 Holm 校正 p 值。 |
| `ablation.csv` | WC6 和 Gothic 的消融正式汇总。R4 只纳入 seed 0，与 A0 seed 0 配对；无 seed 集成或翻转增强。 |
| `selection_log.csv`、`selected_*.json`、`best_baseline_validation_*.json` | 全部验证记录、最终配置和仅按验证集选出的比较基线。 |
| `ensemble_oof.csv`、`per_class_recall.csv` | 集成测试预测汇总和 WC6/WC14 逐类召回。 |
| `gate_by_style.csv` | 验证集门控权重。 |
| `rq4_chronoarch_seed*.csv`、`rq4_chronoarch_draws.csv`、`rq4_chronoarch.csv`、`rq4_chronoarch_tests.csv` | RQ4 每 seed、每次随机框、逐图及配对统计结果。 |
| `compute_cost.csv` | 从实际记录计算的训练、推理和显存成本。 |
| `pred_*.csv` | 每任务、折、方法、seed 的一次性测试预测。完整性审计核对了 675 个文件的图像索引顺序及预测有限性。 |
| `checkpoints/` | 本次压缩方案的检查点；包含选中和未选中的配置，均保留。 |

## 完整性、代码与执行记录

- `interim_integrity.json`：生成报告前的 675 个预测 CSV 和统计表核查。该快照生成时，正式报告及有效放行标记尚不存在；最终状态以 `RESULTS_v2.md` 和本清单为准。
- `code_src_sha256_compressed_before_training.csv`：压缩方案正式训练前 `code/src/*.py` 的冻结哈希。
- `code_src_sha256_post_analysis.csv`、`code_src_sha256_diff.csv`：训练后全量哈希及与冻结记录的逐文件差异。其余 `code_src_sha256_*.csv` 是中间审计快照，均保留。
- `b1_integrity_*`、`dinob224_integrity.csv`、`b2_*.json`、`b3_preflight.json`、`online_batch_preflight.json`：特征完整性、CUDA 和任务预检记录。
- `pipeline_status.log`、`pipeline_*.log`、`compressed_pipeline.ps1`、`resume_dating_pipeline.ps1`：执行和中断恢复记录。
- `audit_interim.py`、`build_main_all.py`、`summarize_b1_reused.py`：只读核查或结果合并脚本，不运行模型训练。
- `main_results_reported.ok`：本次用户授权后用于生成正式报告的有效放行标记。
- `pred_*.test_started`：对应成功预测文件的一次性测试启动标记；与同名 CSV 配对时属正常过程记录。

## 作废或不完整记录，保留但不纳入正式结果

| 路径或模式 | 状态 |
|---|---|
| `selection_log.csv.superseded`、`checkpoints.superseded/`、`pipeline_status.log.superseded`、`pipeline_wc6_f.log.superseded`、`resume_pipeline.ps1.superseded` | 被压缩方案取代的旧训练、检查点与执行记录；不纳入正式结果。 |
| `*.incomplete` | 中断尝试的不完整标记或文件，包括 `pred_B-2_224_dating_0_0.test_started.incomplete`、`selected_B-2_448_dating_0.json.incomplete`，以及旧检查点目录中的两项；均未删除。 |
| `main_results_reported.ok.deferred` | 较早回合创建后按用户指示暂停的旧放行标记；已改名保留，不是有效标记。 |
| `pred_R4_*_1.csv`、`pred_R4_*_2.csv` | 依先前“三 seed R4”指令产生的额外预测，保留供追溯；正式 R4 汇总只使用 seed 0。 |

具体中断、恢复、代码修正和结果影响见相邻目录的 `RUN_LOG.md`。

## 后续公平基线重跑

`baselines_fair/` 保存后来按用户要求独立重跑的 B-2 至 B-7、公平验证选模、272 份测试预测、显著性检验及修改前后代码 SHA-256。其详细清单见 `baselines_fair/README.md`；本目录原有结果和 ChronoArch-F 预测均未覆盖。
