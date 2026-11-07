# S0–S1 run log

- Run date: 2026-11-04 (Asia/Shanghai). Seed: 0.
- Plan file in this workspace: `实验方案_Attributing and dating historic church architecture using vision foundation models.md` (the requested `EXPERIMENT_PLAN.md` name was not present). Part B was read in full before work began.
- Source: WikiChurches v1.0, Zenodo record 5166987. Downloaded the nine small S0 files. Their MD5 values matched the Zenodo record. `models.zip` was not downloaded.
- `images.zip`: download in progress. Original single connection transferred at about 2 MB/s, so the download was switched to resumable HTTP range requests. Exact expected size: 12,389,751,014 bytes. Full MD5 and extraction are pending.
- Environment: no Python executable was on PATH. Portable Python 3.12.10 was installed inside `env/python`; S1 packages were installed and frozen in `env/requirements.lock.txt`. GPU detected: NVIDIA GeForce RTX 5060 Laptop GPU, 8151 MiB, driver 617.42.
- The automatic approval review rejected installation of `torch`, `torchvision`, `timm`, `open_clip_torch`, `transformers`, `matplotlib`, and `seaborn`, stating that model and plotting dependencies were beyond the user's current S0–S1 authorization. No attempt was made to bypass that rejection. These are needed for S2 onward, not for S1 processing.
- Dataset README confirms `churches.json` fields `built_year` and `country`, image names `<church-id>_wd<index>.jpg`, and canonical label files inside `labels.zip`.
- For missing `built_year`, Wikidata SPARQL P571 was queried by QID and cached in `data/processed/wikidata_cache.json` with timestamp. Churches with multiple distinct P571 years retain the raw values but no selected year. `country` was present in all original church records, so P17 was not needed.
- The generated WC14 split is stratified by class at church level. The split uses 25% test (minimum five churches per class), 10% validation, and the remainder train; rounding is performed at class level. The pHash threshold is Hamming distance ≤4. Test images near duplicated in training will be marked `excluded_duplicate`.

## Completed S0 verification

- `images.zip` was assembled to 12,389,751,014 bytes and its MD5 was `510367ec6dbac77d40f77c58673a1026`, matching Zenodo. `results/download_checksums.json` records matching MD5 values for all 10 requested files. The download ran from approximately 20:15 to 21:40 Asia/Shanghai (about 1 hour 25 minutes); intermittent connection resets were resumed at the exact byte offset.
- `labels.zip` was extracted (12 files). `images.zip` was extracted (9,502 JPEGs). No file failed the archive path safety check. The image archive contains 17 files whose QID is absent from `churches.json`; these 17 files are retained under `data/raw/images` but excluded from the analytic metadata, leaving 9,485 eligible images. Their names are in `data/processed/dedup_report.json`.

## Completed S1 verification

- Wikidata SPARQL P571 was queried for all 8,046 church records whose `built_year` is null. Query completion time is in `data/processed/wikidata_cache.json`; all 8,046 QIDs were cached and no batches failed. There were 2,725 unambiguous new single-year values. Forty-seven QIDs returned multiple distinct P571 years and were left without a selected year.
- Nine original `churches.json` entries have a **list** of `built_year` values. Their original lists are preserved in `construction_year_raw`; `construction_year` is null for these nine because selecting one year would be arbitrary. This is a documented adaptation to the actual source format.
- `data/processed/metadata.parquet` has 9,485 rows. `splits/wc14.csv` uses church-level stratification with seed 0. The generated WC14 split has 5,798 train, 891 validation and 2,242 test images before duplicate filtering.
- pHash was calculated for all 8,931 WC14 images and cached in `data/processed/phash.csv`. With Hamming distance ≤4, there were zero train/test near-duplicate pairs and zero test images removed. `data/processed/near_duplicate_pairs.csv` and `data/processed/dedup_report.json` contain the audit result.
- `tests/test_no_leakage.py` passed for WC14, WC6 and WC4. All dataset-statistics class image and year counts sum to their subset totals, and the country Top-10 is in descending image count.
- Archive and `image_meta.json` have 28 nonmatching filenames in each direction; 27 eligible images lack an `image_meta.json` record. The exact filenames are in `data/processed/dedup_report.json`. This does not affect the church/style/year fields but may affect later EXIF-based analysis.
- Year coverage is defined as images with one usable numeric `construction_year` divided by images in the stated subset. Across all eligible images it is 4,086/9,485 (43.0785%); across WC14 it is 3,713/8,931 (41.5743%). By unique church, the corresponding counts are 4,016/9,346 (42.9703%) and 3,650/8,800 (41.4773%). Full class and subset statistics are in `results/dataset_stats.csv`.

Stopped after S1 as requested. S2–S8 have not been run.

## Resumption for S2–S8 (2026-11-04, Asia/Shanghai)

- The user confirmed S0–S1 and explicitly authorized S2–S8, all B1 dependencies, and the specified model-weight downloads. The updated `EXPERIMENT_PLAN.md` was read in full. The additional RQ3a analyses will run for both Gothic and Romanesque, with mixed-rule primary analyses and strict-year sensitivity analyses. The original four models and main metrics remain unchanged.
- S8 now requires moving all project contents into two desktop folders, then leaving only `MOVED_TO_DESKTOP.txt` in the original project folder. This will occur only after S2–S7 are complete and verified.
- Installed portable PyTorch 2.14.1+cu130 and torchvision 0.29.1+cu130. The first installation attempt failed at a Windows long-path limit inside the long project path; retrying through a temporary `subst X:` mapping to the **same project directory** succeeded. CUDA 13.0 detected the NVIDIA RTX 5060 Laptop GPU. Remaining B1 libraries were installed without changing models.
- S2 image cache: 9,485 images resized to short side 256 and saved as JPEG quality 90 in `data/processed/img256/`; original width, height, file size, and EXIF capture year (if present) were saved in `data/processed/image_properties.parquet`. One image emitted a corrupt-EXIF warning, but its pixels were processed. The original image files remain intact.
- RQ3a strict-year sensitivity convention: original images retain the confirmed non-revival-label restriction and original cutoffs (Gothic ≤1550, Romanesque ≤1250); revival images require the respective level-1 style and construction year ≥1800. This removes label-only revival images without a qualifying year. Mixed-rule primary groups follow S5, including revival labels even when year is missing, excluding label/year conflicts.
- RQ3b sensitivity exclusion rule, set before running the model analysis: remove explicit Gothic Revival labels with year ≤1550 and Romanesque Revival labels with year ≤1250; for each other WC14 style with ≥10 dated non-revival churches, compute the tenth percentile of church-level years and remove images more than 200 years earlier than that percentile (floor at year 800). This is a conservative data-derived early-year flag, not a claim about exact historical onset. The per-style cutoffs are saved in `results/rq3b_early_year_thresholds.json`, and affected image IDs/reasons in `results/rq3b_sensitivity_exclusions.csv`.
- The RQ3b main candidate set is WC14 with years 800–2020. It has 3,668 images. The sensitivity filter removes 81 of those, leaving 3,587. The complete dataset contains 49 Gothic Revival label/year conflicts, but only 20 of them are in the WC14 dating set; two Romanesque Revival conflicts are in that set. Other exclusions are empirical early-year flags. These are pre-analysis data counts, not model results.
- RQ4 annotation audit: `building_parts.json` contains 139 annotated images, including 8 without an official WC4 single-label assignment. All 8 have expert style labels from the four WC4 classes. RQ4 will retain all 139 and sum classifier probability across an image's one or two expert-annotated true styles; the top-1 flip compares the original prediction with the occluded prediction. All 2,780 planned random-box placements passed a geometry preflight with the center-80% and ≤10% IoU conditions.

## CUDA audit and S2 continuation (2026-11-05, Asia/Shanghai)

- The updated plan's mandatory-GPU requirement was read. `env/gpu_info.txt` records the executed PyTorch CUDA check, including `sm_120`, a tensor on `cuda:0`, `torch.cuda.is_available(): True`, and `open_clip_torch` version. The installed PyTorch was not reinstalled.
- The deep-model adapter now requires `device="cuda"` and raises an error if CUDA is unavailable; S2 also checks availability before using any existing feature cache. S5 recompression and S6 occlusion instantiate the same CUDA-only adapter.
- Before continuation, `R50.npy` was verified as `(9485, 2048)` and `DINO-B.npy` as `(9485, 768)`. Both image ID CSV files have 9,485 rows in exactly the same order as `metadata.parquet`. Their extraction records state `device: cuda`, batch 64, 8 workers, and elapsed times 138.36 and 234.33 seconds respectively. No CPU deep-model output needs a `.cpu_backup` rename.
- The first CLIP-L load used the exact requested `ViT-L-14`/`openai` weights but emitted an `open_clip` warning that the model activation default (GELU) did not match the weight metadata (QuickGELU). Its seven output files were renamed with `.pre_quickgelu_backup`. The same model and weights are being re-extracted with `force_quick_gelu=True` to match the weight's activation setting. This is a configuration correction, not a model substitution.
- SigLIP 2's Hugging Face request initially failed under the restricted network sandbox with Windows socket error 10013 before feature computation. An approved elevated network call to the same requested repository succeeded; the exact `google/siglip2-base-patch16-224` download/extraction was restarted without model substitution.
- The Hugging Face Xet transfer of SigLIP 2 `model.safetensors` stalled with an empty incomplete file. A HEAD request confirmed the exact file has 1,500,800,904 bytes. Retrying the same file with `HF_HUB_DISABLE_XET=1` switched to Hugging Face's direct file route, and the incomplete file began growing. No alternate weights were used.
- Corrected CLIP-L extraction completed on CUDA with batch 64 and 8 workers in 172.59 seconds. `CLIP-L.npy` has shape `(9485, 768)` and its image-ID index matches the metadata row for row.
- SigLIP 2's direct Hugging Face transfer later stalled and was resumed via `curl -C -` against the same authorized `model.safetensors` using the pre-existing `X:` short drive mapping of this project. The file's official linked size and SHA-256 are checked before use. The earlier abandoned incomplete transfer is retained for audit.
- The exact SigLIP 2 `model.safetensors` completed at 1,500,800,904 bytes. SHA-256 was `612923381c76ec5a9bed335d1c48827e3f2e506ac31b044b63b2031fadee6a0b`, matching Hugging Face's linked ETag. It was placed in the corresponding cached snapshot as a hard link, not substituted. S2 SigLIP 2 extraction then started in offline mode and reported `device=cuda`, batch 64, feature dimension 768.

## S2–S4 completed (2026-11-05, Asia/Shanghai)

- SigLIP 2 S2 extraction completed in 90.8 seconds on CUDA, batch 64, 8 workers, feature dimension 768. All four feature arrays have 9,485 rows, exactly matching the image-ID index and metadata order, with finite values. Their extraction JSON records all state `device: cuda`.
- S3 RQ1 completed in 121.61 seconds. The WC6 validation-set rule selected DINO-B as the best linear probe. Complete protocol results, per-class recalls, test probabilities, paired bootstrap comparison and exact McNemar test are in `results/`. The first S3 process stopped immediately due to portable Python not automatically adding `src` to `sys.path`; explicit insertion was added to the relevant scripts before the successful run. No failed-run outputs were used.
- S4 RQ2 completed in 6.36 seconds. A first attempt stopped before results because the locally generated `.npz` contained object-typed string arrays; loading it with `allow_pickle=True` for this trusted local file fixed that issue. Twelve styles had at least ten dated churches; two were excluded. The full 14-class confusion matrix is row-normalized before selecting the 12 dated classes, retaining predictions into excluded classes in each row denominator. Both 10,000-permutation Mantel tests returned nonsignificant results; see `results/rq2_mantel.csv`.
- S5 Romanesque group audit corrected a preliminary count distinction: 494 is the label-bearing Romanesque Revival group after conflict exclusion. The prescribed mixed rule also includes 16 level-1 Romanesque images dated ≥1800 without an explicit Revival label, for 510 mixed-rule revival images. The original group has 392 images; strict-year revival has 187. No grouping rule was changed.

## S5–S7 completed (2026-11-05, Asia/Shanghai)

- S5 RQ3a and RQ3b completed in 464.1 seconds. Gothic and Romanesque each ran four models under mixed primary and strict-year sensitivity definitions. DINO-B was the WC6-validation-selected model used for JPEG quality-75 re-extraction: the 2,337 Gothic and 902 Romanesque mixed-rule images were each resized/recompressed, then passed through the CUDA-only FrozenModel adapter with batch 16. The image-property and country-weighted controls were also completed. The RQ3b main dated WC14 set has 3,668 images; its predefined sensitivity exclusion removed 81 and left 3,587. The exclusion CSV contains 155 images across the entire metadata; only 81 fall in this dating set. All 5-fold models, baselines, bootstraps, and style-controlled partial correlations were run for both sets.
- S6 RQ4 completed for DINO-B and R50, both using the CUDA-only FrozenModel adapter for every occluded variant. All 139 annotated images and 20 random repeats per image were processed. The first 71.7-second run revealed that this SciPy version did not expose `zstat` from `wilcoxon`; saved probability results were valid, but effect-size fields were NaN. A signed-rank z statistic was computed from the ranks of nonzero paired differences, and the complete GPU experiment was rerun in 72.5 seconds. The final results have finite effect sizes. The validation/train exclusion counts and selected C are in `results/rq4_classifier_info.json`.
- S7 plotting final run took 15.62 seconds and produced the eight required figure pairs plus a Renaissance→Baroque error montage (nine PNG and nine PDF files). `RESULTS.md` was generated from saved result files with `src/write_results.py`; final S8 inventory will be inserted after movement. `tests/test_no_leakage.py` passed after the path refactor. The result summary explicitly reports nonsignificant RQ2 and best-model RQ4 findings.

## S8 preflight (2026-11-05, Asia/Shanghai)

- The current user desktop resolved to `C:\Users\Administrator\Desktop`; the project folder is directly beneath it. Both are on C:, so the planned moves are same-volume moves. `System.IO.DriveInfo` reported 97,429,999,616 available bytes (90.739 GiB) before movement.
- Before movement, the original project tree had 66,058 files with 34,918,504,006 logical bytes (32.520 GiB); this was measured using a Windows extended-path inventory routine. The count may increase by final log updates and the post-move inventory file.
- `RESULTS.md` exists; all four feature arrays have 9,485 rows and exact metadata image-ID order; the church-level leakage test passed. The eight required figure pairs and additional error montage are present. Desktop targets with the exact requested names did not exist before S8.

## S8 completion (2026-11-05, Asia/Shanghai)

- Same-volume Move-Item placed all experiment files and both dataset directories into the exact requested desktop folders. The original project folder contains only MOVED_TO_DESKTOP.txt. The core move command took 5.28 seconds.
- Post-move MD5 verification of all ten original Zenodo files passed in 19.94 seconds; details are in results/download_checksums.json and the materials README.md.
- Original raw directory inventory remained 9,524 files / 24,891,471,762 logical bytes; processed directory remained 12,730 files / 319,070,644 logical bytes, matching their pre-move inventories exactly.
- After moving, prepare_data.py was rerun from code/ against the new material paths. SHA-256 of metadata.parquet and splits/wc14.csv was identical before and after; the recheck output is in results/s8_prepare_recheck.log.
- C:\Users\Administrator\Desktop\Attributing and dating historic church architecture using vision foundation models_实验数据: 43,817 files, 9,708,845,913 logical bytes (9.042 GiB).
- C:\Users\Administrator\Desktop\Attributing and dating historic church architecture using vision foundation models_实验材料: 22,255 files, 25,210,544,743 logical bytes (23.479 GiB).
- Final file counts and logical sizes were computed with Windows extended paths; hard links count as directory entries. The moved config resolves both material directories, and the church-leakage test passed after moving.

## 补充分析 (2026-11-05, Asia/Shanghai)

- The desktop experiment-data folder had been nested inside the original project folder at this turn's start. A direct move was interrupted by Windows long-path handling. The remaining directories were moved with extended paths, the single residual cache lock was merged, and only verified-empty source directories were removed. The original project folder again contains only `MOVED_TO_DESKTOP.txt`; no model or data file was deleted.
- RQ3b B2 uses the one `all_styles` finest-style QID assigned to each dated WC14 image. The main and conflict-excluded sets each have one such QID per image. Each original church-grouped outer fold uses its own training-fold median by this QID; unseen QIDs use that fold's global median. The original five folds, seed 0, metrics, and 1,000 church bootstrap samples were retained. Outputs are `results/rq3b_b2_fine_style.csv` and `results/rq3b_b2_oof.csv`.
- RQ4 expert area is the union of the pixel masks drawn from all expert boxes on each cached 256-pixel image, divided by image pixel count. `sum_area_fraction` separately retains the un-unioned normalized box-area sum. Outputs are `results/rq4_expert_box_area.csv` and `results/rq4_expert_box_area_distribution.csv`.
- R_keep uses the existing random-box placement rule: 20 draws per image with the same box count and sizes, centers in the image's central 80%, and IoU <=0.1 against each expert box. It applies the same keep-only occlusion as K. DINO-B and R50 were each loaded through `FrozenModel(device="cuda")`; their image forwards used CUDA. The original seed 0 and WC4 classifier selection were retained. K was also forwarded again in the same CUDA run. True-class probability sums over expert true styles. Top-1 retention is evaluated among images with a correct original top-1 prediction, with each R_keep image represented by its 20-draw correctness rate. The paired Wilcoxon test uses per-image K versus R_keep mean; the mean-difference CI uses 1,000 church bootstrap samples. Outputs are `results/rq4_r_keep_per_image.csv`, `results/rq4_r_keep_draws.csv`, `results/rq4_r_keep_tests.csv`, and `results/rq4_k_recompute_checks.csv`.
- A preliminary retention calculation inferred K correctness from whether its top-1 matched the original top-1. This failed to count a switch to another expert-approved true style as correct. The final run directly evaluated K's predicted class and replaced the preliminary output. An intermediate exact K-probability assertion also exposed small batch-dependent GPU numeric differences; the final run recorded these differences. Maximum absolute old-versus-recomputed K probability difference was 0 for DINO-B and 0.0003344714641571045 for R50. The completed final supplemental script took 54.5 seconds; the preliminary runs are not used in the report.
- `RESULTS.md` reports p values that underflow numerically to zero as `p < 1e-16`, including Table 2. The raw statistical output remains available in the CSV files for audit.

## v2 中断后只读核查（2026-11-06，Asia/Shanghai）

- 重新完整读取了 `code/EXPERIMENT_PLAN_v2_model.md.md`（磁盘上实际文件名带两个 `.md`）和本日志。B0 第 1 条规定：新代码位于实验数据文件夹的 `code/src/`，新特征位于同级 `features_v2/`，新结果位于同级 `results_v2/`，新图位于同级 `figures_v2/`，`RESULTS_v2.md` 位于实验数据文件夹根目录；实验材料只读，第一轮文件不覆盖或删除。
- 上次会话中断时，最后已观察到 `DINO-L_224_flip_rq4` 完成；接续阶段为 B1 的 `DINO-L_448_flip_rq4_tokens.npy`、`DINO-L_448_flip_rq4_cls.npy`。后台顺序脚本继续完成后续阶段，最后的 `SIG2-B_rq4_flip_progress.json` 于 2026-11-06 00:18:38（本地时间）记录 5,838 条完成。核查时无 Python 提取进程。
- 本次没有运行深度模型或新实验。逐行扫描 `features_v2/` 全部 42 个 `.npy`：各数组的行数、形状、float16 类型与对应 CUDA 进度记录一致；全 0 行、含 NaN 行、含 Inf 行均为 0。9,485 行的 `image_ids.csv` 与 `metadata.parquet`、第一轮 DINO-B 特征索引逐行同序；RQ4 索引 5,838 行，涵盖 139 张专家标注图，每图 E、K 各 1 条，R、R_keep 各 20 条，成对随机框一致。完整逐文件清单与摘要保存在 `results_v2/b1_integrity_audit.csv`、`b1_integrity_summary.json`。
- 审计前 `results_v2/` 没有 CSV；新产生的审计 CSV 有 42 行、42 个唯一文件名，均标为完整。`results_v2/b2_unit_test.json` 记录 ChronoArch-F 与 ChronoArch-L 在 CUDA 上的单次前向/反向通过。没有发现需改名为 `.incomplete` 的文件；没有删除任何文件。
- 状态：B1 特征提取及本次完整性核验完成；B2 核心模型定义、参数计数和单次 CUDA 前向/反向单元测试完成，但 RQ3a 的复兴标签排除、RQ3b 任务专用损失及推理集成流程尚无完整单元测试/训练流程，故不将完整 B2 规范标为全部完成。B3–B7 尚未开始。后续继续时必须沿用 seed 0、原超参数网格与验证集选择规则，且只在用户确认后继续。

## v2 B2 补全与用户批准的压缩方案（2026-11-06）

- B2 任务专用流程已实现于 `code/src/chronoarch_tasks.py` 和 `code/src/chronoarch_train.py`；`code/src/test_chronoarch_tasks.py` 对 WC6、WC14、Gothic、Romanesque、dating 五项任务检查划分、辅助标签屏蔽、CUDA 前向/反向与真实训练折/验证折特征读出，全部通过；测试折没有用于 B2 检查。正式实验训练尚未开始，v2 测试折尚未评估。
- B2 检查时桌面 C: 剩余 46,745,161,728 bytes（43.535 GiB）；`features_v2/` 共 68 个文件、71,882,958,348 bytes（66.946 GiB）。ChronoArch-F 可训练参数 7,287,910，ChronoArch-L 可训练参数 7,484,518（LoRA 增加 196,608）；两者 CUDA 前向/反向均正常。B1/B2 已记录的最高峰值显存为 allocated 2,034,578,432 bytes（1.895 GiB）、reserved 2,212,495,360 bytes（2.061 GiB），见 `results_v2/b2_unit_test.json` 及 `features_v2/*_progress.json`。
- 用户批准的压缩：ChronoArch-F 主结果与全部基线使用 5 seeds；ChronoArch-L、B-8、B-9 使用 seeds 0–2。B4 消融只在 WC6 与 RQ3a Gothic 上使用 seeds 0–2；WC14 与 RQ3b 仅报告完整模型及基线。分阶段验证集搜索：阶段 1 固定 λ_fine=1、λ_year=0.3、τ=1，搜索 K∈{4,8} × lr∈{5e-4,1e-3}（4 组）；阶段 2 固定最优 K、lr，搜索 λ_fine∈{0.3,1} × λ_year∈{0.3,1} × τ∈{0,1}（8 组）。所有网格结果写入 `results_v2/selection_log.csv`。数据划分、其他训练设置、验证集选择与最终配置测试折单次评估规则保持不变。
- 在正式训练开始前修改了 `chronoarch.py` 并新增任务/训练代码与单元测试；训练开始后冻结模型与训练代码。如必须修复 bug，先停止并告知用户拟修改代码及已完成结果的影响。
- 训练前另以 WC14 训练折的真实特征预检 ChronoArch-F 指定 batch 64 的 CUDA 前向/反向（无 optimizer step）：通过；峰值 allocated 821,807,616 bytes、reserved 1,109,393,408 bytes，详见 `results_v2/b2_task_tests.json`。这不是正式训练，也未访问测试折。
- ChronoArch-L 的在线 224 像素分支、缓存 448/语义特征融合、LoRA 梯度累积训练与验证读出已加入 `chronoarch_train.py`。WC14 真实训练折 1 图 CUDA 前向/反向及验证折 1 图双视图读出通过；LoRA up 权重获得有限梯度，峰值 allocated 1,485,631,488 bytes、reserved 1,507,852,288 bytes；无 optimizer step、未访问测试折。详见 `results_v2/b2_task_tests.json`。
- 为 B4 的 R1 骨干大小消融补提取 DINO-B 224 的 base、flip、crop0、crop1 patch token 与 CLS。8 个新数组逐行核验：各 9,485 行，均无全零、NaN、Inf 行，`image_id` 顺序与 metadata 一致，进度文件显示 CUDA；详见 `results_v2/dinob224_integrity.csv`。新增完成后 `features_v2/` 80 个文件、86,859,850,994 bytes（80.895 GiB）；C: 剩余 32,085,008,384 bytes（29.881 GiB）。原有特征未覆盖。
- B4 的 A0、A1、A1a、A1b、A2、A2a、A2b、A3、A3a、A3b、A4、R1、R2、R3、R4 变体均通过合成输入 CUDA 前向/反向单元测试；详见 `results_v2/b2_task_tests.json`。正式训练仍未开始。
- B3 的 B-2 至 B-9 输入/模型预检通过：WC14 训练折输入维度和有限性正常，B-3/B-5/B-7/B-8/B-9 的 CUDA 前向/反向及验证折读出正常；没有 optimizer step，没有访问测试折，详见 `results_v2/b3_preflight.json`。
- 为避免所有网格检查点耗尽桌面空间，用户明确授权：每个折在验证集确定最终配置后，仅删除该折未选中配置的 v2 临时检查点；`selection_log.csv` 保留全部验证指标，选中配置的检查点保留。自动审批此前拒绝了未获明确授权时加入此清理；获授权后才将该受限清理加入执行入口。
- 在线模型指定 batch 16 的真实 WC14 训练折 CUDA 前向/反向预检通过，未执行优化器更新：ChronoArch-L 峰值 allocated 2,083,366,400 bytes、reserved 2,298,478,592 bytes；B-8 为 792,261,120 / 884,998,144 bytes；B-9 为 1,957,527,040 / 2,065,694,720 bytes。详见 `results_v2/online_batch_preflight.json`。R1 新 DINO-B 224/448 特征的真实训练折反向与验证折读出也通过，见 `results_v2/b2_task_tests.json`。
- 正式 WC6 ChronoArch-F 网格搜索已启动，但在首个 seed 的训练尚未完成时主动中断：B5 RQ4 需要与原 DINO-B 线性探针相同的 WC4 四类、排除专家标注教堂的训练/验证流程，当前 v2 训练入口缺少该任务。依据用户“训练开始后如需改训练代码先停下说明”的要求，未修改任何模型或训练代码。中断时 `selection_log.csv` 尚不存在，v2 测试预测文件 0 个；唯一 29,132,283-byte 检查点改名为 `results_v2/checkpoints/f_wc6_0_558dc3c74560_0.pt.incomplete` 保留，不计入已完成结果。已完成的 B1/B2/B3 预检均不受影响。等待用户决定如何补齐 WC4 对照。

## v2 WC4/RQ4 补充与恢复训练前代码冻结（2026-11-06）

- 原因：B5 RQ4 必须与第一轮 DINO-B WC4 四类探针使用相同的专家标注教堂排除规则；原 v2 训练入口缺少 WC4 任务。用户已明确授权这一次训练代码补充。
- 修改：`chronoarch_tasks.py` 增加 `wc4_rq4` 任务，沿用 `split_wc4`，从训练和验证中排除 139 张专家标注图所属教堂的全部图像；139 张标注图仅用于最终 RQ4 评估，训练和选模不读取其类别标签。`validation_metric` 将该任务按平衡准确率选择；`run_chronoarch_v2.py` 加入 WC4 入口，沿用相同的两阶段 4+8 网格、5 seeds 与最终测试一次性标记。`test_chronoarch_tasks.py` 增加与旧 RQ4 计数的核对；新增 `rq4_chronoarch_v2.py` 用于最终四条件读出。未改动模型结构、其他任务划分或主指标。
- 核验：WC4 训练 5,748 张、验证 198 张，分别与第一轮 `rq4_classifier_info.json` 相同；专家标注评估图 139 张、四个类别；真实训练折 CUDA 前向/反向及验证折读出通过。首轮未完成 WC6 seed 继续作废，`.incomplete` 文件保留。此前没有 v2 完整训练配置、没有 v2 测试集评估。
- 以下为恢复正式训练前 `code/src/` 全部 `.py` 文件 SHA-256；文件数 
37
。

| 文件 | SHA-256 |
|---|---|
| `audit_dinob224_v2.py` | `ffaba0f587d5276e13c8931e8ac882845c1797fb40ef2f7477b56834b847ae37` |
| `audit_s5_groups.py` | `fa112432f6cc484dca1a4a587c4113f027e6e9d9b42f6a2f637342fcfcfdbbe7` |
| `audit_v2_integrity.py` | `ab0b8cf40e225c3cff75dc98ebef344711030adec9c910a260d7051d3d166369` |
| `baselines_v2.py` | `7904113d54e39af472a034068492f7effcbd92498a9af8f7ee09f69893b75629` |
| `cache_images.py` | `4dde885e715cac02f1943f3a41c0973e73aa75cdae12d90d0f80bd10df43f9b4` |
| `chronoarch.py` | `a701bb7f1cb914d99f4fed5cf7c4e35cd29a34c29d4fcc52a34a5fb40b9ef526` |
| `chronoarch_tasks.py` | `12dff07bdc6fc3d24f0c620863b5b8a9aa98cf0cf0e9fc2814a102cf03c52032` |
| `chronoarch_train.py` | `f958c29fc655e58aa6cb2de6e0b3098e01095f1a4b9c33b8c834743c82222c28` |
| `download_images.py` | `67f87fa776de375b5c47e95ac8fff735c176e55f028064179946cbe844233397` |
| `download_v2.py` | `a89d375d0837466024f108574e74b75a5d09207071bbcc0a9d4f1fe6f0f2f5d8` |
| `extract.py` | `8255b47b7415e5ecdbe15ecbb2ece789c2562fbb4b3e12eba619acd1152a4890` |
| `extract_images.py` | `49cff8c4270c552052f1cc32dab05ee9edabce01ad6fc57e2e2e3fe30ad203f4` |
| `extract_v2.py` | `447aa8a7a91275a3ff63c0830576027a00343799501fcda60ba61122d8dace4e` |
| `fetch_wikidata.py` | `cdb64792ce8b954fc43478ded5b0247f469638941c0eb5723774d0821de33600` |
| `finalize_s8.py` | `403a3e73b2c0f339fbeac8932ddec78a137bcedb81b83ead5015e75301dc32e1` |
| `inventory.py` | `299f7f2c84b3ea7a8738ae062be1347685431b708a4004a6bdbf4ede94893150` |
| `models.py` | `003c9870a637d593ef53b1d338cfdc67ad3a3d55462f9e2d2aaca124d8cd4b9c` |
| `pathing.py` | `73ea1b9187310088aeb9d62b9ebdec8efff568a580fd7132f2c935ca643f453e` |
| `plots.py` | `e964668eb4d26bcd236e58aeb139e36de5bf3b7c783f1991505832b493c8ea2a` |
| `prepare_data.py` | `8933776eac6c2349857952b1f015b8fd654ea0b0003c04e8c618e568273c2b4c` |
| `rq1.py` | `afa5492a571031302bbecaad29c713c3d4584f7a13b689797bff2cc8096c0cb5` |
| `rq2.py` | `e7127199242d33a177bf6bf357f86c677dbcd5a472d181a212fb3b5983c567bd` |
| `rq3.py` | `b9eaedad07b24a5ccf1f993782e687a9ef47d6b72351d66f0b907a8dad5dd470` |
| `rq4.py` | `538af117b6e5c79ab67de0917e2cc44c3449dccaaf8ad5196a6872d31eed72f7` |
| `rq4_chronoarch_v2.py` | `3da13cd97ff0654a827e4dc306bd949ffd04bddd8410e4930d8990b7d6bd368c` |
| `run_ablation_v2.py` | `bcc226fba12a0fe625d7a3d40c6047f8309b4c62dd8cea5f4bf5e1ef274247b2` |
| `run_baselines_v2.py` | `ecaa872f2c0f2125dfa6bcfa584b667f722fc5e1ab8ce87f293c9df67a1d9ab6` |
| `run_chronoarch_v2.py` | `00ab22f020e8e2adb89cf0b8ba63b843e15f4ee4e85221a19561eaebcd29ecd6` |
| `supplement.py` | `3d0ef6554ff12ac005d6bb066453924fc4b3c50f73194737aba5fe35c4bf7c62` |
| `test_baselines_v2.py` | `fe7dece3492964aae9f217d3b7cfc92a616d7178ef47fa22cfce638fd49441c2` |
| `test_chronoarch.py` | `1c31f58220fcb19f83cd4913bb9b4852aa4941986ed5054c53a8c6a42003ca3d` |
| `test_chronoarch_tasks.py` | `df8cf55fd7cb13a1e5808b4fc1ae58c4c8b60f2a0c1d8851162f7ade882d7b2b` |
| `test_online_batch_v2.py` | `a723d1a07316bd94bcee4876edd641d4a081410f177bc51cb99e1bdf623519bb` |
| `verify_b1_v2.py` | `578b2c264748980ccbb662f58cd88f80f992035f19ed54074ff728bd024b43a9` |
| `verify_downloads.py` | `444c09fd87d72a63f4a55b151b7e7bc4a4edb085588983b0769be3c88caa4e62` |
| `write_materials_readme.py` | `86b870aeb0313b9cef6637924ec62e9f80c385a438a6ad5267fbe67d2cc4fc84` |
| `write_results.py` | `15fd9b4b00601d8535d1b2276dfae570c51d73f62cb4a940baff02954db3f895` |

### 训练启动后新增的分析脚本（不参与模型训练或选模）

训练前冻结表中的 37 个文件 SHA-256 复核均未变化。下列文件仅从已保存预测生成汇总、统计、图和报告；新增后单列哈希，不改变训练前冻结表。

| 文件 | SHA-256 |
|---|---|
| `ablation_summary_v2.py` | `4b9b7c0d97b2fafd9bf8bf9a75c28442a68a7042b75c3217734f32803f710365` |
| `compute_cost_v2.py` | `9f6474c8a3725fc1d5bad51915fd021f0781802f4540225a8161d11251b83aca` |
| `figure_architecture_v2.py` | `59fed9b25daadaef55b75b05d10f983b08b97b0f65f808e4498fb3fba132eec1` |
| `gates_attention_v2.py` | `d6ef85204d9ce9f2ccab7e0f7d105e28705f5d9c71165a5f96df0b614532b48c` |
| `significance_v2.py` | `cc585515d84fcebd85826ddfadbf8c0e689f5d820bdbe15c4b1163b7bf4a821d` |
| `summarize_v2.py` | `737e925d83aa44cac300f50b2b8e2271836ca3ae41f96ba8d97f6b73822d7553` |
| `write_results_v2.py` | `93e2e3097f2a2a83065437bd0ad07f6c7d942683b47a00a3a366db2afa0b3443` |

### 后续只读分析脚本哈希复核（2026-11-06）

冻结的 37 个训练前文件再次复核：SHA-256 变化数 0。以下分析脚本在训练开始后新增或更新，均未参与模型训练；这些哈希取自本次复核。

| 文件 | SHA-256 |
|---|---|
| `ablation_summary_v2.py` | `4b9b7c0d97b2fafd9bf8bf9a75c28442a68a7042b75c3217734f32803f710365` |
| `compute_cost_v2.py` | `9f6474c8a3725fc1d5bad51915fd021f0781802f4540225a8161d11251b83aca` |
| `figure_architecture_v2.py` | `59fed9b25daadaef55b75b05d10f983b08b97b0f65f808e4498fb3fba132eec1` |
| `gates_attention_v2.py` | `d6ef85204d9ce9f2ccab7e0f7d105e28705f5d9c71165a5f96df0b614532b48c` |
| `mark_selection_v2.py` | `5f9aa89cea7b6e91b840c39fee16e58029bd061e3e910031dcd522c01f0b97d4` |
| `reuse_b1_v2.py` | `2412d073299b1efe8c58767fe03ad1a15ee6f2cefa44dad6f41316ec3c4a5721` |
| `significance_v2.py` | `7bf168f42ed4e325107e4f12d65cdbf4470d07b11bd3b1013783748011e9ac59` |
| `summarize_v2.py` | `e8cf760a495c11b5aa0ff9e31b50cac6b07d8be935df68fc1051d303be755076` |
| `write_results_v2.py` | `172e3227934d82937144253d35b8b915465556442fccf3519bb715ecdab4e39e` |

- 为持续完成已授权的 B3–B7，2026-11-06 22:45:52 +08:00 启动隐藏 PowerShell 顺序执行器（PID 22460）。它等待当前 WC6-F 进程（PID 13508）正常结束并确认五个最终预测文件齐全，再依次调用冻结的 ChronoArch、基线、消融与分析脚本；任一步非零退出即停止，不自动修改代码或重试测试评估。状态见 `results_v2/pipeline_status.log`，每步日志见 `results_v2/pipeline_*.log`。执行器本身没有写入 `code/src/`。

- `figures_v2/model_architecture.pdf` 已生成，并渲染为同目录 `model_architecture_preview.png` 目视核验：三编码器输入、局部注意力、门控融合和三任务头的结构、箭头及文字均完整可读。此图不使用测试集。

- 第一轮 B-1 四种模型的既有预测已从原 `results/` 只读转换至 `results_v2/b1_reused_oof.csv`，共 43,484 行；WC6 1,722、WC14 2,242、Gothic 2,337、Romanesque 902、dating 3,668 张图，各模型/任务索引检查通过。未重新提取特征、训练或评估旧模型。

## 用户更新的自主恢复规则（2026-11-06）

- 用户授权连续执行至 B3–B7 与 RESULTS_v2.md 完成，无需中途确认。显存不足可减 batch 或用梯度累积、梯度检查点；磁盘紧张仅可删除未选中配置的 v2 检查点并保留 selection_log.csv，其他文件不删；非代码进程中断可从该步骤恢复。
- 如发现必须修正的 bug，修正后重新记录 code/src/ 全部 .py 的 SHA-256，并重跑所有受影响的已完成结果，在 RESULTS_v2.md 说明。仅更换模型结构、删减实验、改变主指标或数据划分时停下征询。验证集选超参、每个最终配置测试集只评估一次、深度模型 CUDA 运行及不得编造数字继续有效。
- 本次核查：WC6-F 进程及顺序执行器仍存活，selection_log.csv 已有首组 K/lr 五个 seed 的完整验证记录；尚无 v2 测试预测和 RESULTS_v2.md，无失败或中断。

## 2026-11-07 后台中断恢复

- 核查时原 WC6-F 进程 PID 13508 与顺序执行器 PID 22460 均已退出；pipeline_status.log 仅保留等待 WC6-F 的记录，无后续成功或失败行。selection_log.csv 有 23 条 WC6-F 验证记录，至 loss_tau 配置 dc4fba75f79a 的 seed 2；没有 v2 测试预测。
- 同配置 seed 3 检查点存在，但无验证记录，无法认定训练步骤完整；已改名为 .pt.incomplete 保留。已完成且验证记录与检查点齐全的 23 个 seed 将由现有代码跳过，不重跑。按用户授权从 WC6-F 当前 seed 恢复，设置及数据划分不变。


## 2026-11-07 用户授权的新压缩方案

- 用户要求当前 WC6-F 配置 a095c11ce2ba seed 0 完成后停止原流水线。该 seed 已写入 selection_log.csv，旧方案共 26 条验证记录；PID 14796 与 28996 已停止。旧 selection_log.csv、pipeline 状态/步骤日志、执行器脚本及 checkpoints 文件夹改名为 .superseded 保留，未删除。旧方案没有 v2 测试预测。
- 新方案：DINO-L 224/448 patch token 各 2x2 平均池化为 8x8，base/flip 与 CLS、CLIP-L、SIG2-B 一次性内存载入，不使用 crop0/crop1；fp16 AMP；只保留 ChronoArch-F、B-1 至 B-7；seed 0 两阶段验证集选模，固定任务使用各自验证集，交叉验证任务仅第 1 折内部验证选模后共享五折；max_epochs=40、patience=6；最终 F/B-3/B-5/B-7 三 seed 及集成，翻转 TTA；WC6/Gothic 的 A0-A4、R2、R4 三 seed，R1 仅 WC6；原论文微调结果只作引文比较。修改训练代码后重算全体 SHA-256，重新开始。
- 速度目标为用户期望，不作为可伪造的完成时间；新方案前三次实测后报告估算。


### 压缩方案正式训练前代码冻结（2026-11-07）

- code/src/ 全部 46 个 .py 文件 SHA-256 已写入 results_v2/code_src_sha256_compressed_before_training.csv。旧训练输出均为 .superseded。
- 预检：46 个 .py AST 解析通过；ChronoArch-F 64-token CUDA 前向/反向通过；真实 FeatureStore 载入 9,485 行、每尺度 64 token，CUDA batch 形状正确；载入耗时 109.61 秒（含 DINO-B，后改为仅 R1 消融载入 B）。修正过池化输出需展平为 64 token 的预训练 bug，尚无新方案正式训练结果受影响。


### 新压缩方案前三次正式训练的实测与估算

- WC6-F 阶段 K_lr 的 seed 0 前三配置训练耗时：31.828、28.0309999999999、31.2190000000001 秒；均值 30.359 秒，数据来自 results_v2/selection_log.csv。流水线在统计期间继续运行。
- 基于当前速度及尚未实测的其他任务/基线，B3–B7 预计还需约 2–6 小时，属于不确定的规划估算，非已完成实验结果。


- 正式训练启动后，为适配用户已批准的删减范围，仅修改汇总/统计/报告脚本：ablation_summary_v2.py, gates_attention_v2.py, rq4_chronoarch_v2.py, significance_v2.py, summarize_v2.py, write_results_v2.py。重新记录 code/src 全部 46 个 .py 文件 SHA-256 至 results_v2/code_src_sha256_compressed_analysis_update.csv；模型与训练代码哈希未变。该修改发生在任何 v2 测试预测生成前，不影响已完成的三次验证训练，无需重跑。


- 报告/评估脚本适配池化后的 8x8 token：gates_attention_v2.py 将注意力拆为两个 8x8 图；rq4_chronoarch_v2.py 对遮挡特征做相同 2x2 池化；figure_architecture_v2.py 更新图注。代码全量 46 个 .py SHA-256 重新记录于 results_v2/code_src_sha256_compressed_analysis_update2.csv。三个脚本在新方案中尚未执行，已完成 WC6 验证训练不受影响，无需重跑。


- 更新结构图去除本次未运行的 ChronoArch-L 标注；旧结构 PDF/预览 PNG 改名 .superseded 保留，新 PDF 在 figures_v2/model_architecture.pdf。全体 .py SHA-256 更新至 results_v2/code_src_sha256_compressed_analysis_update3.csv。


- 遵守用户要求的汇报顺序：write_results_v2.py 在 main_comparison.csv 生成后等待 results_v2/main_results_reported.ok；代理先向用户汇报主结果并创建该标记，然后才写 RESULTS_v2.md，无需用户再次确认。该脚本尚未运行，现有训练结果不受影响。全体 .py SHA-256 见 results_v2/code_src_sha256_compressed_analysis_update4.csv。


- 仅修改尚未执行的报告脚本，为原论文微调参考线增加 NeurIPS 2021 原文链接；60.2% 和 69.5% 已从原论文核对，标记为外部参考，不作为本实验结果。全量 .py SHA-256 记录于 results_v2/code_src_sha256_compressed_analysis_update5.csv；已完成训练和测试预测不受影响。


- 04:35 左右 dating_baselines 的 Python 与 PowerShell 进程均无错误日志地退出。核查：该步骤仅有 B-2_224 折 0 的四个 C/seed0 验证记录及其四个完整 checkpoint、selected C=0.01；没有 dating 基线测试预测。唯一无对应 CSV 的 test_started 标记为 pred_B-2_224_dating_0_0.test_started，改名 .incomplete 保留；无 .tmp 文件。该次最终测试读出未形成任何结果；按用户授权的非代码中断恢复规则，从 dating_baselines 开始重新执行，可能重复一次未完成的读出尝试，报告中明确说明。此前完成的 F、其他任务基线和消融不重跑。


- dating_baselines 第二次无日志退出：B-2_224 折 0 的三份测试预测已完整保存；B-2_448 折 0 的四条 seed0 验证记录与 checkpoint 完整，但 selected_B-2_448_dating_0.json 为 0 字节，已改名 .incomplete 保留；无新的孤立 test_started 标记。推测高内存占用可能参与中断，但无系统错误证据，故不宣称确诊。
- 修正 code/src/chronoarch_train.py 与 run_baselines_v2.py：线性/B-3/B-5 阶段仅载入其所需 CLS/语义特征，B-7 阶段再一次性载入 8x8 pooled token；特征值、模型、超参数、划分及一次性测试读出不变。受影响的未完成 dating_baselines 从步骤起重跑；此前完整的其他任务基线及 dating B-2_224 预测不受数值算法影响，不重跑。全体 .py SHA-256 见 results_v2/code_src_sha256_baseline_memory_fix.csv。


- 恢复诊断更正：在默认沙箱前台运行 dating_baselines，明确捕获 PermissionError，目标是 results_v2/checkpoints/linear_6ee756050749.pkl。故先前的后台无日志退出与写入权限有关；高内存仅为未证实猜测，不作为根因。随后在用户已授权的实验数据目录权限下，以可追踪 Python 会话重启该步骤，B-4/B-6/B-3 折 0 已正常写出。特征延迟载入只优化内存，不改变数值算法。


- 2026-11-07 用户明确要求 R4 仅报告单 seed 且无翻转增强。既有 run_ablation_v2.py 的 R4 推理设置 tta=False；此前额外保存的 seed 1/2 文件保留但不纳入 R4 正式消融汇总。ablation_summary_v2.py 仅使用 seed 0，重算 ablation.csv；不改变训练或已保存预测。全体 code/src/*.py 的 SHA-256 见 results_v2/code_src_sha256_r4_single_seed.csv。main_results_reported.ok 已改名 .deferred 保留，RESULTS_v2.md 暂缓生成。


- R4 对比修正：R4 只使用 seed0 无翻转读出；差值参照改为同一 seed0 的 A0 翻转 TTA 预测（非 A0 三 seed 集成），保证只测翻转增强条件。其余变体仍对 A0 三 seed 集成。仅重算 ablation.csv，既有训练与测试预测不受影响。全量 .py SHA-256 见 results_v2/code_src_sha256_r4_paired_seed0.csv。


- 训练后全量代码哈希核对：压缩方案训练前冻结 46 个 .py，当前 46 个，9 个哈希改变；完整 before/after SHA-256 见 results_v2/code_src_sha256_diff.csv，当前全量见 code_src_sha256_post_analysis.csv。变动：chronoarch_train.py 与 run_baselines_v2.py 只调整基线所需特征的内存载入时机和 B-7 执行顺序，模型、输入数值、seed、划分、选模和测试协议不变；ablation_summary_v2.py 将 R4 正式汇总改为 seed0 并以 A0 seed0 为参照；summarize_v2.py、significance_v2.py 适配用户删去 L/B8/B9 后的方法集合和三 seed；gates_attention_v2.py、rq4_chronoarch_v2.py 适配 8x8 池化；figure_architecture_v2.py 更新结构图；write_results_v2.py 更新已授权方法说明和汇报门槛，尚未执行。已完成结果的数值计算未受内存载入时机更改影响；R4 汇总已重算，门控/RQ4/主结果/显著性均在相应分析脚本变动后首次运行，无需重训。


- 用户最新要求：先汇报主结果与统计分析，暂不生成 RESULTS_v2.md、暂不创建 main_results_reported.ok。此前标记已改名为 main_results_reported.ok.deferred 保留；当前有效标记不存在，报告文件不存在。只读完整性审计 results_v2/interim_integrity.json：675 个预测 CSV 索引与预定测试 image_id 顺序逐一吻合，预测值有限且无缺失；主总表 60 行、显著性 10 行、消融 15 行、门控 60 行、RQ4 逐次 5838 行/逐图 139 行/检验 3 行。


- 用户授权继续并生成 RESULTS_v2.md。仅更新此前未执行的 write_results_v2.py：使用已完成的 60 行 F/B-1–B-7 主总表、配对统计、R4 单 seed 消融、代码哈希与压缩说明生成完整报告；不访问测试图像或重跑模型。全量 46 个 .py 当前 SHA-256 与冻结记录的差异已刷新：9 个变动文件，见 results_v2/code_src_sha256_post_analysis.csv 与 code_src_sha256_diff.csv。


- 2026-11-07T07:58:47.4706668+08:00 用户授权生成正式报告：创建 results_v2/main_results_reported.ok，运行 write_results_v2.py 生成 RESULTS_v2.md；写入 results_v2/README.md 清单，未移动或删除文件。最终只读核查：报告 275 行，主总表 60 行、显著性 10 行、消融 15 行、代码哈希变动 9 行；五任务及所需章节齐全。B3–B7 与报告完成。


- 2026-11-07T08:12:51.0577213+08:00 用户授权清理压缩方案未使用的随机裁剪视图：仅删除 features_v2/ 中名称含 crop 的 24 个 .npy/.json 文件，共 34,946,082,261 字节（32.546 GiB）。保留 base/flip、RQ4 特征、结果与检查点；已完成的 RESULTS_v2.md 数值不变。


- 2026-11-07T08:52:02.2383531+08:00 Fair baseline rerun completed. Only code/src/fair_baselines_v2.py and fair_analysis_v2.py were added; all 46 pre-existing code/src Python SHA-256 hashes remain identical. Before/after manifests: results_v2/baselines_fair/code_src_sha256_before.csv and code_src_sha256_after.csv. The first invocation failed during module import before any training or test readout; adding the portable Python sys.path fixed it. B-2/B-4/B-6 use per-encoder L2 normalization and the 11-value C/Ridge-alpha grid; 10 of 30 independent searches expanded outward, and no selected value is on a final grid edge. B-3/B-5/B-7 used eight seed-0 validation configs per task, with three final seeds; RQ3a/RQ3b selection used only fold 0 validation. New files are isolated in results_v2/baselines_fair; old baselines and ChronoArch-F are unchanged. All 272 final prediction CSVs (173936 rows) passed image_id order, length, and finite-value checks. Neural selection log has 258 rows. Wrote main_comparison_fair.csv, validation_best.csv, significance_fair.csv, and appended only the fair-baseline section to RESULTS_v2.md. The prior report prefix matches the claued.zip copy after newline normalization.
