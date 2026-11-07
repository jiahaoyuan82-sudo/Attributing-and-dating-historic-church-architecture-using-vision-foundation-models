# Attributing and dating historic church architecture using vision foundation models

This repository snapshot contains the experiment source, environment lockfile, plans, run log, and small result tables for the WikiChurches study and its ChronoArch-F extension. It does not contain the image dataset, extracted features, model weights, or checkpoints. `results/legacy/` preserves the first experiment's tables; `results/v2/` preserves the ChronoArch tables, including `results/v2/baselines_fair/`. Published source files are byte-identical copies of the experiment code. The additional `src/paths_example.py` documents path setup; it is not imported by the experiment.

## Data and license

Download WikiChurches v1.0 from [Zenodo, DOI: 10.5281/zenodo.5166987](https://doi.org/10.5281/zenodo.5166987). The dataset and its images remain under **CC BY-SA 4.0**; the repository's MIT `LICENSE` applies to the software only. Preserve image-specific author and license attribution when reusing images. No raw images were copied for this release. Some image files were already present in this desktop folder before the release snapshot was assembled.

## Environment and layout

The recorded environment used Python **3.12**, PyTorch **2.14.1+cu130**, CUDA **13.0**, and an NVIDIA **RTX 5060 Laptop GPU with 8 GB VRAM**. Package versions are in `environment/requirements.lock.txt`; the recorded CUDA check is in `environment/gpu_info.txt`. Deep-model computation requires CUDA. The scripts require additional downloaded model weights described by the plans in `docs/`.

The original scripts resolve the project root from a `code/src/` layout. This publication places the requested unchanged sources in `src/`. For a fresh run, first copy `src/` to `code/src/` and `tests/` to `code/tests/` in a separate reproduction workspace, keep `configs/config.yaml` at the workspace root, and set its `dataset.raw_path` and `dataset.processed_path` to the actual data locations. The copied YAML retains the experiment's relative desktop layout as a record; `src/paths_example.py` lists the directories to configure. Keep the original `src/` snapshot intact.

`src/` contains **no hard-coded local absolute path** (`C:\Users\...`, `/Users/...`, or `/home/...`), so there are no source filename/line entries to list for such paths. `src/pathing.py` resolves paths relative to its location and `configs/config.yaml`; other scripts use this resolver or their own `__file__`-relative root. The copied result logs do contain paths from the original Windows run; see the privacy note below.

## Reproduction order

Run these commands from the root of a **fresh** reproduction workspace after restoring `code/src/`, installing the locked dependencies, and configuring paths. Consult `docs/EXPERIMENT_PLAN.md`, `docs/EXPERIMENT_PLAN_v2_model.md.md`, and `docs/RUN_LOG.md` for the precise selection grids, seed policy, download checksums, and documented deviations. Do not reuse the archived prediction files as destinations for a fresh run.

| Order | Scripts | Main outputs |
|---|---|---|
| 1. Data preparation and leakage checks | `code/src/download_images.py`, `code/src/verify_downloads.py`, `code/src/prepare_data.py`, `code/tests/test_no_leakage.py`, `code/src/cache_images.py` | `data/raw/`, `data/processed/metadata.parquet`, `splits/wc14.csv`, `results/dataset_stats.csv`, duplicate-audit records |
| 2. Feature extraction | `code/src/extract.py --model R50/DINO-B/CLIP-L/SIG2-B`; `code/src/download_v2.py`; `code/src/extract_v2.py --model ... --size ... --view ...` | `features/`, `features_v2/`, feature index and progress files |
| 3. RQ1–RQ4 | `code/src/rq1.py`, `rq2.py`, `rq3.py`, `rq4.py`, then `supplement.py` | `results/rq1_*`, `rq2_*`, `rq3a_*`, `rq3b_*`, `rq4_*` |
| 4. ChronoArch-F | `code/src/run_chronoarch_v2.py --task TASK --model F`, with `TASK` in `wc6`, `wc14`, `gothic`, `romanesque`, `dating`, `wc4_rq4` | `results_v2/selection_log.csv`, `selected_*.json`, `pred_ChronoArch-F_*.csv`, checkpoints |
| 5. Fair baselines | For each of `wc6`, `wc14`, `gothic`, `romanesque`, `dating`: `code/src/fair_baselines_v2.py --task TASK --group linear`, then `--group neural` | `results_v2/baselines_fair/selection_log_*.csv`, `selected_*.json`, `pred_*.csv` |
| 6. Ablations | `code/src/run_ablation_v2.py --task wc6` and `--task gothic`, followed by `code/src/ablation_summary_v2.py` | `results_v2/pred_A*_*.csv`, `pred_R*_*.csv`, `ablation.csv` |
| 7. Statistics | `code/src/summarize_v2.py`, `significance_v2.py`, `gates_attention_v2.py`, `rq4_chronoarch_v2.py`, `compute_cost_v2.py`, `fair_analysis_v2.py` | `results_v2/main_comparison.csv`, `significance.csv`, `gate_by_style.csv`, `rq4_chronoarch*.csv`, `compute_cost.csv`, `baselines_fair/main_comparison_fair.csv`, `significance_fair.csv` |
| 8. Figures and reports | `code/src/plots.py`, `figure_architecture_v2.py`, `gates_attention_v2.py`, `write_results.py`, `write_results_v2.py` | `figures/`, `figures_v2/`, `RESULTS.md`, `RESULTS_v2.md` |

`fair_analysis_v2.py` appends its report section once; it is not an idempotent reporting command. The archived `docs/RESULTS_v2.md` already contains that section. The original R4 report uses seed 0 without flip test augmentation or an ensemble, as documented in `docs/RESULTS_v2.md`.

## Tables and result files

The report headings and figure filenames are used here as identifiers; this mapping does not assign new paper numbers to unnumbered v2 tables.

| Table or reported panel | Archived source files |
|---|---|
| Table 1, RQ1 style attribution | `results/legacy/rq1_main.csv`, `rq1_per_class.csv`, `rq1_comparison.json` |
| Table 2, statistical tests | `results/legacy/rq1_main.csv`, `rq2_mantel.csv`, `rq3b_partial_spearman.csv`, `rq4_tests.csv`; presentation in `docs/RESULTS.md` |
| RQ3a authenticity, Gothic and Romanesque | `results/legacy/rq3a_authenticity.csv`, `rq3a_folds.csv`, `rq3a_confound_controls.csv` |
| RQ3b dating and sensitivity | `results/legacy/rq3b_dating.csv`, `rq3b_b2_fine_style.csv`, `rq3b_sensitivity_exclusions.csv` |
| RQ4 expert/random occlusion and box area | `results/legacy/rq4_occlusion.csv`, `rq4_tests.csv`, `rq4_r_keep_tests.csv`, `rq4_expert_box_area.csv` |
| ChronoArch-F and original B-1–B-7 main tables | `results/v2/main_comparison_all.csv`, `main_comparison.csv`, `per_class_recall.csv` |
| Validation-selected baseline comparisons | `results/v2/significance.csv`, `best_baseline_validation_*.json` |
| Ablations, gate, and ChronoArch RQ4 | `results/v2/ablation.csv`, `gate_by_style.csv`, `rq4_chronoarch_tests.csv` |
| Fair baseline rerun and paired tests | `results/v2/baselines_fair/main_comparison_fair.csv`, `validation_best.csv`, `significance_fair.csv` |

## Figures and their source data

The figure scripts write the files named below. Figure PDFs/PNGs were not copied into this source-and-table snapshot. The six-photo selection manifest already present in this desktop folder is `fig2_photos/photos.csv`.

| Figure output | Source result files or inputs |
|---|---|
| `fig1_dataset` | `results/legacy/dataset_stats.csv` plus separate dataset image and expert-box annotations |
| `fig2_rq1` | `results/legacy/rq1_main.csv`, `rq1_per_class.csv` |
| `fig3_confusion_time` | `results/legacy/rq2_confusion_normalized.csv`, `rq2_class_years.json` |
| `fig3_mantel_scatter` | `results/legacy/rq2_pairs.csv`, `rq2_mantel.csv` |
| `fig3_renaissance_to_baroque_examples` | `results/legacy/rq2_renaissance_to_baroque_images.csv` plus the separate WikiChurches images |
| `fig4_roc` | `results/legacy/rq3a_oof.csv` |
| `fig4_dating` | `results/legacy/rq3b_oof.csv`, `rq3b_dating.csv` |
| `fig5_occlusion` | `results/legacy/rq4_occlusion.csv`, `rq4_random_runs.csv` |
| `fig5_cases` | `results/legacy/rq4_occlusion.csv` plus expert annotations from the separate dataset |
| `model_architecture.pdf` | Architecture defined in `src/chronoarch.py`; generated by `src/figure_architecture_v2.py` |
| `attention_examples.pdf` | Validation-set attention and `results/v2/gate_by_style.csv`; generated by `src/gates_attention_v2.py` |

## Integrity and privacy notes

Training and hyperparameter selection used training/validation splits only. Gothic, Romanesque, and dating selected v2 configurations on the first internal validation fold and reused them for later folds. Each selected final configuration was evaluated on the test split once. `results/v2/selection_log.csv`, `results/v2/baselines_fair/selection_log_*.csv`, `docs/RUN_LOG.md`, and the SHA-256 manifests record selection and code provenance. The test predictions and aggregate tables are archived; weights and large features are omitted.

An automated text scan found no email address, password assignment, API key assignment, or credential-shaped token in the copied source and result text. It did find original machine paths containing `C:\Users\Administrator` in result logs and manifests. These files remain unchanged as requested and should be reviewed before any public upload.
