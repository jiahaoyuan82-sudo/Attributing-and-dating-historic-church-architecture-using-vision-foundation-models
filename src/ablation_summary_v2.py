"""B4 three-seed ablation metrics and church-bootstrap differences."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd
from chronoarch_tasks import task_folds
from pathing import PROCESSED
from summarize_v2 import RESULTS, metric, prediction_values, primary
from significance_v2 import bootstrap_difference

VARIANTS = ("A0", "A1", "A2", "A3", "A4", "R1", "R2", "R4")


def ensemble(task, folds, variant, *, single=False):
    if variant == "A0":
        model, seeds = "ChronoArch-F", range(1) if single else range(3)
    else:
        model, seeds = variant, range(1) if variant == "R4" else range(3)
    parts = []
    seed_parts = {seed: [] for seed in seeds}
    for fold in folds:
        frames = [pd.read_csv(RESULTS / f"pred_{model}_{task}_{fold.fold}_{seed}.csv")
                  for seed in seeds]
        assert all(frame.image_id.tolist() == frames[0].image_id.tolist()
                   for frame in frames)
        values = [prediction_values(frame) for frame in frames]
        parts.append((frames[0][["image_id", "church_id", "truth"]],
                      np.mean(values, axis=0)))
        for seed, frame, value in zip(seeds, frames, values):
            seed_parts[seed].append((frame[["image_id", "church_id", "truth"]], value))
    info = pd.concat([frame for frame, _ in parts], ignore_index=True)
    prediction = np.concatenate([value for _, value in parts])
    metrics = []
    for seed, seed_fold_parts in seed_parts.items():
        truth = pd.concat([frame for frame, _ in seed_fold_parts]).truth.to_numpy()
        value = np.concatenate([value for _, value in seed_fold_parts])
        metrics.append(metric(task, truth, value)[primary(task)])
    return info, prediction, metrics


def main():
    metadata = pd.read_parquet(PROCESSED / "metadata.parquet")
    rows = []
    for task in ("wc6", "gothic"):
        folds = task_folds(metadata, task)
        reference_info, reference, _ = ensemble(task, folds, "A0")
        for variant in VARIANTS:
            if variant == "R1" and task != "wc6":
                continue
            info, pred, seeds = ensemble(task, folds, variant)
            assert info.equals(reference_info)
            truth = info.truth.to_numpy()
            key = primary(task)
            score = metric(task, truth, pred)[key]
            comparison_info, comparison_pred = reference_info, reference
            comparison_name = "A0"
            if variant == "R4":
                comparison_info, comparison_pred, _ = ensemble(task, folds, "A0", single=True)
                assert comparison_info.equals(info)
                comparison_name = "A0_seed0"
            if variant == "A0":
                improvement, ci = 0.0, (0.0, 0.0)
            else:
                improvement = score - metric(task, truth, comparison_pred)[key]
                ci, _ = bootstrap_difference(task, info, pred, comparison_pred)
            rows.append({"task": task, "variant": variant, "images": len(info),
                         "seeds": len(seeds), "metric": key,
                         "seed_mean": np.mean(seeds),
                         "seed_sd": np.std(seeds, ddof=1) if len(seeds) > 1 else np.nan,
                         "ensemble": score, "comparison": comparison_name,
                         "delta_vs_comparison": improvement,
                         "delta_ci_low": ci[0], "delta_ci_high": ci[1]})
    frame = pd.DataFrame(rows)
    frame.to_csv(RESULTS / "ablation.csv", index=False)
    print(frame.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
