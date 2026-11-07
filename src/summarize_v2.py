"""B5/B7 metrics from saved one-time test predictions; no model forwards."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import (balanced_accuracy_score, f1_score,
                             mean_absolute_error, roc_auc_score)
from chronoarch_tasks import task_folds
from pathing import PROCESSED

RESULTS = ROOT / "results_v2"
TASKS = ("wc6", "wc14", "gothic", "romanesque", "dating")
MODELS = ("ChronoArch-F", "B-2_224", "B-2_448",
          "B-3", "B-4", "B-5", "B-6", "B-7")


def metric(task, truth, prediction):
    if task in ("wc6", "wc14"):
        labels = prediction.argmax(axis=1)
        return {"balanced_accuracy": balanced_accuracy_score(truth, labels),
                "macro_f1": f1_score(truth, labels, average="macro")}
    if task in ("gothic", "romanesque"):
        return {"auroc": roc_auc_score(truth, prediction)}
    error = np.abs(truth - prediction)
    return {"mae": float(error.mean()), "median_ae": float(np.median(error)),
            "spearman_rho": float(spearmanr(truth, prediction).statistic)}


def primary(task):
    return ("balanced_accuracy" if task in ("wc6", "wc14") else
            "auroc" if task in ("gothic", "romanesque") else "mae")


def prediction_values(frame):
    columns = sorted((name for name in frame if name.startswith("prob_")),
                     key=lambda name: int(name.split("_")[1]))
    return frame[columns].to_numpy() if columns else frame.prediction.to_numpy()


def model_predictions(task, model, folds):
    seeds = range(3)
    seed_parts = {seed: [] for seed in seeds}
    ensemble_parts = []
    for fold in folds:
        frames = [pd.read_csv(RESULTS / f"pred_{model}_{task}_{fold.fold}_{seed}.csv")
                  for seed in seeds]
        assert all(frame.image_id.tolist() == frames[0].image_id.tolist()
                   for frame in frames)
        assert frames[0].image_id.tolist() == fold.image_ids[fold.test].tolist()
        values = [prediction_values(frame) for frame in frames]
        for seed, frame, prediction in zip(seeds, frames, values):
            seed_parts[seed].append((frame, prediction))
        ensemble_parts.append((frames[0], np.mean(values, axis=0)))
    return seed_parts, ensemble_parts


def stack_parts(parts):
    frames = [frame for frame, _ in parts]
    prediction = np.concatenate([value for _, value in parts], axis=0)
    metadata = pd.concat(frames, ignore_index=True)[["image_id", "church_id", "truth"]]
    assert metadata.image_id.is_unique
    return metadata, prediction


def church_bootstrap(task, metadata, prediction, *, draws=1000, seed=0):
    church_groups = metadata.groupby("church_id").indices
    if task in ("wc6", "wc14", "gothic", "romanesque"):
        strata = []
        for label in sorted(metadata.truth.unique()):
            churches = sorted(set(metadata.loc[metadata.truth.eq(label), "church_id"]))
            strata.append([church_groups[church] for church in churches])
    else:
        strata = [list(church_groups.values())]
    rng = np.random.default_rng(seed)
    key = primary(task)
    samples = []
    truth = metadata.truth.to_numpy()
    for _ in range(draws):
        indices = np.concatenate([group
            for stratum in strata
            for group in (stratum[i] for i in rng.integers(0, len(stratum), len(stratum)))])
        try:
            samples.append(metric(task, truth[indices], prediction[indices])[key])
        except ValueError:
            continue
    assert len(samples) >= 0.95 * draws
    return np.quantile(samples, [0.025, 0.975])


def main():
    metadata = pd.read_parquet(PROCESSED / "metadata.parquet")
    comparisons, oof_rows, class_rows = [], [], []
    for task in TASKS:
        folds = task_folds(metadata, task)
        for model in MODELS:
            seed_parts, ensemble_parts = model_predictions(task, model, folds)
            per_seed = []
            for seed, parts in seed_parts.items():
                frame, values = stack_parts(parts)
                per_seed.append(metric(task, frame.truth.to_numpy(), values))
            frame, ensemble = stack_parts(ensemble_parts)
            score = metric(task, frame.truth.to_numpy(), ensemble)
            ci = church_bootstrap(task, frame, ensemble)
            row = {"task": task, "model": model, "images": len(frame),
                   "churches": frame.church_id.nunique(), "seeds": len(per_seed),
                   "primary_metric": primary(task),
                   "primary_seed_mean": np.mean([item[primary(task)] for item in per_seed]),
                   "primary_seed_sd": np.std([item[primary(task)] for item in per_seed], ddof=1),
                   "primary_ensemble": score[primary(task)],
                   "primary_ci_low": ci[0], "primary_ci_high": ci[1]}
            row.update({f"ensemble_{name}": value for name, value in score.items()})
            comparisons.append(row)
            output = frame.copy()
            output["task"], output["model"] = task, model
            if ensemble.ndim == 1:
                output["prediction"] = ensemble
            else:
                for index in range(ensemble.shape[1]):
                    output[f"prob_{index}"] = ensemble[:, index]
            oof_rows.append(output)
            if task in ("wc6", "wc14"):
                labels = ensemble.argmax(axis=1)
                for class_index, name in enumerate(folds[0].style_names):
                    relevant = frame.truth.to_numpy() == class_index
                    class_rows.append({"task": task, "model": model,
                                       "class_id": name, "images": int(relevant.sum()),
                                       "recall": float((labels[relevant] == class_index).mean())})
            print(f"Summarized {task} {model}", flush=True)
    pd.DataFrame(comparisons).to_csv(RESULTS / "main_comparison.csv", index=False)
    pd.concat(oof_rows, ignore_index=True).to_csv(RESULTS / "ensemble_oof.csv", index=False)
    pd.DataFrame(class_rows).to_csv(RESULTS / "per_class_recall.csv", index=False)


if __name__ == "__main__":
    main()
