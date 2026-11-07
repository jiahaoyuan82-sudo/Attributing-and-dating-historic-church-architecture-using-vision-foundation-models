"""B5 validation-selected baseline and paired church-level comparisons."""
import json
import pickle
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd
from scipy.stats import binomtest, norm
from chronoarch_tasks import task_folds, validation_metric
from chronoarch_train import FeatureStore
from pathing import PROCESSED
from run_baselines_v2 import linear_predict
from summarize_v2 import RESULTS, TASKS, metric, model_predictions, primary, stack_parts

BASELINES = ("B-2_224", "B-2_448", "B-3", "B-4", "B-5", "B-6", "B-7")


def best_validation(task, folds):
    log = pd.read_csv(RESULTS / "selection_log.csv")
    scores = {}
    for model in BASELINES:
        if model == "B-6":
            scores[model] = b6_validation(task, folds, log)
            continue
        per_fold = []
        for fold in folds:
            subset = log.loc[(log.model == model) & (log.task == task) &
                             (log.fold == fold.fold)]
            if model in ("B-2_224", "B-2_448", "B-4"):
                chosen = json.loads((RESULTS / f"selected_{model}_{task}_{fold.fold}.json")
                                    .read_text(encoding="utf-8"))["C"]
                subset = subset.loc[np.isclose(pd.to_numeric(subset.config_id), chosen)]
            assert len(subset) == 3
            per_fold.append(float(subset.val_metric.mean()))
        scores[model] = float(np.mean(per_fold))
    winner = max(scores, key=lambda name: (scores[name], name))
    (RESULTS / f"best_baseline_validation_{task}.json").write_text(
        json.dumps({"selected": winner, "validation_scores": scores}, indent=2) + "\n",
        encoding="utf-8")
    return winner, scores


def b6_validation(task, folds, log):
    store = FeatureStore()
    scores = []
    for fold in folds:
        seed_scores = []
        for seed in range(3):
            values = []
            for kind in ("B-6_DINO", "B-6_CLIP", "B-6_SIG"):
                chosen = json.loads((RESULTS / f"selected_{kind}_{task}_{fold.fold}.json")
                                    .read_text(encoding="utf-8"))["C"]
                subset = log.loc[(log.model == kind) & (log.task == task) &
                                 (log.fold == fold.fold) & (log.seed == seed) &
                                 np.isclose(pd.to_numeric(log.config_id, errors="coerce"), chosen)]
                assert len(subset) == 1
                with Path(subset.checkpoint.iloc[0]).open("rb") as stream:
                    model = pickle.load(stream)
                values.append(linear_predict(model, fold, store, fold.val, kind))
            pred = np.mean(values, axis=0)
            truth = fold.year[fold.val] if task == "dating" else fold.style[fold.val]
            seed_scores.append(validation_metric(task, truth, pred))
        scores.append(float(np.mean(seed_scores)))
    return float(np.mean(scores))


def midrank(values):
    order = np.argsort(values)
    sorted_values = values[order]
    ranks = np.empty(len(values))
    index = 0
    while index < len(values):
        end = index + 1
        while end < len(values) and sorted_values[end] == sorted_values[index]:
            end += 1
        ranks[order[index:end]] = (index + end + 1) / 2
        index = end
    return ranks


def delong_p(truth, first, second):
    positive = truth == 1
    m, n = int(positive.sum()), int((~positive).sum())
    assert m > 1 and n > 1
    estimates, v10, v01 = [], [], []
    for score in (first, second):
        pos, neg = score[positive], score[~positive]
        all_ranks = midrank(np.concatenate((pos, neg)))
        pos_ranks, neg_ranks = midrank(pos), midrank(neg)
        auc = (all_ranks[:m].sum() - m * (m + 1) / 2) / (m * n)
        estimates.append(auc)
        v10.append((all_ranks[:m] - pos_ranks) / n)
        v01.append(1 - (all_ranks[m:] - neg_ranks) / m)
    covariance = np.cov(np.stack(v10)) / m + np.cov(np.stack(v01)) / n
    variance = covariance[0, 0] + covariance[1, 1] - 2 * covariance[0, 1]
    return float(2 * norm.sf(abs(estimates[0] - estimates[1]) /
                             np.sqrt(max(variance, 1e-20))))


def bootstrap_difference(task, metadata, left, right, *, draws=1000, seed=0):
    church_groups = metadata.groupby("church_id").indices
    if task in ("wc6", "wc14", "gothic", "romanesque"):
        strata = []
        for label in sorted(metadata.truth.unique()):
            churches = sorted(set(metadata.loc[metadata.truth.eq(label), "church_id"]))
            strata.append([church_groups[church] for church in churches])
    else:
        strata = [list(church_groups.values())]
    rng = np.random.default_rng(seed)
    truth = metadata.truth.to_numpy()
    key = primary(task)
    direction = -1 if task == "dating" else 1
    estimates = []
    for _ in range(draws):
        indices = np.concatenate([group
            for stratum in strata
            for group in (stratum[i] for i in rng.integers(0, len(stratum), len(stratum)))])
        try:
            a = metric(task, truth[indices], left[indices])[key]
            b = metric(task, truth[indices], right[indices])[key]
            estimates.append(direction * (a - b))
        except ValueError:
            continue
    assert len(estimates) >= .95 * draws
    return np.quantile(estimates, [0.025, 0.975]), np.asarray(estimates)


def holm(pvalues):
    order = np.argsort(pvalues)
    corrected = np.empty(len(pvalues))
    running = 0
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (len(pvalues) - rank) * pvalues[index]))
        corrected[index] = running
    return corrected


def old_dinob(task, reference):
    source = pd.read_csv(RESULTS / "b1_reused_oof.csv")
    part = source.loc[source.task.eq(task) & source.model.eq("DINO-B")].copy()
    part = part.set_index("image_id").loc[reference.image_id].reset_index()
    assert part.image_id.tolist() == reference.image_id.tolist()
    assert part.church_id.tolist() == reference.church_id.tolist()
    assert np.allclose(part.truth.to_numpy(), reference.truth.to_numpy())
    columns = [name for name in part if name.startswith("prob_") and part[name].notna().all()]
    return part[columns].to_numpy() if columns else part.prediction.to_numpy()


def main():
    metadata = pd.read_parquet(PROCESSED / "metadata.parquet")
    rows = []
    for task in TASKS:
        folds = task_folds(metadata, task)
        winner, val_scores = best_validation(task, folds)
        right_meta, right = stack_parts(model_predictions(task, winner, folds)[1])
        old = old_dinob(task, right_meta)
        candidates = ((winner, val_scores[winner], right, "validation_best_v2"),
                      ("B-1 DINO-B", np.nan, old, "predeclared_first_round"))
        for proposed in ("ChronoArch-F",):
            left_meta, left = stack_parts(model_predictions(task, proposed, folds)[1])
            assert left_meta[["image_id", "church_id", "truth"]].equals(
                   right_meta[["image_id", "church_id", "truth"]])
            truth = left_meta.truth.to_numpy()
            key = primary(task)
            direction = -1 if task == "dating" else 1
            for baseline, validation_score, comparison, source in candidates:
                delta = direction * (metric(task, truth, left)[key] -
                                     metric(task, truth, comparison)[key])
                ci, bootstrap = bootstrap_difference(task, left_meta, left, comparison)
                if task in ("wc6", "wc14"):
                    a, b = left.argmax(-1) == truth, comparison.argmax(-1) == truth
                    discordant = int(np.count_nonzero(a != b))
                    p = float(binomtest(int(np.count_nonzero(a & ~b)),
                                       discordant, .5).pvalue) if discordant else 1.0
                    method = "exact_McNemar"
                elif task in ("gothic", "romanesque"):
                    p = delong_p(truth, left, comparison)
                    method = "DeLong"
                else:
                    p = float(2 * min((bootstrap <= 0).mean(), (bootstrap >= 0).mean()))
                    method = "paired_church_bootstrap"
                rows.append({"task": task, "proposed": proposed, "baseline": baseline,
                             "baseline_source": source,
                             "validation_baseline_score": validation_score,
                             "metric": key, "proposed_metric": metric(task, truth, left)[key],
                             "baseline_metric": metric(task, truth, comparison)[key],
                             "improvement": delta, "ci_low": ci[0], "ci_high": ci[1],
                             "test": method, "p": p})
    frame = pd.DataFrame(rows)
    frame["p_holm"] = holm(frame.p.to_numpy())
    frame.to_csv(RESULTS / "significance.csv", index=False)
    print(frame.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
