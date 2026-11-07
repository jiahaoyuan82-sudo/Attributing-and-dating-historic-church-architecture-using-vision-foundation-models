"""S3 RQ1: zero-shot, cosine kNN and balanced linear probes."""
import json
from collections import defaultdict
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score, recall_score
from sklearn.neighbors import KNeighborsClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pathing import ROOT, PROCESSED
RESULTS = ROOT / "results"
FEATURES = ROOT / "features"
SEED = 0
MODELS = ["R50", "DINO-B", "CLIP-L", "SIG2-B"]


def cluster_bootstrap_ci(truth, prediction, churches, classes, iterations=1000):
    rng = np.random.default_rng(SEED)
    church_to_indices = defaultdict(list)
    church_class = {}
    for index, (church, label) in enumerate(zip(churches, truth)):
        church_to_indices[church].append(index)
        church_class[church] = label
    by_class = {label: [church for church, value in church_class.items() if value == label] for label in classes}
    estimates = []
    for _ in range(iterations):
        selected = []
        for label in classes:
            group = by_class[label]
            drawn = rng.choice(group, size=len(group), replace=True)
            for church in drawn:
                selected.extend(church_to_indices[church])
        estimates.append(balanced_accuracy_score(truth[selected], prediction[selected]))
    return np.quantile(estimates, [0.025, 0.975]).tolist()


def paired_bootstrap_ci(truth, pred_a, pred_b, churches, classes, iterations=1000):
    rng = np.random.default_rng(SEED)
    church_to_indices = defaultdict(list)
    church_class = {}
    for index, (church, label) in enumerate(zip(churches, truth)):
        church_to_indices[church].append(index)
        church_class[church] = label
    by_class = {label: [church for church, value in church_class.items() if value == label] for label in classes}
    differences = []
    for _ in range(iterations):
        selected = []
        for label in classes:
            group = by_class[label]
            for church in rng.choice(group, size=len(group), replace=True):
                selected.extend(church_to_indices[church])
        differences.append(balanced_accuracy_score(truth[selected], pred_a[selected]) - balanced_accuracy_score(truth[selected], pred_b[selected]))
    return np.quantile(differences, [0.025, 0.975]).tolist()


def model_features(model, metadata):
    ids = pd.read_csv(FEATURES / f"{model}_image_ids.csv").image_id.tolist()
    assert ids == metadata.image_id.tolist(), model
    features = np.load(FEATURES / f"{model}.npy")
    assert len(features) == len(metadata)
    return features


def tune_linear(train_x, train_y, val_x, val_y):
    best = None
    for c in [0.01, 0.1, 1, 10]:
        estimator = LogisticRegression(C=c, class_weight="balanced", max_iter=3000, random_state=SEED)
        estimator.fit(train_x, train_y)
        score = balanced_accuracy_score(val_y, estimator.predict(val_x))
        if best is None or score > best[0]:
            best = (score, c)
    return best


def tune_knn(train_x, train_y, val_x, val_y):
    best = None
    for k in [5, 10, 20, 50]:
        estimator = KNeighborsClassifier(n_neighbors=k, metric="cosine", weights="uniform", n_jobs=-1)
        estimator.fit(train_x, train_y)
        score = balanced_accuracy_score(val_y, estimator.predict(val_x))
        if best is None or score > best[0]:
            best = (score, k)
    return best


def main():
    metadata = pd.read_parquet(PROCESSED / "metadata.parquet")
    rows, class_rows, linear_val = [], [], {}
    linear_predictions = {}
    for task in ("wc6", "wc14"):
        label_column, split_column = f"{task}_style_id", f"split_{task}"
        labels = metadata[label_column]
        class_ids = sorted(labels.dropna().unique())
        indices = {split: np.flatnonzero(metadata[split_column].eq(split).to_numpy()) for split in ("train", "val", "test")}
        train, val, test = (indices[key] for key in ("train", "val", "test"))
        truth = labels.iloc[test].to_numpy()
        churches = metadata.church_id.iloc[test].to_numpy()
        for model in MODELS:
            features = model_features(model, metadata)
            protocols = []
            score, k = tune_knn(features[train], labels.iloc[train], features[val], labels.iloc[val])
            knn = KNeighborsClassifier(n_neighbors=k, metric="cosine", weights="uniform", n_jobs=-1)
            knn.fit(features[np.r_[train, val]], labels.iloc[np.r_[train, val]])
            protocols.append(("knn", knn.predict(features[test]), score, k, None))
            score, c = tune_linear(features[train], labels.iloc[train], features[val], labels.iloc[val])
            linear_val[(task, model)] = score
            linear = LogisticRegression(C=c, class_weight="balanced", max_iter=3000, random_state=SEED)
            linear.fit(features[np.r_[train, val]], labels.iloc[np.r_[train, val]])
            probabilities = linear.predict_proba(features[test])
            prediction = linear.classes_[probabilities.argmax(axis=1)]
            protocols.append(("linear_probe", prediction, score, c, probabilities))
            linear_predictions[(task, model)] = prediction
            np.savez_compressed(RESULTS / f"rq1_prob_{model}_{task}.npz", image_id=metadata.image_id.iloc[test].to_numpy(), church_id=churches, true=truth, pred=prediction, classes=linear.classes_, probabilities=probabilities)
            if model in ("CLIP-L", "SIG2-B"):
                text_ids = json.loads((FEATURES / f"{model}_text_{task}_styles.json").read_text(encoding="utf-8"))
                assert text_ids == class_ids
                text_vectors = np.load(FEATURES / f"{model}_text_{task}.npy")
                zero_prediction = np.array(class_ids)[(features[test] @ text_vectors.T).argmax(axis=1)]
                protocols.append(("zero_shot", zero_prediction, None, None, None))
            for protocol, prediction, val_score, parameter, _ in protocols:
                bal_acc = balanced_accuracy_score(truth, prediction)
                macro_f1 = f1_score(truth, prediction, labels=class_ids, average="macro", zero_division=0)
                ci_low, ci_high = cluster_bootstrap_ci(truth, prediction, churches, class_ids)
                rows.append({"model": model, "protocol": protocol, "task": task, "bal_acc": bal_acc, "macro_f1": macro_f1, "ci_low": ci_low, "ci_high": ci_high, "val_bal_acc": val_score, "selected_parameter": parameter})
                recalls = recall_score(truth, prediction, labels=class_ids, average=None, zero_division=0)
                for style_id, recall in zip(class_ids, recalls):
                    class_rows.append({"model": model, "protocol": protocol, "task": task, "style_id": style_id, "recall": recall, "test_images": int((truth == style_id).sum())})
            print(f"RQ1 {task} {model} complete", flush=True)
    pd.DataFrame(rows).to_csv(RESULTS / "rq1_main.csv", index=False)
    pd.DataFrame(class_rows).to_csv(RESULTS / "rq1_per_class.csv", index=False)
    best_model = max(MODELS, key=lambda model: linear_val[("wc6", model)])
    test = np.flatnonzero(metadata.split_wc6.eq("test").to_numpy())
    truth = metadata.wc6_style_id.iloc[test].to_numpy()
    churches = metadata.church_id.iloc[test].to_numpy()
    best = linear_predictions[("wc6", best_model)]
    baseline = linear_predictions[("wc6", "R50")]
    classes = sorted(metadata.wc6_style_id.dropna().unique())
    ci_low, ci_high = paired_bootstrap_ci(truth, best, baseline, churches, classes)
    best_only = int(((best == truth) & (baseline != truth)).sum())
    baseline_only = int(((best != truth) & (baseline == truth)).sum())
    p_value = binomtest(min(best_only, baseline_only), best_only + baseline_only, 0.5).pvalue if best_only + baseline_only else 1.0
    comparison = {"best_model_selected_by_wc6_val": best_model, "best_model_wc6_val_bal_acc": linear_val[("wc6", best_model)],
                  "paired_bal_acc_diff": balanced_accuracy_score(truth, best) - balanced_accuracy_score(truth, baseline),
                  "paired_ci_low": ci_low, "paired_ci_high": ci_high, "mcnemar_exact_p": p_value,
                  "best_only_correct": best_only, "r50_only_correct": baseline_only}
    (RESULTS / "rq1_comparison.json").write_text(json.dumps(comparison, indent=2), encoding="utf-8")
    print(json.dumps(comparison, indent=2), flush=True)


if __name__ == "__main__":
    main()
