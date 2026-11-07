"""S4 RQ2: historical-distance structure of errors and features."""
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import confusion_matrix

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rq1 import model_features
from pathing import ROOT, PROCESSED

RESULTS = ROOT / "results"
SEED = 0


def mantel(values, years, permutations=10000):
    n = len(years)
    upper = np.triu_indices(n, 1)
    historical = np.abs(years[:, None] - years[None, :])[upper]
    observed = float(spearmanr(values[upper], historical).statistic)
    rng = np.random.default_rng(SEED)
    null = np.empty(permutations)
    for index in range(permutations):
        permuted = rng.permutation(years)
        distance = np.abs(permuted[:, None] - permuted[None, :])[upper]
        null[index] = spearmanr(values[upper], distance).statistic
    p = (1 + np.count_nonzero(np.abs(null) >= abs(observed))) / (permutations + 1)
    return observed, p


def main():
    metadata = pd.read_parquet(PROCESSED / "metadata.parquet")
    comparison = json.loads((RESULTS / "rq1_comparison.json").read_text(encoding="utf-8"))
    best_model = comparison["best_model_selected_by_wc6_val"]
    prediction = np.load(RESULTS / f"rq1_prob_{best_model}_wc14.npz", allow_pickle=True)
    test = metadata[metadata.split_wc14.eq("test")]
    assert prediction["image_id"].tolist() == test.image_id.tolist()
    class_ids = sorted(metadata.wc14_style_id.dropna().unique())
    counts = {}
    years = {}
    for style_id, part in metadata[metadata.wc14_style_id.notna()].groupby("wc14_style_id"):
        churches = part.dropna(subset=["construction_year"]).drop_duplicates("church_id")
        counts[style_id] = len(churches)
        if len(churches) >= 10:
            years[style_id] = float(churches.construction_year.median())
    valid = sorted(years, key=years.get)
    excluded = [style for style in class_ids if style not in years]
    # Keep all predictions in each row denominator, including predictions
    # into classes omitted from the year-distance test for sparse dating.
    full_matrix = confusion_matrix(prediction["true"], prediction["pred"], labels=class_ids)
    full_normalized = full_matrix / full_matrix.sum(axis=1, keepdims=True)
    selected = [class_ids.index(style) for style in valid]
    normalized = full_normalized[np.ix_(selected, selected)]
    confusion = (normalized + normalized.T) / 2
    year_vector = np.array([years[style] for style in valid])
    confusion_r, confusion_p = mantel(confusion, year_vector)
    features = model_features(best_model, metadata)
    center_vectors = []
    for style_id in valid:
        mask = metadata.wc14_style_id.eq(style_id) & metadata.split_wc14.eq("train")
        center = features[mask.to_numpy()].mean(axis=0)
        center_vectors.append(center / np.linalg.norm(center))
    centers = np.stack(center_vectors)
    center_distance = 1 - centers @ centers.T
    center_r, center_p = mantel(center_distance, year_vector)
    pd.DataFrame(normalized, index=valid, columns=valid).to_csv(RESULTS / "rq2_confusion_normalized.csv")
    rows = []
    for i in range(len(valid)):
        for j in range(i + 1, len(valid)):
            rows.append({"style_a": valid[i], "style_b": valid[j], "year_a": year_vector[i], "year_b": year_vector[j],
                         "year_distance": abs(year_vector[i] - year_vector[j]), "symmetric_confusion": confusion[i, j],
                         "center_cosine_distance": center_distance[i, j]})
    pd.DataFrame(rows).to_csv(RESULTS / "rq2_pairs.csv", index=False)
    pd.DataFrame([
        {"test": "confusion_vs_year_distance", "model": best_model, "spearman_r": confusion_r, "permutation_p": confusion_p, "permutations": 10000, "classes": len(valid)},
        {"test": "center_distance_vs_year_distance", "model": best_model, "spearman_r": center_r, "permutation_p": center_p, "permutations": 10000, "classes": len(valid)},
    ]).to_csv(RESULTS / "rq2_mantel.csv", index=False)
    (RESULTS / "rq2_class_years.json").write_text(json.dumps({"representative_years": years, "year_church_counts": counts, "excluded_classes_under_10_churches": excluded, "ordered_styles": valid}, indent=2), encoding="utf-8")
    misclassified = (prediction["true"] == "Q236122") & (prediction["pred"] == "Q840829")
    pd.DataFrame({"image_id": prediction["image_id"][misclassified][:24]}).to_csv(RESULTS / "rq2_renaissance_to_baroque_images.csv", index=False)
    print(f"RQ2 model={best_model}, classes={len(valid)}, excluded={excluded}, confusion r={confusion_r:.4f}, p={confusion_p:.6f}, center r={center_r:.4f}, p={center_p:.6f}")


if __name__ == "__main__":
    main()
