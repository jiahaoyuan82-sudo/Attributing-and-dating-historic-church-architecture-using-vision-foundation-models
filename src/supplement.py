"""Post-S8 supplemental B2 and RQ4 analyses using the original seed and models."""
import json
import time
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw
from scipy.stats import spearmanr, wilcoxon
from sklearn.metrics import mean_absolute_error, median_absolute_error

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pathing import ROOT, RAW, PROCESSED
from models import FrozenModel
from rq1 import model_features
from rq3 import dating_splits, cluster_metric_ci
from rq4 import boxes_for, occlude, random_boxes, tune_wc4_classifier

RESULTS = ROOT / "results"
SEED = 0


def run_b2(metadata):
    exclusions = set(pd.read_csv(RESULTS / "rq3b_sensitivity_exclusions.csv").image_id)
    old = pd.read_csv(RESULTS / "rq3b_dating.csv")
    rows, oof = [], []
    for dataset, exclusion in (("main", False), ("conflict_excluded", True)):
        mask = metadata.wc14_style_id.notna() & metadata.construction_year.between(800, 2020)
        if exclusion:
            mask &= ~metadata.image_id.isin(exclusions)
        part = metadata[mask].copy()
        years = part.construction_year.to_numpy(dtype=float)
        styles = part.wc14_style_id.to_numpy()
        churches = part.church_id.to_numpy()
        fine = part.all_styles.map(json.loads)
        assert fine.map(len).eq(1).all(), "B2 requires a unique finest style label"
        fine = fine.map(lambda values: values[0]).to_numpy()
        prediction = np.empty(len(part), dtype=float)
        folds = dating_splits(styles, churches)
        for fold, (train, test) in enumerate(folds):
            assert not set(churches[train]) & set(churches[test])
            global_median = float(np.median(years[train]))
            medians = {style: float(np.median(years[train][fine[train] == style])) for style in np.unique(fine[train])}
            prediction[test] = [medians.get(style, global_median) for style in fine[test]]
            oof.extend({"dataset": dataset, "fold": fold, "image_id": part.image_id.iloc[i],
                        "church_id": churches[i], "fine_style_id": fine[i], "year": years[i],
                        "prediction": prediction[i], "fallback_global_median": fine[i] not in medians}
                       for i in test)
        mae_ci, medae_ci, rho_ci = cluster_metric_ci(years, prediction, churches)
        rows.append({"dataset": dataset, "model": "baseline", "method": "B2_fine_style_median",
                     "images": len(part), "churches": part.church_id.nunique(),
                     "fine_styles": len(np.unique(fine)),
                     "fallback_images": sum(row["fallback_global_median"] for row in oof if row["dataset"] == dataset),
                     "mae": mean_absolute_error(years, prediction), "mae_ci_low": mae_ci[0], "mae_ci_high": mae_ci[1],
                     "median_ae": median_absolute_error(years, prediction), "median_ae_ci_low": medae_ci[0],
                     "median_ae_ci_high": medae_ci[1], "spearman_rho": float(spearmanr(years, prediction).statistic),
                     "rho_ci_low": rho_ci[0], "rho_ci_high": rho_ci[1]})
        assert len(old[(old.dataset == dataset) & (old.method == "B1_true_style_median")]) == 1
    pd.DataFrame(rows).to_csv(RESULTS / "rq3b_b2_fine_style.csv", index=False)
    pd.DataFrame(oof).to_csv(RESULTS / "rq3b_b2_oof.csv", index=False)
    print("B2 complete", flush=True)


def union_fraction(boxes, image_size):
    mask = Image.new("1", image_size, 0)
    draw = ImageDraw.Draw(mask)
    width, height = image_size
    for x, y, w, h in boxes:
        draw.rectangle((round(x * width), round(y * height), round((x + w) * width), round((y + h) * height)), fill=1)
    return np.asarray(mask, dtype=np.uint8).sum() / (width * height)


def run_rq4(metadata):
    annotations = json.loads((RAW / "building_parts.json").read_text(encoding="utf-8"))["annotations"]
    ids = list(annotations)
    prior = pd.read_csv(RESULTS / "rq4_occlusion.csv")
    index = {image_id: i for i, image_id in enumerate(metadata.image_id)}
    area_rows, per_image, draws, k_checks = [], [], [], []
    for image_id in ids:
        with Image.open(PROCESSED / "img256" / image_id) as source:
            size = source.size
        boxes = boxes_for(annotations[image_id])
        area_rows.append({"image_id": image_id, "church_id": metadata.church_id.iloc[index[image_id]],
                          "box_count": len(boxes), "image_width": size[0], "image_height": size[1],
                          "sum_area_fraction": sum(w * h for _, _, w, h in boxes),
                          "union_area_fraction": union_fraction(boxes, size)})
    area = pd.DataFrame(area_rows)
    area.to_csv(RESULTS / "rq4_expert_box_area.csv", index=False)
    bins = np.array([0, .05, .10, .20, .30, .40, .50, .60, .70, .80, .90, 1.0])
    counts, _ = np.histogram(area.union_area_fraction, bins)
    pd.DataFrame({"lower_fraction_inclusive": bins[:-1], "upper_fraction_exclusive_except_last": bins[1:],
                  "images": counts}).to_csv(RESULTS / "rq4_expert_box_area_distribution.csv", index=False)
    for model_name in ("DINO-B", "R50"):
        print(f"R_keep {model_name} starting CUDA", flush=True)
        frozen = FrozenModel(model_name, device="cuda")
        features = model_features(model_name, metadata)
        classifier, counts = tune_wc4_classifier(features, metadata, ids)
        old_info = json.loads((RESULTS / "rq4_classifier_info.json").read_text(encoding="utf-8"))[model_name]
        assert counts["selected_C"] == old_info["selected_C"]
        old_rows = prior[prior.model == model_name].set_index("image_id")
        class_to_index = {name: i for i, name in enumerate(classifier.classes_)}
        rng = np.random.default_rng(SEED)
        for sequence, image_id in enumerate(ids, 1):
            annotation = annotations[image_id]
            expert = boxes_for(annotation)
            true_indices = [class_to_index[style] for style in annotation["styles"]]
            original = classifier.predict_proba(features[index[image_id]][None, :])[0]
            original_top = classifier.classes_[original.argmax()]
            original_correct = original_top in annotation["styles"]
            with Image.open(PROCESSED / "img256" / image_id) as source:
                image = source.convert("RGB")
            variants = [occlude(image, expert, keep_only=True)]
            variants.extend(occlude(image, random_boxes(expert, rng), keep_only=True) for _ in range(20))
            tensors = torch.stack([frozen.transform(item) for item in variants])
            probabilities = classifier.predict_proba(frozen.image_features(tensors).numpy())
            k_probability = probabilities[0]
            true_p = probabilities[1:, true_indices].sum(axis=1)
            correct = np.isin(classifier.classes_[probabilities[1:].argmax(axis=1)], annotation["styles"])
            previous = old_rows.loc[image_id]
            assert abs(float(original[true_indices].sum()) - previous.p_original) < 1e-5
            new_p_k = float(k_probability[true_indices].sum())
            k_difference = new_p_k - previous.p_K
            k_checks.append({"model": model_name, "image_id": image_id, "old_p_K": previous.p_K,
                             "recomputed_p_K": new_p_k, "difference": k_difference})
            assert abs(k_difference) < 0.005, f"K probability mismatch: {model_name} {image_id} {k_difference}"
            row = {"model": model_name, "image_id": image_id, "church_id": metadata.church_id.iloc[index[image_id]],
                   "original_correct": bool(original_correct), "p_original": previous.p_original,
                   "p_K": new_p_k, "p_R_keep_mean": float(true_p.mean()),
                   "correct_K": bool(classifier.classes_[k_probability.argmax()] in annotation["styles"]),
                   "correct_R_keep_rate": float(correct.mean())}
            per_image.append(row)
            for draw, (p, c) in enumerate(zip(true_p, correct)):
                draws.append({"model": model_name, "image_id": image_id, "draw": draw,
                              "p_true": float(p), "top1_correct": bool(c)})
            if sequence % 20 == 0 or sequence == len(ids):
                print(f"R_keep {model_name}: {sequence}/{len(ids)}", flush=True)
        del frozen
        torch.cuda.empty_cache()
    pd.DataFrame(per_image).to_csv(RESULTS / "rq4_r_keep_per_image.csv", index=False)
    pd.DataFrame(draws).to_csv(RESULTS / "rq4_r_keep_draws.csv", index=False)
    pd.DataFrame(k_checks).to_csv(RESULTS / "rq4_k_recompute_checks.csv", index=False)
    summarize_rq4(pd.DataFrame(per_image))


def summarize_rq4(frame):
    rows = []
    for model, part in frame.groupby("model"):
        retained = part[part.original_correct]
        for metric, left, right, sample in (
            ("true_class_probability", "p_K", "p_R_keep_mean", part),
            ("top1_retention", "correct_K", "correct_R_keep_rate", retained)):
            difference = sample[left].to_numpy(dtype=float) - sample[right].to_numpy(dtype=float)
            test = wilcoxon(difference, alternative="two-sided", method="approx")
            groups = list(sample.groupby("church_id").indices.values())
            rng = np.random.default_rng(SEED)
            boot = np.array([difference[np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])].mean()
                             for _ in range(1000)])
            ci = np.quantile(boot, [.025, .975])
            rows.append({"model": model, "metric": metric, "images": len(sample),
                         "K_mean": sample[left].mean(), "R_keep_mean": sample[right].mean(),
                         "K_minus_R_keep": difference.mean(), "ci_low": ci[0], "ci_high": ci[1],
                         "wilcoxon_statistic": test.statistic, "wilcoxon_p": test.pvalue})
    pd.DataFrame(rows).to_csv(RESULTS / "rq4_r_keep_tests.csv", index=False)


if __name__ == "__main__":
    start = time.time()
    metadata = pd.read_parquet(PROCESSED / "metadata.parquet")
    run_b2(metadata)
    run_rq4(metadata)
    print(f"Supplement elapsed {time.time()-start:.1f}s", flush=True)
