"""S5 RQ3a authenticity and RQ3b dating, including sensitivity analyses."""
import json
import time
from collections import defaultdict
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from scipy.stats import pearsonr, spearmanr
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import balanced_accuracy_score, mean_absolute_error, median_absolute_error, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from scipy.stats import rankdata

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rq1 import MODELS, model_features
from pathing import ROOT, RAW, PROCESSED, image_path

RESULTS = ROOT / "results"
FEATURES = ROOT / "features"
SEED = 0


def style_parents():
    return {child: parent for parent, child in (line.split() for line in (RAW / "parent_child_rel.txt").read_text(encoding="utf-8").splitlines())}


def has_ancestor(style, ancestor, parents):
    while True:
        if style == ancestor:
            return True
        if style not in parents:
            return False
        style = parents[style]


def tagged(raw, ancestor, parents):
    return any(has_ancestor(style, ancestor, parents) for style in json.loads(raw))


def authenticity_groups(metadata):
    parents = style_parents()
    year = metadata.construction_year
    result = {}
    specs = [("Gothic", "Q186363", 1550), ("Romanesque", "Q744373", 1250)]
    for style, revival_id, original_max in specs:
        root = metadata.level1_style.eq(style)
        revival_tag = metadata.all_styles.map(lambda raw: tagged(raw, revival_id, parents))
        conflict = revival_tag & year.le(original_max)
        original = root & ~revival_tag & year.le(original_max)
        mixed_revival = (revival_tag | (root & year.ge(1800))) & ~conflict
        strict_revival = root & year.ge(1800)
        assert not (original & mixed_revival).any()
        result[style] = {"mixed": (original.to_numpy(), mixed_revival.to_numpy()),
                         "strict_year": (original.to_numpy(), strict_revival.to_numpy()),
                         "conflict": conflict.to_numpy()}
    return result


def choose_logistic(x, y, groups):
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    inner_train, inner_val = next(splitter.split(x, y, groups))
    best = None
    for c in [0.01, 0.1, 1, 10]:
        model = LogisticRegression(C=c, class_weight="balanced", max_iter=3000, random_state=SEED)
        model.fit(x[inner_train], y[inner_train])
        score = balanced_accuracy_score(y[inner_val], model.predict(x[inner_val]))
        if best is None or score > best[0]:
            best = (score, c)
    return best[1]


def cross_validate_authenticity(x, y, groups, image_ids, country, style, rule, model_name):
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    folds, predictions = [], []
    for fold, (train, test) in enumerate(splitter.split(x, y, groups)):
        assert not set(groups[train]) & set(groups[test])
        assert len(np.unique(y[test])) == 2
        c = choose_logistic(x[train], y[train], groups[train])
        estimator = LogisticRegression(C=c, class_weight="balanced", max_iter=3000, random_state=SEED)
        estimator.fit(x[train], y[train])
        probability = estimator.predict_proba(x[test])[:, 1]
        predicted = (probability >= 0.5).astype(int)
        auc = roc_auc_score(y[test], probability)
        balanced = balanced_accuracy_score(y[test], predicted)
        folds.append({"fold": fold, "auroc": auc, "bal_acc": balanced, "C": c})
        predictions.extend({"style": style, "rule": rule, "model": model_name, "fold": fold, "image_id": image_ids[i],
                            "church_id": groups[i], "country": country[i], "true": int(y[i]), "score": float(probability[j])}
                           for j, i in enumerate(test))
    return folds, predictions


def authenticity_cluster_ci(predictions, iterations=1000):
    frame = pd.DataFrame(predictions)
    grouped = frame.groupby("church_id").indices
    church_class = frame.groupby("church_id").true.first()
    by_class = {label: church_class.index[church_class.eq(label)].to_numpy() for label in (0, 1)}
    rng = np.random.default_rng(SEED)
    estimates = np.empty((iterations, 2))
    for draw in range(iterations):
        selected = np.concatenate([grouped[church] for label in (0, 1)
                                   for church in rng.choice(by_class[label], size=len(by_class[label]), replace=True)])
        truth = frame.true.to_numpy()[selected]
        score = frame.score.to_numpy()[selected]
        estimates[draw] = (roc_auc_score(truth, score), balanced_accuracy_score(truth, score >= 0.5))
    return np.quantile(estimates, [0.025, 0.975], axis=0)


def summarize_authenticity(folds, predictions, y, style, rule, model):
    ci = authenticity_cluster_ci(predictions)
    return {"style": style, "rule": rule, "model": model, "images": len(y), "original_images": int((y == 0).sum()),
            "revival_images": int((y == 1).sum()), "auroc_mean": np.mean([f["auroc"] for f in folds]),
            "auroc_sd": np.std([f["auroc"] for f in folds], ddof=1),
            "bal_acc_mean": np.mean([f["bal_acc"] for f in folds]),
            "bal_acc_sd": np.std([f["bal_acc"] for f in folds], ddof=1),
            "oof_auroc_ci_low": ci[0, 0], "oof_auroc_ci_high": ci[1, 0],
            "oof_bal_acc_ci_low": ci[0, 1], "oof_bal_acc_ci_high": ci[1, 1]}


def property_control(properties, metadata, indices, y, groups, style):
    subset = properties.set_index("image_id").loc[metadata.image_id.iloc[indices]]
    x = subset[["original_width", "original_height", "original_file_bytes", "exif_year"]].to_numpy(dtype=float)
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    aucs = []
    for train, test in splitter.split(x, y, groups):
        estimator = make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler(),
                                  LogisticRegression(C=1, class_weight="balanced", max_iter=3000, random_state=SEED))
        estimator.fit(x[train], y[train])
        aucs.append(roc_auc_score(y[test], estimator.predict_proba(x[test])[:, 1]))
    return {"style": style, "control": "image_properties_only", "auroc_mean": np.mean(aucs), "auroc_sd": np.std(aucs, ddof=1),
            "images": len(y), "exif_year_present": int(np.isfinite(x[:, 3]).sum())}


def recompressed_features(metadata, indices, style, model_name):
    import torch
    from models import FrozenModel
    output = FEATURES / f"{model_name}_recompressed_{style.lower()}.npy"
    ids_file = FEATURES / f"{model_name}_recompressed_{style.lower()}_image_ids.csv"
    image_ids = metadata.image_id.iloc[indices].tolist()
    if output.exists() and ids_file.exists() and pd.read_csv(ids_file).image_id.tolist() == image_ids:
        return np.load(output)
    cache = PROCESSED / "recompressed224_q75" / style.lower()
    cache.mkdir(parents=True, exist_ok=True)
    for row in metadata.iloc[indices].itertuples(index=False):
        target = cache / row.image_id
        if target.exists():
            continue
        with Image.open(image_path(row.image_path)) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")
            w, h = image.size
            scale = 224 / min(w, h)
            image = image.resize((round(w * scale), round(h * scale)), Image.Resampling.LANCZOS)
            left, top = (image.width - 224) // 2, (image.height - 224) // 2
            image.crop((left, top, left + 224, top + 224)).save(target, format="JPEG", quality=75)
    model = FrozenModel(model_name, device="cuda")
    batch_size = 16
    vectors = []
    for start in range(0, len(image_ids), batch_size):
        images = []
        for image_id in image_ids[start:start + batch_size]:
            with Image.open(cache / image_id) as image:
                images.append(model.transform(image.convert("RGB")))
        vectors.append(model.image_features(torch.stack(images)).numpy())
        if start % 320 == 0:
            print(f"Recompressed {style} {start}/{len(image_ids)}", flush=True)
    result = np.concatenate(vectors)
    np.save(output, result)
    pd.DataFrame({"image_id": image_ids}).to_csv(ids_file, index=False)
    return result


def run_authenticity(metadata):
    properties = pd.read_parquet(PROCESSED / "image_properties.parquet")
    groups = authenticity_groups(metadata)
    comparison = json.loads((RESULTS / "rq1_comparison.json").read_text(encoding="utf-8"))
    best_model = comparison["best_model_selected_by_wc6_val"]
    rows, fold_rows, oof_rows, controls = [], [], [], []
    for style, definitions in groups.items():
        print(f"RQ3a {style}", flush=True)
        for rule in ("mixed", "strict_year"):
            original, revival = definitions[rule]
            indices = np.flatnonzero(original | revival)
            y = revival[indices].astype(int)
            church_ids = metadata.church_id.iloc[indices].to_numpy()
            image_ids = metadata.image_id.iloc[indices].to_numpy()
            countries = metadata.country.iloc[indices].to_numpy()
            for model in MODELS:
                x = model_features(model, metadata)[indices]
                folds, predictions = cross_validate_authenticity(x, y, church_ids, image_ids, countries, style, rule, model)
                rows.append(summarize_authenticity(folds, predictions, y, style, rule, model))
                fold_rows.extend({"style": style, "rule": rule, "model": model, **fold} for fold in folds)
                oof_rows.extend(predictions)
                print(f"RQ3a {style} {rule} {model} complete", flush=True)
            if rule == "mixed":
                controls.append(property_control(properties, metadata, indices, y, church_ids, style))
                recompressed = recompressed_features(metadata, indices, style, best_model)
                folds, predictions = cross_validate_authenticity(recompressed, y, church_ids, image_ids, countries, style, "recompressed_q75", best_model)
                controls.append({"style": style, "control": "recompressed_q75", "auroc_mean": np.mean([f["auroc"] for f in folds]),
                                 "auroc_sd": np.std([f["auroc"] for f in folds], ddof=1), "images": len(y), "exif_year_present": None})
                oof_rows.extend(predictions)
    oof = pd.DataFrame(oof_rows)
    for style in groups:
        part = oof[(oof["style"] == style) & (oof.rule == "mixed") & (oof.model == best_model)]
        country_scores = []
        for country, group in part.groupby("country"):
            if len(group) >= 20 and group.true.nunique() == 2:
                country_scores.append({"country": country, "images": len(group), "auroc": roc_auc_score(group.true, group.score)})
        weighted = np.average([item["auroc"] for item in country_scores], weights=[item["images"] for item in country_scores]) if country_scores else np.nan
        controls.append({"style": style, "control": "country_weighted_auroc", "auroc_mean": weighted, "auroc_sd": None,
                         "images": sum(item["images"] for item in country_scores), "exif_year_present": None,
                         "eligible_countries": len(country_scores)})
        pd.DataFrame(country_scores).to_csv(RESULTS / f"rq3a_countries_{style.lower()}.csv", index=False)
    pd.DataFrame(rows).to_csv(RESULTS / "rq3a_authenticity.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(RESULTS / "rq3a_folds.csv", index=False)
    oof.to_csv(RESULTS / "rq3a_oof.csv", index=False)
    pd.DataFrame(controls).to_csv(RESULTS / "rq3a_confound_controls.csv", index=False)
    conflict_report = {style: int(definitions["conflict"].sum()) for style, definitions in groups.items()}
    (RESULTS / "rq3a_conflicts.json").write_text(json.dumps(conflict_report, indent=2), encoding="utf-8")
    return best_model


def dating_conflicts(metadata):
    """Conservative, data-derived early-year sensitivity filter."""
    names = dict(line.split(" ", 1) for line in (RAW / "style_names.txt").read_text(encoding="utf-8").splitlines())
    revival_label = metadata.all_styles.map(lambda raw: any("revival" in names.get(style, "").lower() for style in json.loads(raw)))
    year = metadata.construction_year
    reasons = [[] for _ in range(len(metadata))]
    for i in np.flatnonzero(revival_label & metadata.all_styles.map(lambda raw: "Q186363" in json.loads(raw)) & year.le(1550)):
        reasons[i].append("Gothic Revival label with year <=1550")
    for i in np.flatnonzero(revival_label & metadata.all_styles.map(lambda raw: "Q744373" in json.loads(raw)) & year.le(1250)):
        reasons[i].append("Romanesque Revival label with year <=1250")
    thresholds = {}
    valid = metadata[metadata.wc14_style_id.notna() & year.between(800, 2020) & ~revival_label]
    for style_id, group in valid.groupby("wc14_style_id"):
        church_years = group.drop_duplicates("church_id").construction_year.to_numpy()
        if len(church_years) < 10:
            continue
        q10 = float(np.quantile(church_years, 0.10))
        threshold = max(800, q10 - 200)
        thresholds[style_id] = {"nonrevival_churches": len(church_years), "tenth_percentile": q10, "early_cutoff": threshold}
        for i in np.flatnonzero(metadata.wc14_style_id.eq(style_id) & year.lt(threshold)):
            reasons[i].append(f"year earlier than style empirical tenth percentile minus 200 years ({threshold:g})")
    excluded = []
    for i, item in enumerate(reasons):
        if item:
            excluded.append({"image_id": metadata.image_id.iloc[i], "church_id": metadata.church_id.iloc[i],
                             "style_id": metadata.wc14_style_id.iloc[i], "construction_year": year.iloc[i],
                             "reason": "; ".join(item)})
    pd.DataFrame(excluded, columns=["image_id", "church_id", "style_id", "construction_year", "reason"]).to_csv(RESULTS / "rq3b_sensitivity_exclusions.csv", index=False)
    (RESULTS / "rq3b_early_year_thresholds.json").write_text(json.dumps(thresholds, indent=2), encoding="utf-8")
    return np.array([not item for item in reasons], dtype=bool)


def choose_ridge(x, y, styles, churches):
    splitter = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=SEED)
    best = None
    for alpha in [0.1, 1, 10, 100]:
        errors = []
        for train, val in splitter.split(x, styles, churches):
            estimator = Ridge(alpha=alpha, solver="lsqr")
            estimator.fit(x[train], y[train])
            errors.append(mean_absolute_error(y[val], estimator.predict(x[val])))
        score = np.mean(errors)
        if best is None or score < best[0]:
            best = (score, alpha)
    return best[1]


def dating_splits(styles, churches, n=5):
    return list(StratifiedGroupKFold(n_splits=n, shuffle=True, random_state=SEED).split(np.zeros(len(styles)), styles, churches))


def dating_oof(x, years, styles, churches, splits):
    predictions = np.empty(len(years), dtype=float)
    selected_alpha = []
    for train, test in splits:
        alpha = choose_ridge(x[train], years[train], styles[train], churches[train])
        estimator = Ridge(alpha=alpha, solver="lsqr")
        estimator.fit(x[train], years[train])
        predictions[test] = estimator.predict(x[test])
        selected_alpha.append(alpha)
    return predictions, selected_alpha


def cluster_metric_ci(years, predictions, churches, iterations=1000):
    rng = np.random.default_rng(SEED)
    unique = np.unique(churches)
    indices = {church: np.flatnonzero(churches == church) for church in unique}
    metrics = []
    for _ in range(iterations):
        sampled = np.concatenate([indices[church] for church in rng.choice(unique, size=len(unique), replace=True)])
        metrics.append((mean_absolute_error(years[sampled], predictions[sampled]),
                        median_absolute_error(years[sampled], predictions[sampled]),
                        float(spearmanr(years[sampled], predictions[sampled]).statistic)))
    values = np.array(metrics)
    return [np.nanquantile(values[:, i], [0.025, 0.975]).tolist() for i in range(3)]


def partial_spearman(years, predictions, styles):
    # Spearman partial correlation: residualize both rank vectors on style dummies.
    design = pd.get_dummies(styles, dtype=float).to_numpy()
    rank_y = rankdata(years)
    rank_pred = rankdata(predictions)
    residual_y = rank_y - design @ np.linalg.lstsq(design, rank_y, rcond=None)[0]
    residual_pred = rank_pred - design @ np.linalg.lstsq(design, rank_pred, rcond=None)[0]
    statistic, p = pearsonr(residual_y, residual_pred)
    return float(statistic), float(p)


def partial_cluster_ci(years, predictions, styles, churches, iterations=1000):
    rng = np.random.default_rng(SEED)
    unique = np.unique(churches)
    indices = {church: np.flatnonzero(churches == church) for church in unique}
    samples = []
    for _ in range(iterations):
        selected = np.concatenate([indices[church] for church in rng.choice(unique, size=len(unique), replace=True)])
        samples.append(partial_spearman(years[selected], predictions[selected], styles[selected])[0])
    return np.nanquantile(samples, [0.025, 0.975]).tolist()


def run_dating(metadata, best_model):
    base = metadata.wc14_style_id.notna() & metadata.construction_year.between(800, 2020)
    clean = dating_conflicts(metadata)
    rows, prediction_rows, partial_rows = [], [], []
    for dataset_name, mask in [("main", base.to_numpy()), ("conflict_excluded", base.to_numpy() & clean)]:
        indices = np.flatnonzero(mask)
        subset = metadata.iloc[indices]
        years = subset.construction_year.to_numpy(dtype=float)
        styles = subset.wc14_style_id.to_numpy()
        churches = subset.church_id.to_numpy()
        split = dating_splits(styles, churches)
        all_styles = sorted(metadata.wc14_style_id.dropna().unique())
        one_hot = np.column_stack([(styles == style).astype(float) for style in all_styles])
        baseline0 = np.empty(len(subset), dtype=float)
        baseline1 = np.empty(len(subset), dtype=float)
        for train, test in split:
            global_median = float(np.median(years[train]))
            medians = {style: float(np.median(years[train][styles[train] == style])) for style in np.unique(styles[train])}
            baseline0[test] = global_median
            baseline1[test] = [medians.get(style, global_median) for style in styles[test]]
        methods = [("baseline", "B0_global_median", baseline0, []), ("baseline", "B1_true_style_median", baseline1, [])]
        for model in MODELS:
            image_features = model_features(model, metadata)[indices]
            for method, x in [("image_only", image_features), ("image_plus_style", np.hstack([image_features, one_hot]))]:
                prediction, alphas = dating_oof(x, years, styles, churches, split)
                methods.append((model, method, prediction, alphas))
                print(f"RQ3b {dataset_name} {model} {method} complete", flush=True)
        for model, method, prediction, alphas in methods:
            mae_ci, medae_ci, rho_ci = cluster_metric_ci(years, prediction, churches)
            rho = float(spearmanr(years, prediction).statistic)
            rows.append({"dataset": dataset_name, "model": model, "method": method, "images": len(subset), "churches": subset.church_id.nunique(),
                         "mae": mean_absolute_error(years, prediction), "mae_ci_low": mae_ci[0], "mae_ci_high": mae_ci[1],
                         "median_ae": median_absolute_error(years, prediction), "median_ae_ci_low": medae_ci[0], "median_ae_ci_high": medae_ci[1],
                         "spearman_rho": rho, "rho_ci_low": rho_ci[0], "rho_ci_high": rho_ci[1], "selected_alpha_by_fold": json.dumps(alphas)})
            prediction_rows.extend({"dataset": dataset_name, "model": model, "method": method, "image_id": subset.image_id.iloc[i],
                                    "church_id": churches[i], "style_id": styles[i], "true_year": years[i], "predicted_year": prediction[i]}
                                   for i in range(len(subset)))
            if model == best_model and method == "image_only":
                statistic, p = partial_spearman(years, prediction, styles)
                low, high = partial_cluster_ci(years, prediction, styles, churches)
                partial_rows.append({"dataset": dataset_name, "model": model, "partial_spearman_rho": statistic,
                                     "approximate_p": p, "cluster_ci_low": low, "cluster_ci_high": high, "images": len(subset)})
    pd.DataFrame(rows).to_csv(RESULTS / "rq3b_dating.csv", index=False)
    pd.DataFrame(prediction_rows).to_csv(RESULTS / "rq3b_oof.csv", index=False)
    pd.DataFrame(partial_rows).to_csv(RESULTS / "rq3b_partial_spearman.csv", index=False)


def main():
    start = time.time()
    metadata = pd.read_parquet(PROCESSED / "metadata.parquet")
    best_model = run_authenticity(metadata)
    print(f"RQ3a elapsed {time.time() - start:.1f}s", flush=True)
    run_dating(metadata, best_model)
    print(f"RQ3 total elapsed {time.time() - start:.1f}s", flush=True)


if __name__ == "__main__":
    main()
