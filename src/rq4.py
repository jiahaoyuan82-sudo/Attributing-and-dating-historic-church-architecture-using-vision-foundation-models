"""S6 RQ4 expert-region, random-region and keep-only occlusion tests."""
import json
import time
from collections import defaultdict
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw, ImageFilter
from scipy.stats import wilcoxon, rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from models import FrozenModel, ROOT
from pathing import RAW, PROCESSED
from rq1 import model_features

RESULTS = ROOT / "results"
SEED = 0


def boxes_for(annotation):
    return [(element["left"], element["top"], element["width"], element["height"])
            for group in annotation["bbox_groups"] for element in group["elements"]]


def pixel_box(box, width, height):
    left, top, w, h = box
    return (round(left * width), round(top * height), round((left + w) * width), round((top + h) * height))


def occlude(image, boxes, keep_only=False):
    average = tuple(np.asarray(image).reshape(-1, 3).mean(axis=0).astype(np.uint8))
    mask = Image.new("L", image.size, color=0)
    draw = ImageDraw.Draw(mask)
    for box in boxes:
        draw.rectangle(pixel_box(box, *image.size), fill=255)
    mask = mask.filter(ImageFilter.GaussianBlur(radius=3))
    filled = Image.new("RGB", image.size, color=average)
    return Image.composite(image, filled, mask) if keep_only else Image.composite(filled, image, mask)


def iou(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    intersection = max(0, min(ax + aw, bx + bw) - max(ax, bx)) * max(0, min(ay + ah, by + bh) - max(ay, by))
    union = aw * ah + bw * bh - intersection
    return intersection / union if union else 0


def random_boxes(expert, rng):
    result = []
    for _, _, width, height in expert:
        xmin, xmax = max(0, 0.1 - width / 2), min(1 - width, 0.9 - width / 2)
        ymin, ymax = max(0, 0.1 - height / 2), min(1 - height, 0.9 - height / 2)
        if xmin > xmax or ymin > ymax:
            raise RuntimeError("Expert box too large for center-80% random placement")
        for _ in range(5000):
            box = (float(rng.uniform(xmin, xmax)), float(rng.uniform(ymin, ymax)), width, height)
            if all(iou(box, other) <= 0.1 for other in expert):
                result.append(box)
                break
        else:
            raise RuntimeError("Could not place random box at <=10% IoU with expert boxes")
    return result


def tune_wc4_classifier(features, metadata, annotation_ids):
    annotated_churches = set(metadata.loc[metadata.image_id.isin(annotation_ids), "church_id"])
    nonexpert = ~metadata.church_id.isin(annotated_churches)
    train = metadata.split_wc4.eq("train") & nonexpert
    val = metadata.split_wc4.eq("val") & nonexpert
    labels = metadata.wc4_style_id
    best = None
    for c in [0.01, 0.1, 1, 10]:
        estimator = LogisticRegression(C=c, class_weight="balanced", max_iter=3000, random_state=SEED)
        estimator.fit(features[train.to_numpy()], labels[train])
        score = balanced_accuracy_score(labels[val], estimator.predict(features[val.to_numpy()]))
        if best is None or score > best[0]:
            best = (score, c)
    classifier = LogisticRegression(C=best[1], class_weight="balanced", max_iter=3000, random_state=SEED)
    classifier.fit(features[train.to_numpy()], labels[train])
    counts = {"train_images_removed": int((metadata.split_wc4.eq("train") & ~nonexpert).sum()),
              "validation_images_removed": int((metadata.split_wc4.eq("val") & ~nonexpert).sum()),
              "train_images_retained": int(train.sum()), "val_images_retained": int(val.sum()),
              "selected_C": best[1], "val_bal_acc": best[0]}
    return classifier, counts


def image_conditions(image, expert_boxes, rng):
    variants = [("E", 0, occlude(image, expert_boxes)), ("K", 0, occlude(image, expert_boxes, keep_only=True))]
    for draw in range(20):
        boxes = random_boxes(expert_boxes, rng)
        variants.append(("R", draw, occlude(image, boxes)))
    return variants


def group_results(frame, annotations):
    rows = []
    for model, part in frame.groupby("model"):
        for style, subset in part.explode("annotation_styles").groupby("annotation_styles"):
            rows.append({"model": model, "group_type": "style", "group": style, "images": len(subset),
                         "mean_delta_E": subset.delta_E.mean(), "mean_delta_R": subset.delta_R.mean(),
                         "mean_delta_difference": (subset.delta_E - subset.delta_R).mean()})
        image_labels = {}
        meta_names = annotations["meta"]
        for image_id, annotation in annotations["annotations"].items():
            image_labels[image_id] = {meta_names[element["label"]]["name"] for group in annotation["bbox_groups"] for element in group["elements"]}
        by_label = defaultdict(list)
        for row in part.itertuples(index=False):
            for label in image_labels[row.image_id]:
                by_label[label].append(row)
        for label, items in by_label.items():
            if len(items) >= 10:
                rows.append({"model": model, "group_type": "part_label", "group": label, "images": len(items),
                             "mean_delta_E": np.mean([item.delta_E for item in items]),
                             "mean_delta_R": np.mean([item.delta_R for item in items]),
                             "mean_delta_difference": np.mean([item.delta_E - item.delta_R for item in items])})
    return pd.DataFrame(rows)


def main():
    start = time.time()
    metadata = pd.read_parquet(PROCESSED / "metadata.parquet")
    annotations = json.loads((RAW / "building_parts.json").read_text(encoding="utf-8"))
    annotation_ids = list(annotations["annotations"])
    index = {image_id: i for i, image_id in enumerate(metadata.image_id)}
    assert len(annotation_ids) == 139 and all(image_id in index for image_id in annotation_ids)
    best = json.loads((RESULTS / "rq1_comparison.json").read_text(encoding="utf-8"))["best_model_selected_by_wc6_val"]
    models = list(dict.fromkeys([best, "R50"]))
    rows, random_rows, test_rows, classifier_info = [], [], [], {}
    for model_name in models:
        frozen = FrozenModel(model_name, device="cuda")
        features = model_features(model_name, metadata)
        classifier, classifier_info[model_name] = tune_wc4_classifier(features, metadata, annotation_ids)
        class_to_index = {name: i for i, name in enumerate(classifier.classes_)}
        rng = np.random.default_rng(SEED)
        for sequence, image_id in enumerate(annotation_ids, 1):
            annotation = annotations["annotations"][image_id]
            true_styles = annotation["styles"]
            assert all(style in class_to_index for style in true_styles)
            original_probability = classifier.predict_proba(features[index[image_id]][None, :])[0]
            true_indices = [class_to_index[style] for style in true_styles]
            p_original = float(original_probability[true_indices].sum())
            original_top = classifier.classes_[original_probability.argmax()]
            with Image.open(PROCESSED / "img256" / image_id) as source:
                original_image = source.convert("RGB")
            expert = boxes_for(annotation)
            variants = image_conditions(original_image, expert, rng)
            tensors = torch.stack([frozen.transform(image) for _, _, image in variants])
            new_features = frozen.image_features(tensors).numpy()
            probabilities = classifier.predict_proba(new_features)
            condition = {}
            for (name, draw, _), probability in zip(variants, probabilities):
                value = float(probability[true_indices].sum())
                flip = int(classifier.classes_[probability.argmax()] != original_top)
                condition.setdefault(name, []).append((value, flip))
                if name == "R":
                    random_rows.append({"model": model_name, "image_id": image_id, "draw": draw, "p_original": p_original,
                                        "p_random": value, "delta_p": p_original - value, "top1_flip": flip})
            p_e, flip_e = condition["E"][0]
            p_k, flip_k = condition["K"][0]
            p_r = np.mean([value for value, _ in condition["R"]])
            rows.append({"model": model_name, "image_id": image_id, "church_id": metadata.church_id.iloc[index[image_id]],
                         "annotation_styles": true_styles, "expert_boxes": len(expert), "p_original": p_original,
                         "p_E": p_e, "p_R_mean": p_r, "p_K": p_k, "delta_E": p_original - p_e,
                         "delta_R": p_original - p_r, "delta_K": p_original - p_k,
                         "flip_E": flip_e, "flip_R_rate": np.mean([flip for _, flip in condition["R"]]), "flip_K": flip_k})
            if sequence % 20 == 0 or sequence == len(annotation_ids):
                print(f"RQ4 {model_name}: {sequence}/{len(annotation_ids)}", flush=True)
        part = pd.DataFrame([row for row in rows if row["model"] == model_name])
        difference = part.delta_E - part.delta_R
        statistic = wilcoxon(difference, alternative="two-sided", method="approx")
        nonzero = difference.to_numpy()[difference.to_numpy() != 0]
        ranks = rankdata(np.abs(nonzero))
        z = (ranks[nonzero > 0].sum() - ranks.sum() / 2) / np.sqrt(np.square(ranks).sum() / 4)
        church_groups = list(part.groupby("church_id").indices.values())
        bootstrap = np.random.default_rng(SEED)
        draws = np.empty((1000, 3))
        for draw in range(len(draws)):
            selected = np.concatenate([church_groups[i] for i in bootstrap.integers(0, len(church_groups), len(church_groups))])
            sampled = part.iloc[selected]
            draws[draw] = (sampled.delta_E.mean(), sampled.delta_R.mean(), (sampled.delta_E - sampled.delta_R).mean())
        ci = np.quantile(draws, [0.025, 0.975], axis=0)
        test_rows.append({"model": model_name, "images": len(part), "mean_delta_E": part.delta_E.mean(),
                          "mean_delta_R": part.delta_R.mean(), "mean_delta_difference": difference.mean(),
                          "delta_E_ci_low": ci[0, 0], "delta_E_ci_high": ci[1, 0],
                          "delta_R_ci_low": ci[0, 1], "delta_R_ci_high": ci[1, 1],
                          "difference_ci_low": ci[0, 2], "difference_ci_high": ci[1, 2],
                          "wilcoxon_statistic": float(statistic.statistic), "wilcoxon_p": float(statistic.pvalue),
                          "effect_r": z / np.sqrt(len(nonzero)), "flip_E_rate": part.flip_E.mean(),
                          "flip_R_rate": part.flip_R_rate.mean(), "flip_K_rate": part.flip_K.mean()})
        del frozen
        torch.cuda.empty_cache()
    frame = pd.DataFrame(rows)
    frame.to_csv(RESULTS / "rq4_occlusion.csv", index=False)
    pd.DataFrame(random_rows).to_csv(RESULTS / "rq4_random_runs.csv", index=False)
    group_results(frame, annotations).to_csv(RESULTS / "rq4_by_group.csv", index=False)
    pd.DataFrame(test_rows).to_csv(RESULTS / "rq4_tests.csv", index=False)
    (RESULTS / "rq4_classifier_info.json").write_text(json.dumps(classifier_info, indent=2), encoding="utf-8")
    print(f"RQ4 elapsed {time.time() - start:.1f}s", flush=True)


if __name__ == "__main__":
    main()
