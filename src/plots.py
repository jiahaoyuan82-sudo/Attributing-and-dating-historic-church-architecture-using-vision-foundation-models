"""S7 publication figures built directly from verified result files."""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import roc_curve, auc

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rq4 import boxes_for, pixel_box
from pathing import ROOT, RAW, PROCESSED

RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
FIGURES.mkdir(exist_ok=True)
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "savefig.bbox": "tight"})


def names():
    return dict(line.split(" ", 1) for line in (RAW / "style_names.txt").read_text(encoding="utf-8").splitlines())


def save(fig, stem):
    for extension in ("png", "pdf"):
        fig.savefig(FIGURES / f"{stem}.{extension}", dpi=300)
    plt.close(fig)


def dataset_figure():
    metadata = pd.read_parquet(PROCESSED / "metadata.parquet")
    stats = pd.read_csv(RESULTS / "dataset_stats.csv")
    classes = stats[(stats.section == "class") & (stats.subset == "wc14")].sort_values("images", ascending=True)
    annotation = json.loads((RAW / "building_parts.json").read_text(encoding="utf-8"))["annotations"]
    fig = plt.figure(figsize=(13, 9))
    grid = fig.add_gridspec(2, 3, width_ratios=[1.45, 1, 1])
    ax = fig.add_subplot(grid[:, 0])
    ax.barh(classes["style"], classes.images, color="#4f6f9f")
    ax.set_xlabel("Images")
    ax.set_title("A  WikiChurches-14 class distribution")
    ax = fig.add_subplot(grid[0, 1])
    ax.hist(metadata.construction_year.dropna(), bins=30, color="#b67653")
    ax.set_xlabel("Construction year")
    ax.set_ylabel("Images")
    ax.set_title("B  Recorded construction years")
    ax = fig.add_subplot(grid[0, 2])
    counts = stats[(stats.section == "country") & (stats.subset == "all")]
    ax.barh(counts.country.iloc[::-1], counts.images.iloc[::-1], color="#6a9d88")
    ax.set_xlabel("Images")
    ax.set_title("C  Top countries")
    sample_id = next(iter(annotation))
    ax = fig.add_subplot(grid[1, 1])
    with Image.open(PROCESSED / "img256" / sample_id) as source:
        image = source.convert("RGB")
    ax.imshow(image)
    for box in boxes_for(annotation[sample_id]):
        x0, y0, x1, y1 = pixel_box(box, *image.size)
        ax.add_patch(plt.Rectangle((x0, y0), x1-x0, y1-y0, fill=False, edgecolor="red", linewidth=1))
    ax.axis("off")
    ax.set_title("D  Expert feature boxes")
    ax = fig.add_subplot(grid[1, 2])
    ax.axis("off")
    ax.text(0.02, 0.9, "Images → frozen encoders\n→ style attribution\n→ historical error structure\n→ authenticity and dating\n→ causal occlusion", va="top", fontsize=13, linespacing=1.8)
    ax.set_title("E  Analysis workflow")
    fig.tight_layout()
    save(fig, "fig1_dataset")


def rq1_figure():
    data = pd.read_csv(RESULTS / "rq1_main.csv")
    models = ["R50", "DINO-B", "CLIP-L", "SIG2-B"]
    protocols = ["zero_shot", "knn", "linear_probe"]
    colors = {"zero_shot": "#8b77ad", "knn": "#6a9d88", "linear_probe": "#4f6f9f"}
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for ax, task in zip(axes, ("wc6", "wc14")):
        subset = data[data.task == task]
        for j, protocol in enumerate(protocols):
            x, y, low, high = [], [], [], []
            for i, model in enumerate(models):
                row = subset[(subset.model == model) & (subset.protocol == protocol)]
                if row.empty:
                    continue
                record = row.iloc[0]
                x.append(i + (j - 1) * 0.24)
                y.append(record.bal_acc)
                low.append(record.bal_acc - record.ci_low)
                high.append(record.ci_high - record.bal_acc)
            ax.bar(x, y, width=0.22, color=colors[protocol], label=protocol if task == "wc6" else None)
            ax.errorbar(x, y, yerr=[low, high], fmt="none", color="black", capsize=2, linewidth=0.7)
        if task == "wc6":
            ax.axhline(0.602, linestyle="--", color="#a75151", linewidth=1, label="Original fine-tuned R50 60.2%")
            ax.axhline(0.695, linestyle=":", color="#a75151", linewidth=1, label="Original fine-tuned TResNet 69.5%")
        ax.set_xticks(range(4), models)
        ax.set_ylim(0, 1)
        ax.set_ylabel("Balanced accuracy")
        ax.set_title(task.upper())
    axes[0].legend(loc="upper left", fontsize=7)
    fig.suptitle("RQ1 frozen-feature attribution (95% church bootstrap CI)")
    fig.tight_layout()
    save(fig, "fig2_rq1")


def rq2_figures():
    matrix = pd.read_csv(RESULTS / "rq2_confusion_normalized.csv", index_col=0)
    label_names = names()
    fig, ax = plt.subplots(figsize=(9, 8))
    display = ax.imshow(matrix.to_numpy(), cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(matrix.columns)), [label_names.get(x, x) for x in matrix.columns], rotation=70, ha="right")
    ax.set_yticks(range(len(matrix.index)), [label_names.get(x, x) for x in matrix.index])
    ax.set_xlabel("Predicted style")
    ax.set_ylabel("True style")
    ax.set_title("RQ2 WC14 confusion, ordered by median construction year")
    fig.colorbar(display, ax=ax, label="Row-normalized proportion")
    fig.tight_layout()
    save(fig, "fig3_confusion_time")
    pairs = pd.read_csv(RESULTS / "rq2_pairs.csv")
    tests = pd.read_csv(RESULTS / "rq2_mantel.csv")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, y, label, test in zip(axes, ("symmetric_confusion", "center_cosine_distance"),
                                  ("Symmetric confusion", "Class-center cosine distance"), tests.itertuples(index=False)):
        ax.scatter(pairs.year_distance, pairs[y], alpha=0.55, s=30)
        ax.set_xlabel("Median construction-year gap")
        ax.set_ylabel(label)
        ax.set_title(f"Spearman r={test.spearman_r:.3f}, permutation p={test.permutation_p:.4g}")
    fig.tight_layout()
    save(fig, "fig3_mantel_scatter")

    examples = pd.read_csv(RESULTS / "rq2_renaissance_to_baroque_images.csv")
    if len(examples):
        columns = 4
        rows = (len(examples) + columns - 1) // columns
        fig, axes = plt.subplots(rows, columns, figsize=(3 * columns, 3 * rows), squeeze=False)
        for ax in axes.flat:
            ax.axis("off")
        for ax, image_id in zip(axes.flat, examples.image_id):
            with Image.open(PROCESSED / "img256" / image_id) as source:
                ax.imshow(source.convert("RGB"))
            ax.set_title(image_id, fontsize=7)
        fig.suptitle("Renaissance images predicted as Baroque (WC14 test)")
        fig.tight_layout()
        save(fig, "fig3_renaissance_to_baroque_examples")


def rq3_figures():
    comparison = json.loads((RESULTS / "rq1_comparison.json").read_text(encoding="utf-8"))
    best = comparison["best_model_selected_by_wc6_val"]
    oof = pd.read_csv(RESULTS / "rq3a_oof.csv")
    fig, axes = plt.subplots(2, 2, figsize=(10, 9))
    for i, style in enumerate(("Gothic", "Romanesque")):
        for j, rule in enumerate(("mixed", "strict_year")):
            ax = axes[i, j]
            for model in dict.fromkeys([best, "R50"]):
                part = oof[(oof["style"] == style) & (oof.rule == rule) & (oof.model == model)]
                fpr, tpr, _ = roc_curve(part.true, part.score)
                ax.plot(fpr, tpr, label=f"{model} AUC={auc(fpr, tpr):.3f}")
            ax.plot([0, 1], [0, 1], color="gray", linestyle="--", linewidth=0.8)
            ax.set_xlabel("False positive rate")
            ax.set_ylabel("True positive rate")
            ax.set_title(f"{style} — {rule}")
            ax.legend()
    fig.tight_layout()
    save(fig, "fig4_roc")
    dating = pd.read_csv(RESULTS / "rq3b_oof.csv")
    summary = pd.read_csv(RESULTS / "rq3b_dating.csv")
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharex=True, sharey=True)
    for ax, dataset in zip(axes, ("main", "conflict_excluded")):
        part = dating[(dating.dataset == dataset) & (dating.model == best) & (dating.method == "image_only")]
        row = summary[(summary.dataset == dataset) & (summary.model == best) & (summary.method == "image_only")].iloc[0]
        ax.scatter(part.true_year, part.predicted_year, s=6, alpha=0.3)
        ax.plot([800, 2020], [800, 2020], color="black", linestyle="--", linewidth=0.7)
        ax.set_title(f"{dataset}: MAE={row.mae:.1f} years")
        ax.set_xlabel("Recorded year")
        ax.set_ylabel("Predicted year")
    fig.tight_layout()
    save(fig, "fig4_dating")


def rq4_figures():
    data = pd.read_csv(RESULTS / "rq4_occlusion.csv")
    models = data.model.unique().tolist()
    fig, axes = plt.subplots(1, len(models), figsize=(5 * len(models), 4), squeeze=False)
    for ax, model in zip(axes[0], models):
        part = data[data.model == model]
        ax.scatter(part.delta_R, part.delta_E, s=18, alpha=0.6)
        limits = [min(part.delta_E.min(), part.delta_R.min()), max(part.delta_E.max(), part.delta_R.max())]
        ax.plot(limits, limits, color="black", linestyle="--", linewidth=0.8)
        ax.set_xlabel("Random-box mean Δp")
        ax.set_ylabel("Expert-box Δp")
        ax.set_title(model)
    fig.tight_layout()
    save(fig, "fig5_occlusion")
    best = json.loads((RESULTS / "rq1_comparison.json").read_text(encoding="utf-8"))["best_model_selected_by_wc6_val"]
    part = data[data.model == best].copy()
    part["difference"] = part.delta_E - part.delta_R
    cases = pd.concat([part.nlargest(4, "difference"), part.nsmallest(4, "difference")]).drop_duplicates("image_id").head(8)
    annotations = json.loads((RAW / "building_parts.json").read_text(encoding="utf-8"))["annotations"]
    fig, axes = plt.subplots(2, 4, figsize=(12, 7))
    for ax, row in zip(axes.flat, cases.itertuples(index=False)):
        with Image.open(PROCESSED / "img256" / row.image_id) as source:
            image = source.convert("RGB")
        ax.imshow(image)
        for box in boxes_for(annotations[row.image_id]):
            x0, y0, x1, y1 = pixel_box(box, *image.size)
            ax.add_patch(plt.Rectangle((x0, y0), x1-x0, y1-y0, fill=False, edgecolor="red", linewidth=0.8))
        ax.set_title(f"ΔpE={row.delta_E:.3f}; ΔpR={row.delta_R:.3f}", fontsize=8)
        ax.axis("off")
    fig.tight_layout()
    save(fig, "fig5_cases")


def main():
    dataset_figure()
    rq1_figure()
    rq2_figures()
    rq3_figures()
    rq4_figures()
    print("Created the eight required figures and any available RQ2 error montage in PNG and PDF", flush=True)


if __name__ == "__main__":
    main()
