"""B5 four-condition RQ4 readout for validation-selected WC4 ChronoArch-F."""
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd
import torch
from scipy.stats import wilcoxon
from chronoarch_tasks import task_folds
from chronoarch_train import TrainConfig, make_model
from pathing import PROCESSED, RAW
from run_chronoarch_v2 import RESULTS, stored_row


class RQ4Store:
    def __init__(self):
        root = ROOT / "features_v2"
        self.values = {}
        for size in (224, 448):
            for view in ("base", "flip"):
                stem = f"DINO-L_{size}_{view}_rq4"
                for suffix in ("tokens", "cls"):
                    self.values[f"{stem}_{suffix}"] = np.load(
                        root / f"{stem}_{suffix}.npy", mmap_mode="r")
        for name in ("CLIP-L", "SIG2-B"):
            for view in ("base", "flip"):
                self.values[f"{name}_rq4_{view}"] = np.load(
                    root / f"{name}_rq4_{view}.npy", mmap_mode="r")

    def batch(self, indices, view):
        names = (f"DINO-L_224_{view}_rq4_tokens", f"DINO-L_448_{view}_rq4_tokens",
                 f"DINO-L_224_{view}_rq4_cls", f"DINO-L_448_{view}_rq4_cls",
                 f"CLIP-L_rq4_{view}", f"SIG2-B_rq4_{view}")
        values = []
        for name in names:
            array = np.array(self.values[name][indices], dtype=np.float32, copy=True)
            if name.endswith("_tokens"):
                channels = array.shape[-1]
                array = array.reshape(-1, 8, 2, 8, 2, channels).mean(axis=(2, 4))
                array = array.reshape(-1, 64, channels)
            values.append(torch.from_numpy(array).to("cuda"))
        return tuple(values)


def condition_predictions(fold, config, checkpoint, store, length, batch_size=64):
    model = make_model(fold, config)
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(saved["state_dict"])
    model.eval()
    result = []
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        for start in range(0, length, batch_size):
            positions = np.arange(start, min(start + batch_size, length))
            base = model(*store.batch(positions, "base"))["style_logits"].softmax(-1)
            flip = model(*store.batch(positions, "flip"))["style_logits"].softmax(-1)
            result.append(((base + flip) / 2).float().cpu().numpy())
    return np.concatenate(result)


def church_ci(frame, column, *, seed=0, draws=1000):
    groups = list(frame.groupby("church_id").indices.values())
    rng = np.random.default_rng(seed)
    estimates = np.empty(draws)
    for draw in range(draws):
        selected = np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])
        estimates[draw] = frame.iloc[selected][column].mean()
    return np.quantile(estimates, [0.025, 0.975])


def main():
    assert torch.cuda.is_available()
    metadata = pd.read_parquet(PROCESSED / "metadata.parquet")
    fold = task_folds(metadata, "wc4_rq4")[0]
    selected = json.loads((RESULTS / "selected_wc4_rq4_0.json").read_text(encoding="utf-8"))
    config = TrainConfig(**selected["final"])
    index = pd.read_csv(ROOT / "features_v2" / "rq4_index.csv")
    assert len(index) == 139 * 42
    store = RQ4Store()
    for seed in range(3):
        target = RESULTS / f"rq4_chronoarch_seed{seed}.csv"
        if target.exists():
            continue
        marker = target.with_suffix(".test_started")
        assert not marker.exists(), "Condition test evaluation was already attempted"
        record = stored_row("ChronoArch-F", fold, config, seed)
        assert record is not None
        marker.write_text("condition evaluation started\n", encoding="utf-8")
        started = time.monotonic()
        probabilities = condition_predictions(fold, config, Path(record["checkpoint"]),
                                              store, len(index))
        output = index[["image_id", "condition", "draw"]].copy()
        for class_index in range(probabilities.shape[1]):
            output[f"prob_{class_index}"] = probabilities[:, class_index]
        output["inference_seconds"] = time.monotonic() - started
        temporary = target.with_suffix(".tmp")
        output.to_csv(temporary, index=False)
        temporary.replace(target)
        print(f"Completed RQ4 ChronoArch-F seed {seed}", flush=True)
    frames = [pd.read_csv(RESULTS / f"rq4_chronoarch_seed{seed}.csv") for seed in range(3)]
    assert all(frame[["image_id", "condition", "draw"]].equals(
               frames[0][["image_id", "condition", "draw"]]) for frame in frames)
    columns = [f"prob_{i}" for i in range(4)]
    ensemble = frames[0][["image_id", "condition", "draw"]].copy()
    ensemble[columns] = np.mean([frame[columns].to_numpy() for frame in frames], axis=0)
    originals = [pd.read_csv(RESULTS / f"pred_ChronoArch-F_wc4_rq4_0_{seed}.csv")
                 for seed in range(3)]
    assert all(frame.image_id.tolist() == originals[0].image_id.tolist()
               for frame in originals)
    original = pd.DataFrame({"image_id": originals[0].image_id,
        **{column: np.mean([frame[column].to_numpy() for frame in originals], axis=0)
           for column in columns}})
    original = original.set_index("image_id")
    annotations = json.loads((RAW / "building_parts.json").read_text(encoding="utf-8"))["annotations"]
    qid_to_index = {qid: i for i, qid in enumerate(fold.style_names)}
    assert set(ensemble.image_id) == set(original.index) == set(annotations)
    lookup = {image_id: i for i, image_id in enumerate(metadata.image_id)}
    per_draw = []
    for row in ensemble.itertuples(index=False):
        true_indices = [qid_to_index[qid] for qid in annotations[row.image_id]["styles"]]
        probability = np.array([getattr(row, name) for name in columns])
        original_probability = original.loc[row.image_id, columns].to_numpy(dtype=float)
        per_draw.append({"image_id": row.image_id,
                         "church_id": metadata.church_id.iloc[lookup[row.image_id]],
                         "condition": row.condition, "draw": row.draw,
                         "true_probability": probability[true_indices].sum(),
                         "original_true_probability": original_probability[true_indices].sum(),
                         "top1_correct": int(probability.argmax() in true_indices),
                         "original_top1_correct": int(original_probability.argmax() in true_indices)})
    draws = pd.DataFrame(per_draw)
    draws.to_csv(RESULTS / "rq4_chronoarch_draws.csv", index=False)
    records = []
    for image_id, part in draws.groupby("image_id", sort=False):
        take = lambda condition: part.loc[part.condition.eq(condition)]
        e, k, r, rk = (take(name) for name in ("E", "K", "R", "R_keep"))
        assert (len(e), len(k), len(r), len(rk)) == (1, 1, 20, 20)
        records.append({"image_id": image_id, "church_id": part.church_id.iloc[0],
            "p_original": part.original_true_probability.iloc[0],
            "p_E": e.true_probability.iloc[0], "p_R_mean": r.true_probability.mean(),
            "p_K": k.true_probability.iloc[0], "p_R_keep_mean": rk.true_probability.mean(),
            "correct_original": part.original_top1_correct.iloc[0],
            "correct_E": e.top1_correct.iloc[0], "correct_R_rate": r.top1_correct.mean(),
            "correct_K": k.top1_correct.iloc[0],
            "correct_R_keep_rate": rk.top1_correct.mean()})
    frame = pd.DataFrame(records)
    frame["delta_E"] = frame.p_original - frame.p_E
    frame["delta_R"] = frame.p_original - frame.p_R_mean
    frame["E_minus_R"] = frame.delta_E - frame.delta_R
    frame["K_minus_R_keep"] = frame.p_K - frame.p_R_keep_mean
    frame.to_csv(RESULTS / "rq4_chronoarch.csv", index=False)
    tests = []
    for metric, subset, left, right in (
            ("E_vs_R_probability_drop", frame, "delta_E", "delta_R"),
            ("K_vs_R_keep_probability", frame, "p_K", "p_R_keep_mean"),
            ("K_vs_R_keep_top1_retention", frame.loc[frame.correct_original.eq(1)],
             "correct_K", "correct_R_keep_rate")):
        difference = subset[left] - subset[right]
        ci = church_ci(subset.assign(difference=difference), "difference")
        result = wilcoxon(difference, method="approx")
        tests.append({"metric": metric, "images": len(subset), "left_mean": subset[left].mean(),
                      "right_mean": subset[right].mean(), "mean_difference": difference.mean(),
                      "ci_low": ci[0], "ci_high": ci[1],
                      "wilcoxon_statistic": result.statistic, "wilcoxon_p": result.pvalue})
    pd.DataFrame(tests).to_csv(RESULTS / "rq4_chronoarch_tests.csv", index=False)
    print(pd.DataFrame(tests).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
