"""Summarize observed training, inference, parameter, and GPU-memory costs."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd


def main():
    results = ROOT / "results_v2"
    log = pd.read_csv(results / "selection_log.csv")
    log["reported_model"] = log.model
    ablation = log.stage.astype(str).str.startswith("ablation_")
    log.loc[ablation, "reported_model"] = log.loc[ablation, "stage"].str.removeprefix("ablation_")
    rows = []
    for (task, model), group in log.groupby(["task", "reported_model"], dropna=False):
        cost = pd.to_numeric(group.train_seconds, errors="coerce")
        parameters = pd.to_numeric(group.trainable_parameters, errors="coerce")
        allocated = pd.to_numeric(group.peak_allocated_bytes, errors="coerce")
        reserved = pd.to_numeric(group.peak_reserved_bytes, errors="coerce")
        predictions = list(results.glob(f"pred_{model}_{task}_*.csv"))
        inference = []
        for path in predictions:
            frame = pd.read_csv(path, nrows=1)
            if "inference_seconds" in frame:
                inference.append(float(frame.inference_seconds.iloc[0]))
        rows.append({"task": task, "model": model,
                     "trained_runs": len(group), "final_test_files": len(predictions),
                     "total_training_seconds": cost.sum(),
                     "mean_training_seconds": cost.mean(),
                     "trainable_parameters": parameters.dropna().iloc[0]
                     if parameters.notna().any() else np.nan,
                     "peak_allocated_bytes": allocated.max(),
                     "peak_reserved_bytes": reserved.max(),
                     "total_final_inference_seconds": np.sum(inference)})
    frame = pd.DataFrame(rows)
    frame.to_csv(results / "compute_cost.csv", index=False)
    print(frame.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
