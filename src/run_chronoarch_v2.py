"""Resumable validation-only staged search and final single test readout."""
import argparse
import csv
import hashlib
import json
import sys
import shutil
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from chronoarch_tasks import task_folds
from chronoarch_train import (FeatureStore, LoRAConfig, RESULTS, TrainConfig,
                              evaluate_checkpoint, evaluate_lora_checkpoint,
                              train_lora, train_one)
from pathing import PROCESSED

CHECKPOINTS = RESULTS / "checkpoints"
SELECTION = RESULTS / "selection_log.csv"
TASKS = ("wc6", "wc14", "gothic", "romanesque", "dating", "wc4_rq4")
FIELDS = ("model", "task", "fold", "stage", "config_id", "seed", "queries",
          "lr", "lambda_fine", "lambda_year", "tau", "val_metric", "best_epoch",
          "epochs_run", "train_seconds", "trainable_parameters", "peak_allocated_bytes",
          "peak_reserved_bytes", "checkpoint")


def config_id(config):
    return hashlib.sha256(json.dumps(asdict(config), sort_keys=True).encode()).hexdigest()[:12]


def rows():
    return pd.read_csv(SELECTION).to_dict("records") if SELECTION.exists() else []


def append_row(row):
    SELECTION.parent.mkdir(parents=True, exist_ok=True)
    with SELECTION.open("a", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        if stream.tell() == 0:
            writer.writeheader()
        writer.writerow({key: row.get(key, "") for key in FIELDS})
        stream.flush()


def stored_row(model, fold, config, seed):
    for row in rows():
        if (row["model"] == model and row["task"] == fold.task
                and int(row["fold"]) == fold.fold and row["config_id"] == config_id(config)
                and int(row["seed"]) == seed and Path(row["checkpoint"]).exists()):
            return row
    return None


def run_config(fold, store, config, seed, stage):
    existing = stored_row("ChronoArch-F", fold, config, seed)
    if existing:
        return existing
    checkpoint = CHECKPOINTS / f"f_{fold.task}_{fold.fold}_{config_id(config)}_{seed}.pt"
    result = train_one(fold, store, config, seed, checkpoint,
                       use_augmentation=config.variant != "R3")
    row = {"model": "ChronoArch-F", "stage": stage,
           "config_id": config_id(config), **result}
    append_row(row)
    return row


def best_config(fold, store, configs, stage):
    scored = []
    for config in configs:
        records = [run_config(fold, store, config, 0, stage)]
        scored.append((float(np.mean([float(row["val_metric"]) for row in records])), config))
    return max(scored, key=lambda pair: (pair[0], -pair[1].queries, -pair[1].lr))[1]


def staged_config(fold, store):
    if fold.task in ("gothic", "romanesque", "dating") and fold.fold > 0:
        reference = RESULTS / f"selected_{fold.task}_0.json"
        assert reference.exists(), "First fold validation selection must finish first"
        chosen = TrainConfig(**json.loads(reference.read_text(encoding="utf-8"))["final"])
        (RESULTS / f"selected_{fold.task}_{fold.fold}.json").write_text(
            json.dumps({"final": asdict(chosen), "selection_metric": "fold_0_validation_only"}, indent=2) + "\n",
            encoding="utf-8")
        return chosen
    base = TrainConfig()
    first = [replace(base, queries=k, lr=lr)
             for k in (4, 8) for lr in (0.0005, 0.001)]
    chosen_first = best_config(fold, store, first, "K_lr")
    second = [replace(chosen_first, lambda_fine=f, lambda_year=y, tau=t)
              for f in (0.3, 1.0) for y in (0.3, 1.0) for t in (0, 1)]
    chosen = best_config(fold, store, second, "loss_tau")
    target = RESULTS / f"selected_{fold.task}_{fold.fold}.json"
    target.write_text(json.dumps({"stage1": asdict(chosen_first),
                                  "final": asdict(chosen),
                                  "selection_metric": "validation_only"}, indent=2) + "\n",
                      encoding="utf-8")
    # Remove only unselected v2 checkpoints if disk space is actually tight.
    if shutil.disk_usage(RESULTS).free < 10 * 1024 ** 3:
        keep = {CHECKPOINTS / f"f_{fold.task}_{fold.fold}_{config_id(chosen)}_{seed}.pt"
                for seed in range(3)}
        allowed_root = CHECKPOINTS.resolve()
        for temporary in CHECKPOINTS.glob(f"f_{fold.task}_{fold.fold}_*.pt"):
            assert temporary.resolve().parent == allowed_root
            if temporary not in keep:
                temporary.unlink()
    return chosen


def save_test(fold, model, seed, prediction, seconds):
    target = RESULTS / f"pred_{model}_{fold.task}_{fold.fold}_{seed}.csv"
    assert not target.exists(), "Test output already exists; evaluation must not repeat"
    values = ({"prediction": prediction} if prediction.ndim == 1 else
              {f"prob_{i}": prediction[:, i] for i in range(prediction.shape[1])})
    frame = pd.DataFrame({"image_id": fold.image_ids[fold.test],
                          "church_id": fold.churches[fold.test],
                          "truth": (np.full(len(fold.test), np.nan) if fold.task == "wc4_rq4"
                                    else fold.year[fold.test] if fold.task == "dating"
                                    else fold.style[fold.test]), **values})
    frame["inference_seconds"] = seconds
    temporary = target.with_suffix(".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(target)


def claim_test(fold, model, seed):
    """Fail closed if a previous attempt may already have read this test fold."""
    target = RESULTS / f"pred_{model}_{fold.task}_{fold.fold}_{seed}.csv"
    marker = target.with_suffix(".test_started")
    assert not target.exists() and not marker.exists(), (
        f"Test already evaluated or attempted: {target}")
    marker.write_text("test evaluation started\n", encoding="utf-8")


def run_fold_f(fold, store):
    selected_file = RESULTS / f"selected_{fold.task}_{fold.fold}.json"
    selected = (TrainConfig(**json.loads(selected_file.read_text(encoding="utf-8"))["final"])
                if selected_file.exists() else staged_config(fold, store))
    for seed in range(3):
        if (RESULTS / f"pred_ChronoArch-F_{fold.task}_{fold.fold}_{seed}.csv").exists():
            continue
        record = stored_row("ChronoArch-F", fold, selected, seed)
        if record is None:
            record = run_config(fold, store, selected, seed, "final_seed")
        claim_test(fold, "ChronoArch-F", seed)
        prediction, seconds = evaluate_checkpoint(fold, store, Path(record["checkpoint"]))
        save_test(fold, "ChronoArch-F", seed, prediction, seconds)
    return selected


def run_fold_l(fold, metadata, selected):
    for seed in range(3):
        if (RESULTS / f"pred_ChronoArch-L_{fold.task}_{fold.fold}_{seed}.csv").exists():
            continue
        record = stored_row("ChronoArch-L", fold, selected, seed)
        if record is None:
            f_record = stored_row("ChronoArch-F", fold, selected, seed)
            assert f_record is not None
            checkpoint = CHECKPOINTS / f"l_{fold.task}_{fold.fold}_{seed}.pt"
            result = train_lora(fold, metadata, selected, LoRAConfig(), seed,
                                checkpoint, Path(f_record["checkpoint"]))
            record = {"model": "ChronoArch-L", "stage": "lora",
                      "config_id": config_id(selected), **result}
            append_row(record)
        claim_test(fold, "ChronoArch-L", seed)
        prediction, seconds = evaluate_lora_checkpoint(fold, metadata,
                                                        Path(record["checkpoint"]))
        save_test(fold, "ChronoArch-L", seed, prediction, seconds)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=TASKS, required=True)
    parser.add_argument("--model", choices=("F", "L", "both"), default="both")
    args = parser.parse_args()
    assert torch.cuda.is_available(), "CUDA is mandatory"
    metadata = pd.read_parquet(PROCESSED / "metadata.parquet")
    store = FeatureStore()
    for fold in task_folds(metadata, args.task):
        selected = (run_fold_f(fold, store) if args.model in ("F", "both") else
                    TrainConfig(**json.loads((RESULTS / f"selected_{fold.task}_{fold.fold}.json")
                                             .read_text(encoding="utf-8"))["final"]))
        if args.model in ("L", "both"):
            run_fold_l(fold, metadata, selected)
        print(f"Completed {args.task} fold {fold.fold} {args.model}", flush=True)


if __name__ == "__main__":
    main()
