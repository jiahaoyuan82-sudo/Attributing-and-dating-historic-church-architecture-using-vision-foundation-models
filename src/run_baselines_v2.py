"""Validation-selected B-2 to B-9 baselines with final one-time test output."""
import argparse
import hashlib
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from baselines_v2 import (evaluate_frozen_probe, evaluate_online_probe,
                          train_frozen_probe, train_online_probe)
from chronoarch_tasks import task_folds, validation_metric
from chronoarch_train import FeatureStore, RESULTS
from pathing import PROCESSED
from run_chronoarch_v2 import CHECKPOINTS, append_row, claim_test, save_test

TASKS = ("wc6", "wc14", "gothic", "romanesque", "dating")
LINEAR = ("B-2_224", "B-2_448", "B-4", "B-6_DINO", "B-6_CLIP", "B-6_SIG")
NEURAL_FROZEN = ("B-3", "B-5", "B-7")
NEURAL_ONLINE = ()


def matrix(store, indices, view, kind):
    a = store.arrays
    semantic = "flip" if view == "flip" else "base"
    keys = {
        "B-2_224": (f"DINO-L_224_{view}_cls",),
        "B-2_448": (f"DINO-L_448_{view}_cls",),
        "B-4": (f"DINO-L_224_{view}_cls", f"DINO-L_448_{view}_cls",
                f"CLIP-L_{semantic}", f"SIG2-B_{semantic}"),
        "B-6_DINO": (f"DINO-L_224_{view}_cls", f"DINO-L_448_{view}_cls"),
        "B-6_CLIP": (f"CLIP-L_{semantic}",),
        "B-6_SIG": (f"SIG2-B_{semantic}",),
    }[kind]
    return np.concatenate([np.asarray(a[key][indices], dtype=np.float32)
                           for key in keys], axis=1)


def linear_fit(fold, store, kind, c, seed):
    x = matrix(store, fold.train, "base", kind)
    y = fold.year[fold.train] if fold.task == "dating" else fold.style[fold.train]
    estimator = (Ridge(alpha=1 / c) if fold.task == "dating" else
                 LogisticRegression(C=c, max_iter=1000, random_state=seed))
    model = make_pipeline(StandardScaler(), estimator)
    started = time.monotonic()
    model.fit(x, y)
    return model, time.monotonic() - started


def linear_predict(model, fold, store, indices, kind):
    base = matrix(store, indices, "base", kind)
    flip = matrix(store, indices, "flip", kind)
    if fold.task == "dating":
        return (model.predict(base) + model.predict(flip)) / 2
    probability = (model.predict_proba(base) + model.predict_proba(flip)) / 2
    return probability[:, 1] if fold.task in ("gothic", "romanesque") else probability


def linear_checkpoint(fold, store, kind, c, seed):
    identifier = hashlib.sha256(f"{kind}:{fold.task}:{fold.fold}:{c}:{seed}".encode()).hexdigest()[:12]
    path = CHECKPOINTS / f"linear_{identifier}.pkl"
    if path.exists():
        with path.open("rb") as stream:
            return pickle.load(stream)
    model, seconds = linear_fit(fold, store, kind, c, seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as stream:
        pickle.dump(model, stream)
    pred = linear_predict(model, fold, store, fold.val, kind)
    truth = fold.year[fold.val] if fold.task == "dating" else fold.style[fold.val]
    metric = validation_metric(fold.task, truth, pred)
    append_row({"model": kind, "task": fold.task, "fold": fold.fold,
                "stage": "linear_C", "config_id": str(c), "seed": seed,
                "val_metric": metric, "train_seconds": seconds,
                "checkpoint": str(path)})
    return model


def linear_select(fold, store, kind):
    selection_file = RESULTS / f"selected_{kind}_{fold.task}_{fold.fold}.json"
    if selection_file.exists():
        return float(json.loads(selection_file.read_text(encoding="utf-8"))["C"])
    if fold.task in ("gothic", "romanesque", "dating") and fold.fold > 0:
        reference = RESULTS / f"selected_{kind}_{fold.task}_0.json"
        assert reference.exists(), "First fold validation selection must finish first"
        chosen = float(json.loads(reference.read_text(encoding="utf-8"))["C"])
        selection_file.write_text(json.dumps({"C": chosen, "source": "fold_0_validation_only"}) + "\n",
                                  encoding="utf-8")
        return chosen
    scored = []
    for c in (0.01, 0.1, 1.0, 10.0):
        metrics = []
        for seed in range(1):
            model = linear_checkpoint(fold, store, kind, c, seed)
            pred = linear_predict(model, fold, store, fold.val, kind)
            truth = fold.year[fold.val] if fold.task == "dating" else fold.style[fold.val]
            metrics.append(validation_metric(fold.task, truth, pred))
        scored.append((float(np.mean(metrics)), c))
    chosen = max(scored, key=lambda pair: (pair[0], -pair[1]))[1]
    selection_file.write_text(json.dumps({"C": chosen, "source": "validation_only"}) + "\n",
                              encoding="utf-8")
    return chosen


def run_linear(fold, store, kind):
    c = linear_select(fold, store, kind)
    for seed in range(3):
        if (RESULTS / f"pred_{kind}_{fold.task}_{fold.fold}_{seed}.csv").exists():
            continue
        model = linear_checkpoint(fold, store, kind, c, seed)
        started = time.monotonic()
        claim_test(fold, kind, seed)
        pred = linear_predict(model, fold, store, fold.test, kind)
        save_test(fold, kind, seed, pred, time.monotonic() - started)


def run_b6(fold):
    for seed in range(3):
        target = RESULTS / f"pred_B-6_{fold.task}_{fold.fold}_{seed}.csv"
        if target.exists():
            continue
        started = time.monotonic()
        frames = [pd.read_csv(RESULTS / f"pred_{kind}_{fold.task}_{fold.fold}_{seed}.csv")
                  for kind in ("B-6_DINO", "B-6_CLIP", "B-6_SIG")]
        assert all(frame.image_id.tolist() == frames[0].image_id.tolist()
                   for frame in frames)
        columns = (["prediction"] if fold.task in ("gothic", "romanesque", "dating")
                   else [name for name in frames[0].columns if name.startswith("prob_")])
        values = np.mean([frame[columns].to_numpy() for frame in frames], axis=0)
        prediction = values[:, 0] if len(columns) == 1 else values
        claim_test(fold, "B-6", seed)
        save_test(fold, "B-6", seed, prediction, time.monotonic() - started)


def run_neural(fold, metadata, store, kind):
    seeds = range(3)
    for seed in seeds:
        target = RESULTS / f"pred_{kind}_{fold.task}_{fold.fold}_{seed}.csv"
        if target.exists():
            continue
        checkpoint = CHECKPOINTS / f"{kind}_{fold.task}_{fold.fold}_{seed}.pt"
        if not checkpoint.exists():
            if kind in NEURAL_FROZEN:
                record = train_frozen_probe(fold, store, kind, seed, checkpoint)
            else:
                record = train_online_probe(fold, metadata, kind, seed, checkpoint)
            append_row({"model": kind, "task": fold.task, "fold": fold.fold,
                        "stage": "early_stop", "config_id": "fixed_recipe",
                        "seed": seed, "checkpoint": str(checkpoint), **record})
        if kind in NEURAL_FROZEN:
            claim_test(fold, kind, seed)
            pred, seconds = evaluate_frozen_probe(fold, store, kind, checkpoint)
        else:
            claim_test(fold, kind, seed)
            pred, seconds = evaluate_online_probe(fold, metadata, kind, checkpoint)
        save_test(fold, kind, seed, pred, seconds)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=TASKS, required=True)
    args = parser.parse_args()
    assert torch.cuda.is_available()
    metadata = pd.read_parquet(PROCESSED / "metadata.parquet")
    folds = task_folds(metadata, args.task)
    store = FeatureStore(include_tokens=False)
    for fold in folds:
        for kind in LINEAR:
            run_linear(fold, store, kind)
            print(f"Completed {kind} {fold.task} fold {fold.fold}", flush=True)
        run_b6(fold)
        for kind in ("B-3", "B-5"):
            run_neural(fold, metadata, store, kind)
            print(f"Completed {kind} {fold.task} fold {fold.fold}", flush=True)
    del store
    token_store = FeatureStore()
    for fold in folds:
        run_neural(fold, metadata, token_store, "B-7")
        print(f"Completed B-7 {fold.task} fold {fold.fold}", flush=True)


if __name__ == "__main__":
    main()
