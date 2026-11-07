"""Validation-only fair rerun of B-2 through B-7, isolated from original results."""
import argparse
import csv
import json
import math
import pickle
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from sklearn.linear_model import LogisticRegression, Ridge

sys.path.insert(0, str(Path(__file__).resolve().parent))
from baselines_v2 import frozen_input
from chronoarch import YEAR_CENTERS
from chronoarch_tasks import task_folds, validation_metric
from chronoarch_train import FeatureStore, set_seed
from pathing import PROCESSED, ROOT

OUT = ROOT / "results_v2" / "baselines_fair"
CKPT = OUT / "checkpoints"
TASKS = ("wc6", "wc14", "gothic", "romanesque", "dating")
LINEAR = ("B-2_224", "B-2_448", "B-4", "B-6_DINO", "B-6_CLIP", "B-6_SIG")
NEURAL = ("B-3", "B-5", "B-7")
INITIAL_GRID = (1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, .1, .3, 1., 3., 10.)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def append_csv(path, row):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(row))
        if stream.tell() == 0:
            writer.writeheader()
        writer.writerow(row)
        stream.flush()


def l2(block):
    block = np.asarray(block, dtype=np.float32)
    return block / np.maximum(np.linalg.norm(block, axis=1, keepdims=True), 1e-12)


def linear_matrix(store, indices, view, kind):
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
    return np.concatenate([l2(a[key][indices]) for key in keys], axis=1)


def linear_predict(model, fold, store, indices, kind):
    base = linear_matrix(store, indices, "base", kind)
    flip = linear_matrix(store, indices, "flip", kind)
    if fold.task == "dating":
        return (model.predict(base) + model.predict(flip)) / 2
    values = (model.predict_proba(base) + model.predict_proba(flip)) / 2
    return values[:, 1] if fold.task in ("gothic", "romanesque") else values


def linear_fit(fold, store, kind, value):
    x = linear_matrix(store, fold.train, "base", kind)
    y = fold.year[fold.train] if fold.task == "dating" else fold.style[fold.train]
    model = (Ridge(alpha=value, solver="lsqr") if fold.task == "dating" else
             LogisticRegression(C=value, max_iter=3000, random_state=0))
    started = time.monotonic()
    model.fit(x, y)
    seconds = time.monotonic() - started
    return model, seconds


def linear_checkpoint(fold, store, kind, value):
    path = CKPT / f"linear_{kind}_{fold.task}_{fold.fold}_{value:.12g}.pkl"
    if path.exists():
        with path.open("rb") as stream:
            return pickle.load(stream)
    model, seconds = linear_fit(fold, store, kind, value)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as stream:
        pickle.dump(model, stream)
    truth = fold.year[fold.val] if fold.task == "dating" else fold.style[fold.val]
    score = validation_metric(fold.task, truth, linear_predict(model, fold, store, fold.val, kind))
    append_csv(OUT / "selection_log_linear.csv", dict(task=fold.task, fold=fold.fold,
               model=kind, parameter="alpha" if fold.task == "dating" else "C",
               value=value, seed=0, val_metric=score, train_seconds=seconds, checkpoint=str(path)))
    return model


def linear_select(fold, store, kind):
    path = OUT / f"selected_{kind}_{fold.task}_{fold.fold}.json"
    if path.exists():
        return float(json.loads(path.read_text(encoding="utf-8"))["value"])
    if fold.task in ("gothic", "romanesque", "dating") and fold.fold > 0:
        reference = OUT / f"selected_{kind}_{fold.task}_0.json"
        assert reference.exists()
        value = float(json.loads(reference.read_text(encoding="utf-8"))["value"])
        write_json(path, dict(value=value, source="fold_0_validation_only"))
        return value
    scored = {}
    truth = fold.year[fold.val] if fold.task == "dating" else fold.style[fold.val]
    def score(value):
        if value not in scored:
            model = linear_checkpoint(fold, store, kind, value)
            scored[value] = validation_metric(fold.task, truth,
                                              linear_predict(model, fold, store, fold.val, kind))
    for value in INITIAL_GRID:
        score(value)
    while True:
        values = sorted(scored)
        best = max(values, key=lambda v: (scored[v], -v))
        if best != values[0] and best != values[-1]:
            break
        edge = values[0] if best == values[0] else values[-1]
        new = float(f"{edge / 10 if best == values[0] else edge * 10:.12g}")
        if new in scored or not 1e-15 <= new <= 1e15:
            raise RuntimeError(f"Grid edge unresolved: {kind} {fold.task} {best}")
        score(new)
    write_json(path, dict(value=best, source="validation_only", grid=sorted(scored),
                          validation_metric=scored[best]))
    return best


def save_test(fold, kind, seed, prediction):
    target = OUT / f"pred_{kind}_{fold.task}_{fold.fold}_{seed}.csv"
    marker = target.with_suffix(".test_started")
    assert not target.exists() and not marker.exists(), str(target)
    marker.write_text("test evaluation started\n", encoding="utf-8")
    values = ({"prediction": prediction} if prediction.ndim == 1 else
              {f"prob_{i}": prediction[:, i] for i in range(prediction.shape[1])})
    data = pd.DataFrame({"image_id": fold.image_ids[fold.test],
                         "church_id": fold.churches[fold.test],
                         "truth": fold.year[fold.test] if fold.task == "dating" else fold.style[fold.test],
                         **values})
    temporary = target.with_suffix(".tmp")
    data.to_csv(temporary, index=False)
    temporary.replace(target)


def run_linear(fold, store, kind):
    value = linear_select(fold, store, kind)
    path = OUT / f"pred_{kind}_{fold.task}_{fold.fold}_0.csv"
    if not path.exists():
        model = linear_checkpoint(fold, store, kind, value)
        save_test(fold, kind, 0, linear_predict(model, fold, store, fold.test, kind))


@dataclass(frozen=True)
class Config:
    lr: float
    tau: int
    dropout: float


class FairProbe(nn.Module):
    def __init__(self, kind, outputs, dropout):
        super().__init__()
        self.kind = kind
        if kind == "B-7":
            self.norm = nn.LayerNorm(1024)
            self.query = nn.Parameter(torch.randn(1, 1024) / math.sqrt(1024))
            self.attention = nn.MultiheadAttention(1024, 8, dropout=dropout, batch_first=True)
            self.head = nn.Linear(1024, outputs)
        else:
            width = 2048 if kind == "B-3" else 3584
            self.net = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 1024), nn.GELU(),
                                     nn.Dropout(dropout), nn.Linear(1024, 1024), nn.GELU(),
                                     nn.Dropout(dropout), nn.Linear(1024, outputs))

    def forward(self, value):
        if self.kind != "B-7":
            return self.net(value)
        tokens = self.norm(value)
        query = self.query[None].expand(len(tokens), -1, -1)
        pooled, _ = self.attention(query, tokens, tokens, need_weights=False)
        return self.head(pooled[:, 0])


def neural_predict(model, fold, store, indices, kind, batch_size=64):
    model.eval()
    outputs = []
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        for start in range(0, len(indices), batch_size):
            part = indices[start:start + batch_size]
            base = model(frozen_input(store, part, "base", kind)).softmax(-1)
            flip = model(frozen_input(store, part, "flip", kind)).softmax(-1)
            prob = (base + flip) / 2
            if fold.task == "dating":
                result = (prob * YEAR_CENTERS.to("cuda")).sum(-1)
            elif fold.task in ("gothic", "romanesque"):
                result = prob[:, 1]
            else:
                result = prob
            outputs.append(result.float().cpu().numpy())
    return np.concatenate(outputs)


def neural_train(fold, store, kind, config, seed, checkpoint):
    assert torch.cuda.is_available()
    set_seed(seed)
    model = FairProbe(kind, 25 if fold.task == "dating" else len(fold.style_names),
                      config.dropout).to("cuda")
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=.05)
    scaler = torch.amp.GradScaler("cuda")
    rng = np.random.default_rng(seed)
    counts = np.bincount(fold.style[fold.train], minlength=len(fold.style_names)) + 1e-6
    prior = torch.as_tensor(np.log(counts / counts.sum()), device="cuda", dtype=torch.float32)
    best, stale = -np.inf, 0
    started = time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    for epoch in range(40):
        factor = (epoch + 1) / 3 if epoch < 3 else .5 * (1 + math.cos(math.pi * (epoch - 3) / 37))
        for group in optimizer.param_groups:
            group["lr"] = config.lr * factor
        model.train()
        order = rng.permutation(fold.train)
        for start in range(0, len(order), 64):
            part = order[start:start + 64]
            view = ("base", "flip")[int(rng.integers(2))]
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16):
                logits = model(frozen_input(store, part, view, kind))
                if fold.task == "dating":
                    year = torch.as_tensor(fold.year[part], device="cuda")
                    bins = (year - 800) / 50 - .5
                    centers = torch.arange(25, device="cuda")
                    target = torch.exp(-.5 * (centers[None, :] - bins[:, None]).square())
                    target /= target.sum(-1, keepdim=True)
                    loss = F.kl_div(F.log_softmax(logits, -1), target, reduction="batchmean")
                else:
                    labels = torch.as_tensor(fold.style[part], device="cuda")
                    loss = F.cross_entropy(logits + config.tau * prior, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        truth = fold.year[fold.val] if fold.task == "dating" else fold.style[fold.val]
        score = validation_metric(fold.task, truth, neural_predict(model, fold, store, fold.val, kind))
        if score > best + 1e-8:
            best, stale = float(score), 0
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"state_dict": model.state_dict(), "config": asdict(config),
                        "seed": seed, "val_metric": best, "best_epoch": epoch}, checkpoint)
        else:
            stale += 1
            if stale >= 6:
                break
    return dict(val_metric=best, best_epoch=epoch, epochs_run=epoch + 1,
                train_seconds=time.monotonic() - started,
                peak_reserved_bytes=torch.cuda.max_memory_reserved())


def neural_checkpoint(fold, store, kind, config, seed):
    key = f"lr{config.lr:g}_tau{config.tau}_drop{config.dropout:g}"
    path = CKPT / f"{kind}_{fold.task}_{fold.fold}_{key}_{seed}.pt"
    if not path.exists():
        record = neural_train(fold, store, kind, config, seed, path)
        append_csv(OUT / "selection_log_neural.csv", dict(task=fold.task, fold=fold.fold,
                   model=kind, **asdict(config), seed=seed, **record, checkpoint=str(path)))
    return path


def neural_select(fold, store, kind):
    path = OUT / f"selected_{kind}_{fold.task}_{fold.fold}.json"
    if path.exists():
        return Config(**json.loads(path.read_text(encoding="utf-8"))["config"])
    if fold.task in ("gothic", "romanesque", "dating") and fold.fold > 0:
        reference = OUT / f"selected_{kind}_{fold.task}_0.json"
        assert reference.exists()
        config = Config(**json.loads(reference.read_text(encoding="utf-8"))["config"])
        write_json(path, dict(config=asdict(config), source="fold_0_validation_only"))
        return config
    scores = []
    for lr in (.0005, .001):
        for tau in (0, 1):
            for dropout in (.1, .2):
                config = Config(lr, tau, dropout)
                checkpoint = neural_checkpoint(fold, store, kind, config, 0)
                saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
                scores.append((float(saved["val_metric"]), config))
    chosen = max(scores, key=lambda item: (item[0], -item[1].lr,
                                         -item[1].tau, -item[1].dropout))[1]
    write_json(path, dict(config=asdict(chosen), source="validation_only",
                          val_metric=max(scores, key=lambda item: item[0])[0]))
    return chosen


def run_neural(fold, store, kind):
    config = neural_select(fold, store, kind)
    for seed in range(3):
        target = OUT / f"pred_{kind}_{fold.task}_{fold.fold}_{seed}.csv"
        if target.exists():
            continue
        checkpoint = neural_checkpoint(fold, store, kind, config, seed)
        saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
        model = FairProbe(kind, 25 if fold.task == "dating" else len(fold.style_names),
                          config.dropout).to("cuda")
        model.load_state_dict(saved["state_dict"])
        save_test(fold, kind, seed, neural_predict(model, fold, store, fold.test, kind))


def run_b6(fold):
    target = OUT / f"pred_B-6_{fold.task}_{fold.fold}_0.csv"
    if target.exists():
        return
    frames = [pd.read_csv(OUT / f"pred_{kind}_{fold.task}_{fold.fold}_0.csv")
              for kind in ("B-6_DINO", "B-6_CLIP", "B-6_SIG")]
    assert all(frame.image_id.tolist() == frames[0].image_id.tolist() for frame in frames)
    columns = (["prediction"] if fold.task in ("gothic", "romanesque", "dating") else
               [name for name in frames[0] if name.startswith("prob_")])
    values = np.mean([frame[columns].to_numpy() for frame in frames], axis=0)
    save_test(fold, "B-6", 0, values[:, 0] if len(columns) == 1 else values)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=TASKS, required=True)
    parser.add_argument("--group", choices=("linear", "neural"), required=True)
    args = parser.parse_args()
    assert torch.cuda.is_available(), "CUDA mandatory"
    metadata = pd.read_parquet(PROCESSED / "metadata.parquet")
    folds = task_folds(metadata, args.task)
    if args.group == "linear":
        store = FeatureStore(include_tokens=False)
        for fold in folds:
            for kind in LINEAR:
                run_linear(fold, store, kind)
                print("completed", fold.task, fold.fold, kind, flush=True)
            run_b6(fold)
    else:
        store = FeatureStore(include_tokens=False)
        for fold in folds:
            for kind in ("B-3", "B-5"):
                run_neural(fold, store, kind)
                print("completed", fold.task, fold.fold, kind, flush=True)
        del store
        store = FeatureStore()
        for fold in folds:
            run_neural(fold, store, "B-7")
            print("completed", fold.task, fold.fold, "B-7", flush=True)


if __name__ == "__main__":
    main()
