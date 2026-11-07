"""B3 frozen-feature and online-backbone baselines; all neural training uses CUDA."""
import copy
import math
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from chronoarch import LoRALinear, YEAR_CENTERS
from chronoarch_tasks import validation_metric
from chronoarch_train import FeatureStore, set_seed
from pathing import ROOT

os.environ.setdefault("HF_HOME", str(ROOT / "env" / "hf_cache"))


def target_loss(logits, fold, indices):
    if fold.task != "dating":
        truth = torch.as_tensor(fold.style[indices], device="cuda")
        return F.cross_entropy(logits, truth)
    year = torch.as_tensor(fold.year[indices], device="cuda")
    bins = (year - 800) / 50 - 0.5
    centers = torch.arange(25, device="cuda")
    target = torch.exp(-0.5 * (centers[None, :] - bins[:, None]).square())
    target /= target.sum(-1, keepdim=True)
    return F.kl_div(F.log_softmax(logits, -1), target, reduction="batchmean")


def readout(logits, fold):
    prob = logits.softmax(-1)
    if fold.task == "dating":
        return (prob * YEAR_CENTERS.to("cuda")).sum(-1)
    if fold.task in ("gothic", "romanesque"):
        return prob[:, 1]
    return prob


class Probe(nn.Module):
    def __init__(self, kind, input_dim, outputs, dropout=0.1):
        super().__init__()
        self.kind = kind
        if kind == "attn":
            self.norm = nn.LayerNorm(1024)
            self.query = nn.Parameter(torch.randn(1, 1024) / math.sqrt(1024))
            self.attention = nn.MultiheadAttention(1024, 8, batch_first=True)
            self.head = nn.Linear(1024, outputs)
        else:
            self.net = nn.Sequential(nn.LayerNorm(input_dim),
                                     nn.Linear(input_dim, 1024), nn.GELU(),
                                     nn.Dropout(dropout), nn.Linear(1024, 1024),
                                     nn.GELU(), nn.Dropout(dropout),
                                     nn.Linear(1024, outputs))

    def forward(self, value):
        if self.kind == "attn":
            tokens = self.norm(value)
            query = self.query[None].expand(len(tokens), -1, -1)
            pooled, _ = self.attention(query, tokens, tokens, need_weights=False)
            return self.head(pooled[:, 0])
        return self.net(value)


def frozen_input(store, indices, view, kind):
    a = store.arrays
    suffix = "flip" if view == "flip" else "base"
    if kind == "B-3":
        names = (f"DINO-L_224_{view}_cls", f"DINO-L_448_{view}_cls")
    elif kind == "B-5":
        names = (f"DINO-L_224_{view}_cls", f"DINO-L_448_{view}_cls",
                 f"CLIP-L_{suffix}", f"SIG2-B_{suffix}")
    elif kind == "B-7":
        names = (f"DINO-L_224_{view}_tokens",)
    else:
        raise ValueError(kind)
    values = [torch.from_numpy(np.array(a[name][indices], dtype=np.float32, copy=True)).to("cuda")
              for name in names]
    return values[0] if kind == "B-7" else torch.cat(values, dim=-1)


def frozen_predictions(model, fold, store, indices, kind, batch_size=64):
    model.eval()
    result = []
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        for start in range(0, len(indices), batch_size):
            part = indices[start:start + batch_size]
            base = model(frozen_input(store, part, "base", kind)).softmax(-1)
            flip = model(frozen_input(store, part, "flip", kind)).softmax(-1)
            probability = (base + flip) / 2
            if fold.task == "dating":
                output = (probability * YEAR_CENTERS.to("cuda")).sum(-1)
            elif fold.task in ("gothic", "romanesque"):
                output = probability[:, 1]
            else:
                output = probability
            result.append(output.float().cpu().numpy())
    return np.concatenate(result)


def train_frozen_probe(fold, store, kind, seed, checkpoint, *,
                       lr=0.0005, max_epochs=40, patience=6):
    assert kind in ("B-3", "B-5", "B-7") and torch.cuda.is_available()
    set_seed(seed)
    width = {"B-3": 2048, "B-5": 3584, "B-7": 1024}[kind]
    outputs = 25 if fold.task == "dating" else len(fold.style_names)
    model = Probe("attn" if kind == "B-7" else "mlp", width, outputs).to("cuda")
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=.05)
    scaler = torch.amp.GradScaler("cuda")
    rng = np.random.default_rng(seed)
    best, best_epoch, stale = -np.inf, -1, 0
    started = time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    for epoch in range(max_epochs):
        factor = (epoch + 1) / 3 if epoch < 3 else .5 * (1 + math.cos(
            math.pi * (epoch - 3) / max(1, max_epochs - 3)))
        for group in optimizer.param_groups:
            group["lr"] = lr * factor
        model.train()
        order = rng.permutation(fold.train)
        for start in range(0, len(order), 64):
            part = order[start:start + 64]
            view = ("base", "flip")[int(rng.integers(2))]
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16):
                logits = model(frozen_input(store, part, view, kind))
                loss = target_loss(logits, fold, part)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        pred = frozen_predictions(model, fold, store, fold.val, kind)
        truth = fold.year[fold.val] if fold.task == "dating" else fold.style[fold.val]
        score = validation_metric(fold.task, truth, pred)
        if score > best + 1e-8:
            best, best_epoch, stale = float(score), epoch, 0
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"state_dict": model.state_dict(), "kind": kind,
                        "seed": seed, "best_epoch": epoch, "val_metric": best}, checkpoint)
        else:
            stale += 1
            if stale >= patience:
                break
    return {"val_metric": best, "best_epoch": best_epoch,
            "epochs_run": epoch + 1, "train_seconds": time.monotonic() - started,
            "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved()}


def evaluate_frozen_probe(fold, store, kind, checkpoint):
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    width = {"B-3": 2048, "B-5": 3584, "B-7": 1024}[kind]
    outputs = 25 if fold.task == "dating" else len(fold.style_names)
    model = Probe("attn" if kind == "B-7" else "mlp", width, outputs).to("cuda")
    model.load_state_dict(saved["state_dict"])
    started = time.monotonic()
    pred = frozen_predictions(model, fold, store, fold.test, kind)
    return pred, time.monotonic() - started


class OnlineProbe(nn.Module):
    def __init__(self, kind, outputs):
        super().__init__()
        from transformers import AutoModel
        assert kind in ("B-8", "B-9")
        self.kind = kind
        identifier = ("facebook/dinov2-base" if kind == "B-8"
                      else str(ROOT / "env" / "dinov2-large"))
        self.backbone = AutoModel.from_pretrained(identifier, local_files_only=True)
        for p in self.backbone.parameters():
            p.requires_grad_(False)
        if kind == "B-8":
            for layer in self.backbone.encoder.layer[-2:]:
                for p in layer.parameters():
                    p.requires_grad_(True)
        else:
            for layer in self.backbone.encoder.layer[-4:]:
                attention = layer.attention
                for name in ("q_proj", "k_proj", "v_proj"):
                    setattr(attention, name, LoRALinear(getattr(attention, name),
                                                        rank=8, alpha=16))
            self.backbone.gradient_checkpointing_enable()
        self.head = nn.Linear(self.backbone.config.hidden_size, outputs)

    def forward(self, pixels):
        hidden = self.backbone(pixel_values=pixels).last_hidden_state
        return self.head(hidden[:, 0])


class OnlineImages:
    def __init__(self, metadata, kind):
        from transformers import AutoImageProcessor
        from extract_v2 import ImageViewDataset
        identifier = ("facebook/dinov2-base" if kind == "B-8"
                      else str(ROOT / "env" / "dinov2-large"))
        processor = AutoImageProcessor.from_pretrained(identifier, local_files_only=True)
        self.views = {view: ImageViewDataset(metadata, 224, view,
                                            processor.image_mean, processor.image_std)
                      for view in ("base", "flip", "crop0", "crop1")}

    def batch(self, indices, view):
        return torch.stack([self.views[view][int(i)] for i in indices]).to("cuda")


def online_predictions(model, fold, images, indices, batch_size=16):
    model.eval()
    result = []
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        for start in range(0, len(indices), batch_size):
            part = indices[start:start + batch_size]
            base = model(images.batch(part, "base")).softmax(-1)
            flip = model(images.batch(part, "flip")).softmax(-1)
            probability = (base + flip) / 2
            if fold.task == "dating":
                output = (probability * YEAR_CENTERS.to("cuda")).sum(-1)
            elif fold.task in ("gothic", "romanesque"):
                output = probability[:, 1]
            else:
                output = probability
            result.append(output.float().cpu().numpy())
    return np.concatenate(result)


def train_online_probe(fold, metadata, kind, seed, checkpoint):
    assert torch.cuda.is_available() and kind in ("B-8", "B-9")
    set_seed(seed)
    outputs = 25 if fold.task == "dating" else len(fold.style_names)
    model = OnlineProbe(kind, outputs).to("cuda")
    images = OnlineImages(metadata, kind)
    lr = .0001
    max_epochs = 20 if kind == "B-8" else 25
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                                  lr=lr, weight_decay=.05)
    scaler = torch.amp.GradScaler("cuda")
    rng = np.random.default_rng(seed)
    best, best_epoch, stale = -np.inf, -1, 0
    started = time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    for epoch in range(max_epochs):
        factor = (epoch + 1) / 3 if epoch < 3 else .5 * (1 + math.cos(
            math.pi * (epoch - 3) / max(1, max_epochs - 3)))
        for group in optimizer.param_groups:
            group["lr"] = lr * factor
        model.train()
        order = rng.permutation(fold.train)
        optimizer.zero_grad(set_to_none=True)
        steps = math.ceil(len(order) / 16)
        for step, start in enumerate(range(0, len(order), 16)):
            part = order[start:start + 16]
            view = ("base", "flip", "crop0", "crop1")[int(rng.integers(4))]
            with torch.autocast("cuda", dtype=torch.float16):
                logits = model(images.batch(part, view))
                loss = target_loss(logits, fold, part) / 4
            scaler.scale(loss).backward()
            if (step + 1) % 4 == 0 or step + 1 == steps:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
        pred = online_predictions(model, fold, images, fold.val)
        truth = fold.year[fold.val] if fold.task == "dating" else fold.style[fold.val]
        score = validation_metric(fold.task, truth, pred)
        if score > best + 1e-8:
            best, best_epoch, stale = float(score), epoch, 0
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            state = {k: v.cpu() for k, v in model.state_dict().items()
                     if k.startswith("head.") or
                     (kind == "B-8" and k.startswith(("backbone.encoder.layer.10.",
                                                       "backbone.encoder.layer.11."))) or
                     (kind == "B-9" and (".down." in k or ".up." in k))}
            torch.save({"state_dict": state, "kind": kind, "seed": seed,
                        "best_epoch": epoch, "val_metric": best}, checkpoint)
        else:
            stale += 1
            if stale >= 10:
                break
    return {"val_metric": best, "best_epoch": best_epoch,
            "epochs_run": epoch + 1, "train_seconds": time.monotonic() - started,
            "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved()}


def evaluate_online_probe(fold, metadata, kind, checkpoint):
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    outputs = 25 if fold.task == "dating" else len(fold.style_names)
    model = OnlineProbe(kind, outputs).to("cuda")
    model.load_state_dict(saved["state_dict"], strict=False)
    images = OnlineImages(metadata, kind)
    started = time.monotonic()
    pred = online_predictions(model, fold, images, fold.test)
    return pred, time.monotonic() - started
