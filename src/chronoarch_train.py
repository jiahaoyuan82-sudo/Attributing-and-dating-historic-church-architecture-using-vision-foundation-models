"""B2 task-specific CUDA training, validation selection, and one-time final readout.

No test-split features are accessed by search or early stopping. Every final
checkpoint is evaluated on its test/outer fold once, after selection is fixed.
"""
import json
import math
import os
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score, mean_absolute_error, roc_auc_score

from chronoarch import ChronoArchF, ChronoArchL, trainable_parameters
from chronoarch_tasks import TaskFold, validation_metric
from pathing import ROOT

FEATURES = ROOT / "features_v2"
OLD = ROOT / "features"
RESULTS = ROOT / "results_v2"
os.environ.setdefault("HF_HOME", str(ROOT / "env" / "hf_cache"))


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class FeatureStore:
    def __init__(self, include_b=False, include_tokens=True):
        self.arrays = {}
        for size in (224, 448):
            for view in ("base", "flip"):
                stem = f"DINO-L_{size}_{view}"
                if include_tokens:
                    source = np.load(FEATURES / f"{stem}_tokens.npy", mmap_mode="r")
                    pooled = np.empty((len(source), 64, source.shape[-1]), dtype=np.float16)
                    for start in range(0, len(source), 64):
                        end = min(start + 64, len(source))
                        block = np.asarray(source[start:end], dtype=np.float32).reshape(-1, 8, 2, 8, 2, source.shape[-1])
                        pooled[start:end] = block.mean(axis=(2, 4)).reshape(-1, 64, source.shape[-1]).astype(np.float16)
                    self.arrays[f"{stem}_tokens"] = pooled
                self.arrays[f"{stem}_cls"] = np.asarray(np.load(FEATURES / f"{stem}_cls.npy"), dtype=np.float16)
                b_stem = f"DINO-B_{size}_{view}"
                b_tokens = FEATURES / f"{b_stem}_tokens.npy"
                if include_tokens and include_b and b_tokens.exists():
                    b_source = np.load(b_tokens, mmap_mode="r")
                    b_pooled = np.empty((len(b_source), 64, b_source.shape[-1]), dtype=np.float16)
                    for start in range(0, len(b_source), 64):
                        end = min(start + 64, len(b_source))
                        block = np.asarray(b_source[start:end], dtype=np.float32).reshape(-1, 8, 2, 8, 2, b_source.shape[-1])
                        b_pooled[start:end] = block.mean(axis=(2, 4)).reshape(-1, 64, b_source.shape[-1]).astype(np.float16)
                    self.arrays[f"{b_stem}_tokens"] = b_pooled
                    self.arrays[f"{b_stem}_cls"] = np.asarray(np.load(FEATURES / f"{b_stem}_cls.npy"), dtype=np.float16)
        self.arrays["CLIP-L_base"] = np.asarray(np.load(OLD / "CLIP-L.npy"), dtype=np.float16)
        self.arrays["SIG2-B_base"] = np.asarray(np.load(OLD / "SIG2-B.npy"), dtype=np.float16)
        self.arrays["CLIP-L_flip"] = np.asarray(np.load(FEATURES / "CLIP-L_flip.npy"), dtype=np.float16)
        self.arrays["SIG2-B_flip"] = np.asarray(np.load(FEATURES / "SIG2-B_flip.npy"), dtype=np.float16)

    def batch(self, indices, view="base", backbone="DINO-L"):
        semantic_view = "flip" if view == "flip" else "base"
        names = (f"{backbone}_224_{view}_tokens", f"{backbone}_448_{view}_tokens",
                 f"{backbone}_224_{view}_cls", f"{backbone}_448_{view}_cls",
                 f"CLIP-L_{semantic_view}", f"SIG2-B_{semantic_view}")
        return tuple(torch.from_numpy(np.array(self.arrays[name][indices], dtype=np.float16, copy=True)).to("cuda")
                     for name in names)


class OnlineFeatureStore(FeatureStore):
    """224-pixel online DINO input with aligned cached 448/semantic features."""
    def __init__(self, metadata, processor):
        super().__init__()
        from extract_v2 import ImageViewDataset
        self.images = {view: ImageViewDataset(metadata, 224, view,
                                             processor.image_mean, processor.image_std)
                       for view in ("base", "flip", "crop0", "crop1")}

    def online_batch(self, indices, view="base"):
        pixels = torch.stack([self.images[view][int(index)] for index in indices]).to("cuda")
        _, t448, _, cls448, clip, sig = self.batch(indices, view)
        return pixels, t448, cls448, clip, sig


@dataclass(frozen=True)
class TrainConfig:
    queries: int = 4
    lambda_fine: float = 1.0
    lambda_year: float = 0.3
    tau: int = 1
    lr: float = 0.0005
    dropout: float = 0.1
    batch_size: int = 64
    max_epochs: int = 40
    patience: int = 6
    warmup_epochs: int = 3
    weight_decay: float = 0.05
    variant: str = "A0"


@dataclass(frozen=True)
class LoRAConfig:
    batch_size: int = 16
    max_epochs: int = 25
    patience: int = 10
    warmup_epochs: int = 3
    lora_lr: float = 0.0001
    head_lr: float = 0.0005
    weight_decay: float = 0.05
    accumulation_steps: int = 4


def task_mode(task):
    if task in ("gothic", "romanesque"):
        return "authenticity"
    return "dating" if task == "dating" else "style"


def make_model(fold, config):
    counts = np.bincount(fold.style[fold.train], minlength=len(fold.style_names)) + 1e-6
    return ChronoArchF(len(fold.style_names), len(fold.fine_names),
                       queries=config.queries, dropout=config.dropout,
                       tau=config.tau, class_counts=counts,
                       task=task_mode(fold.task), variant=config.variant,
                       token_dim=768 if config.variant == "R1" else 1024).to("cuda")


def make_lora_model(fold, config, *, f_checkpoint=None):
    from transformers import AutoImageProcessor, AutoModel
    identifier = str(ROOT / "env" / "dinov2-large")
    processor = AutoImageProcessor.from_pretrained(identifier, local_files_only=True)
    backbone = AutoModel.from_pretrained(identifier, local_files_only=True)
    head = make_model(fold, config)
    if f_checkpoint is not None:
        saved = torch.load(f_checkpoint, map_location="cpu", weights_only=False)
        head.load_state_dict(saved["state_dict"])
    model = ChronoArchL(backbone, head).to("cuda")
    return model, processor


def epoch_lr(epoch, config):
    if epoch < config.warmup_epochs:
        return (epoch + 1) / config.warmup_epochs
    progress = (epoch - config.warmup_epochs) / max(1, config.max_epochs - config.warmup_epochs)
    return 0.5 * (1 + math.cos(math.pi * progress))


def predictions(model, fold, store, indices, *, tta, batch_size=64):
    model.eval()
    outputs = []
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        for start in range(0, len(indices), batch_size):
            batch_indices = indices[start:start + batch_size]
            backbone = "DINO-B" if model.variant == "R1" else "DINO-L"
            base = model(*store.batch(batch_indices, "base", backbone=backbone))
            if tta:
                flipped = model(*store.batch(batch_indices, "flip", backbone=backbone))
                if fold.task == "dating":
                    probability = (base["year_logits"].softmax(-1) + flipped["year_logits"].softmax(-1)) / 2
                    output = (probability * model.year_centers).sum(-1)
                else:
                    probability = (base["style_logits"].softmax(-1) + flipped["style_logits"].softmax(-1)) / 2
                    output = probability[:, 1] if fold.task in ("gothic", "romanesque") else probability
            else:
                output = model.readout(base)
            outputs.append(output.float().cpu().numpy())
    return np.concatenate(outputs)


def lora_predictions(model, fold, store, indices, *, tta=True, batch_size=16):
    model.eval()
    outputs = []
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        for start in range(0, len(indices), batch_size):
            part = indices[start:start + batch_size]
            base = model(*store.online_batch(part, "base"))
            if tta:
                flip = model(*store.online_batch(part, "flip"))
                key = "year_logits" if fold.task == "dating" else "style_logits"
                probability = (base[key].softmax(-1) + flip[key].softmax(-1)) / 2
                if fold.task == "dating":
                    readout = (probability * model.head.year_centers).sum(-1)
                elif fold.task in ("gothic", "romanesque"):
                    readout = probability[:, 1]
                else:
                    readout = probability
            else:
                readout = model.head.readout(base)
            outputs.append(readout.float().cpu().numpy())
    return np.concatenate(outputs)


def train_lora(fold, metadata, f_config, l_config, seed, checkpoint, f_checkpoint):
    """Tune last four DINO-L q/k/v LoRA blocks and the F head on CUDA."""
    assert torch.cuda.is_available()
    set_seed(seed)
    torch.cuda.reset_peak_memory_stats()
    model, processor = make_lora_model(fold, f_config, f_checkpoint=f_checkpoint)
    store = OnlineFeatureStore(metadata, processor)
    lora = [p for n, p in model.named_parameters() if p.requires_grad and n.startswith("backbone.")]
    head = [p for n, p in model.named_parameters() if p.requires_grad and n.startswith("head.")]
    optimizer = torch.optim.AdamW([{"params": lora, "lr": l_config.lora_lr},
                                   {"params": head, "lr": l_config.head_lr}],
                                  weight_decay=l_config.weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,
        lambda epoch: epoch_lr(epoch, l_config))
    scaler = torch.amp.GradScaler("cuda")
    best_metric, best_epoch, stale = -np.inf, -1, 0
    rng = np.random.default_rng(seed)
    started = time.monotonic()
    views = ("base", "flip", "crop0", "crop1")
    for epoch in range(l_config.max_epochs):
        model.train()
        order = rng.permutation(fold.train)
        optimizer.zero_grad(set_to_none=True)
        steps = math.ceil(len(order) / l_config.batch_size)
        for step, start in enumerate(range(0, len(order), l_config.batch_size)):
            indices = order[start:start + l_config.batch_size]
            view = views[int(rng.integers(len(views)))]
            with torch.autocast("cuda", dtype=torch.float16):
                output = model(*store.online_batch(indices, view))
                loss = model.head.loss(output,
                    torch.as_tensor(fold.style[indices], device="cuda"),
                    torch.as_tensor(fold.fine[indices], device="cuda"),
                    torch.as_tensor(fold.year[indices], device="cuda"),
                    f_config.lambda_fine, f_config.lambda_year)
                loss = loss / l_config.accumulation_steps
            scaler.scale(loss).backward()
            if (step + 1) % l_config.accumulation_steps == 0 or step + 1 == steps:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
        scheduler.step()
        pred = lora_predictions(model, fold, store, fold.val, tta=True)
        truth = fold.year[fold.val] if fold.task == "dating" else fold.style[fold.val]
        score = validation_metric(fold.task, truth, pred)
        if score > best_metric + 1e-8:
            best_metric, best_epoch, stale = float(score), epoch, 0
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"state_dict": {k: v.cpu() for k, v in model.state_dict().items()
                                       if k.startswith("head.") or ".down." in k or ".up." in k},
                        "f_config": asdict(f_config), "l_config": asdict(l_config),
                        "seed": seed, "task": fold.task, "fold": fold.fold,
                        "best_epoch": epoch, "val_metric": best_metric}, checkpoint)
        else:
            stale += 1
            if stale >= l_config.patience:
                break
    return {"task": fold.task, "fold": fold.fold, "seed": seed,
            "val_metric": best_metric, "best_epoch": best_epoch,
            "epochs_run": epoch + 1, "train_seconds": time.monotonic() - started,
            "trainable_parameters": trainable_parameters(model),
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
            "checkpoint": str(checkpoint)}


def evaluate_lora_checkpoint(fold, metadata, checkpoint):
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model, processor = make_lora_model(fold, TrainConfig(**saved["f_config"]))
    model.load_state_dict(saved["state_dict"], strict=False)
    store = OnlineFeatureStore(metadata, processor)
    started = time.monotonic()
    pred = lora_predictions(model, fold, store, fold.test, tta=True)
    return pred, time.monotonic() - started


def train_one(fold, store, config, seed, checkpoint, *, use_augmentation=True):
    assert torch.cuda.is_available(), "All deep-model computation requires CUDA"
    set_seed(seed)
    torch.cuda.reset_peak_memory_stats()
    model = make_model(fold, config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: epoch_lr(epoch, config))
    scaler = torch.amp.GradScaler("cuda")
    best_metric, best_epoch, stale = -np.inf, -1, 0
    started = time.monotonic()
    rng = np.random.default_rng(seed)
    views = ("base", "flip") if use_augmentation else ("base",)
    for epoch in range(config.max_epochs):
        model.train()
        order = rng.permutation(fold.train)
        for start in range(0, len(order), config.batch_size):
            indices = order[start:start + config.batch_size]
            view = views[int(rng.integers(len(views)))]
            batch = store.batch(indices, view,
                                backbone="DINO-B" if config.variant == "R1" else "DINO-L")
            style = torch.as_tensor(fold.style[indices], device="cuda")
            fine = torch.as_tensor(fold.fine[indices], device="cuda")
            year = torch.as_tensor(fold.year[indices], device="cuda")
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16):
                output = model(*batch)
                loss = model.loss(output, style, fine, year, config.lambda_fine, config.lambda_year)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        scheduler.step()
        val_pred = predictions(model, fold, store, fold.val, tta=True)
        truth = fold.year[fold.val] if fold.task == "dating" else fold.style[fold.val]
        score = validation_metric(fold.task, truth, val_pred)
        if score > best_metric + 1e-8:
            best_metric, best_epoch, stale = float(score), epoch, 0
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"state_dict": model.state_dict(), "config": asdict(config),
                        "seed": seed, "task": fold.task, "fold": fold.fold,
                        "best_epoch": epoch, "val_metric": best_metric}, checkpoint)
        else:
            stale += 1
            if stale >= config.patience:
                break
    return {"task": fold.task, "fold": fold.fold, "seed": seed,
            **asdict(config), "val_metric": best_metric,
            "best_epoch": best_epoch, "epochs_run": epoch + 1,
            "train_seconds": time.monotonic() - started,
            "trainable_parameters": trainable_parameters(model),
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
            "checkpoint": str(checkpoint)}


def evaluate_checkpoint(fold, store, checkpoint, *, tta=True):
    """Call only once after validation has fixed the final configuration."""
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    config = TrainConfig(**saved["config"])
    model = make_model(fold, config)
    model.load_state_dict(saved["state_dict"])
    started = time.monotonic()
    pred = predictions(model, fold, store, fold.test, tta=tta)
    return pred, time.monotonic() - started
