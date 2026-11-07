"""B2 task split, leakage mask, loss and readout checks; no dataset test evaluation."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import torch

from chronoarch import ChronoArchF
from chronoarch_tasks import task_folds
from chronoarch_train import (FeatureStore, OnlineFeatureStore, TrainConfig,
                              make_model, make_lora_model, predictions, lora_predictions)
from pathing import PROCESSED, RAW


def check_folds(metadata):
    names = dict(line.split(" ", 1) for line in (RAW / "style_names.txt").read_text(encoding="utf-8").splitlines())
    result = {}
    for task in ("wc4_rq4", "wc6", "wc14", "gothic", "romanesque", "dating"):
        folds = task_folds(metadata, task)
        assert len(folds) == (1 if task in ("wc4_rq4", "wc6", "wc14") else 5)
        for fold in folds:
            assert len(set(fold.train) & set(fold.val)) == 0
            assert len(set(fold.train) & set(fold.test)) == 0
            assert len(set(fold.val) & set(fold.test)) == 0
            assert fold.style[fold.train].min() >= 0
            assert fold.style[fold.val].min() >= 0
            if task != "wc4_rq4":
                assert fold.style[fold.test].min() >= 0
            if task in ("gothic", "romanesque"):
                assert all("revival" not in names.get(qid, "").lower() for qid in fold.fine_names)
                assert np.any(fold.fine[fold.eligible] == -1)
            if task == "dating":
                assert np.isfinite(fold.year[fold.eligible]).all()
            if task == "wc4_rq4":
                import json
                annotations = json.loads((RAW / "building_parts.json").read_text(
                    encoding="utf-8"))["annotations"]
                assert len(fold.test) == len(annotations) == 139
                assert set(fold.image_ids[fold.test]) == set(annotations)
                annotated_churches = set(fold.churches[fold.test])
                assert not annotated_churches & set(fold.churches[fold.train])
                assert not annotated_churches & set(fold.churches[fold.val])
                assert np.all(metadata.split_wc4.to_numpy()[fold.train] == "train")
                assert np.all(metadata.split_wc4.to_numpy()[fold.val] == "val")
                old = json.loads((ROOT / "results" / "rq4_classifier_info.json").read_text(
                    encoding="utf-8"))["DINO-B"]
                assert len(fold.train) == old["train_images_retained"]
                assert len(fold.val) == old["val_images_retained"]
        result[task] = {"folds": len(folds), "eligible_images": len(folds[0].eligible),
                        "style_classes": len(folds[0].style_names), "fine_classes": len(folds[0].fine_names)}
    return result


def check_task_forward_backward():
    assert torch.cuda.is_available()
    result = {}
    for task, classes in (("style", 6), ("authenticity", 2), ("dating", 14)):
        model = ChronoArchF(classes, 10, token_dim=64, clip_dim=32, sig_dim=32,
                            task=task, tau=1, class_counts=np.ones(classes)).to("cuda")
        batch = (torch.randn(2, 256, 64, device="cuda"), torch.randn(2, 256, 64, device="cuda"),
                 torch.randn(2, 64, device="cuda"), torch.randn(2, 64, device="cuda"),
                 torch.randn(2, 32, device="cuda"), torch.randn(2, 32, device="cuda"))
        output = model(*batch)
        target = torch.tensor([0, 1], device="cuda")
        fine = torch.tensor([-1, 2], device="cuda")
        year = torch.tensor([float("nan"), 1420.], device="cuda")
        loss = model.loss(output, target, fine, year, lambda_fine=1, lambda_year=.3)
        assert torch.isfinite(loss)
        loss.backward()
        assert model.query.grad is not None and torch.isfinite(model.query.grad).all()
        readout = model.readout(output)
        assert readout.shape == ((2,) if task in ("authenticity", "dating") else (2, classes))
        if task == "dating":
            assert torch.all((readout >= 825) & (readout <= 2025))
        else:
            assert torch.all((readout >= 0) & (readout <= 1))
        result[task] = {"loss_finite": True, "backward_ok": True, "readout_shape": list(readout.shape)}
    return result


def check_real_train_and_validation_batches(metadata):
    store = FeatureStore()
    result = {}
    for task in ("wc4_rq4", "wc6", "wc14", "gothic", "romanesque", "dating"):
        fold = task_folds(metadata, task)[0]
        model = make_model(fold, TrainConfig())
        indexes = fold.train[:2]
        with torch.autocast("cuda", dtype=torch.float16):
            output = model(*store.batch(indexes, "base"))
            loss = model.loss(output,
                              torch.as_tensor(fold.style[indexes], device="cuda"),
                              torch.as_tensor(fold.fine[indexes], device="cuda"),
                              torch.as_tensor(fold.year[indexes], device="cuda"), 1, .3)
        assert torch.isfinite(loss)
        loss.backward()
        assert model.query.grad is not None and torch.isfinite(model.query.grad).all()
        score = predictions(model, fold, store, fold.val[:2], tta=True, batch_size=2)
        assert len(score) == 2 and np.isfinite(score).all()
        result[task] = {"real_training_batch_backward": True,
                        "validation_tta_readout_shape": list(score.shape),
                        "test_split_accessed": False}
        del model
        torch.cuda.empty_cache()
    return result


def check_planned_batch_size(metadata):
    """Preflight the specified F batch of 64 without optimizer updates."""
    fold = task_folds(metadata, "wc14")[0]
    store = FeatureStore()
    model = make_model(fold, TrainConfig())
    indexes = fold.train[:64]
    torch.cuda.reset_peak_memory_stats()
    with torch.autocast("cuda", dtype=torch.float16):
        output = model(*store.batch(indexes, "base"))
        loss = model.loss(output,
                          torch.as_tensor(fold.style[indexes], device="cuda"),
                          torch.as_tensor(fold.fine[indexes], device="cuda"),
                          torch.as_tensor(fold.year[indexes], device="cuda"), 1, .3)
    assert torch.isfinite(loss)
    loss.backward()
    return {"batch_size": len(indexes), "backward_ok": True,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
            "optimizer_step": False}


def check_lora_real_batch(metadata):
    fold = task_folds(metadata, "wc14")[0]
    model, processor = make_lora_model(fold, TrainConfig())
    store = OnlineFeatureStore(metadata, processor)
    indexes = fold.train[:1]
    torch.cuda.reset_peak_memory_stats()
    with torch.autocast("cuda", dtype=torch.float16):
        output = model(*store.online_batch(indexes, "base"))
        loss = model.head.loss(output,
            torch.as_tensor(fold.style[indexes], device="cuda"),
            torch.as_tensor(fold.fine[indexes], device="cuda"),
            torch.as_tensor(fold.year[indexes], device="cuda"), 1, .3)
    assert torch.isfinite(loss)
    loss.backward()
    lora_gradients = [p.grad for n, p in model.named_parameters()
                      if ".up." in n and p.requires_grad]
    assert lora_gradients and all(g is not None and torch.isfinite(g).all()
                                  for g in lora_gradients)
    score = lora_predictions(model, fold, store, fold.val[:1], tta=True, batch_size=1)
    assert score.shape == (1, 14) and np.isfinite(score).all()
    return {"real_training_batch_backward": True, "validation_tta_readout_shape": list(score.shape),
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
            "optimizer_step": False, "test_split_accessed": False}


def check_ablation_variants():
    variants = ("A0", "A1", "A1a", "A1b", "A2", "A2a", "A2b",
                "A3", "A3a", "A3b", "A4", "R1", "R2", "R3", "R4")
    batch = (torch.randn(2, 256, 64, device="cuda"),
             torch.randn(2, 256, 64, device="cuda"),
             torch.randn(2, 64, device="cuda"), torch.randn(2, 64, device="cuda"),
             torch.randn(2, 32, device="cuda"), torch.randn(2, 32, device="cuda"))
    for variant in variants:
        model = ChronoArchF(6, 8, token_dim=64, clip_dim=32, sig_dim=32,
                            variant=variant).to("cuda")
        with torch.autocast("cuda", dtype=torch.float16):
            output = model(*batch)
            loss = model.loss(output, torch.tensor([0, 1], device="cuda"),
                              torch.tensor([2, 3], device="cuda"),
                              torch.tensor([1450., 1750.], device="cuda"), 1, .3)
        assert torch.isfinite(loss)
        loss.backward()
        assert model.style_head.weight.grad is not None
    return {"variants": list(variants), "all_cuda_forward_backward_ok": True}


def check_r1_real_batch(metadata):
    fold = task_folds(metadata, "wc6")[0]
    model = make_model(fold, TrainConfig(variant="R1"))
    store = FeatureStore()
    indices = fold.train[:2]
    with torch.autocast("cuda", dtype=torch.float16):
        output = model(*store.batch(indices, "base", backbone="DINO-B"))
        loss = model.loss(output,
                          torch.as_tensor(fold.style[indices], device="cuda"),
                          torch.as_tensor(fold.fine[indices], device="cuda"),
                          torch.as_tensor(fold.year[indices], device="cuda"), 1, .3)
    assert torch.isfinite(loss)
    loss.backward()
    score = predictions(model, fold, store, fold.val[:1], tta=True, batch_size=1)
    assert score.shape == (1, 6) and np.isfinite(score).all()
    return {"backbone": "DINO-B", "train_backward": True,
            "validation_readout_shape": list(score.shape), "test_split_accessed": False}


if __name__ == "__main__":
    metadata = pd.read_parquet(PROCESSED / "metadata.parquet")
    report = {"device": "cuda", "split_checks": check_folds(metadata),
              "task_forward_backward": check_task_forward_backward(),
              "real_train_and_val_batches": check_real_train_and_validation_batches(metadata),
              "planned_batch_preflight": check_planned_batch_size(metadata),
              "lora_real_batch": check_lora_real_batch(metadata),
              "ablation_variants": check_ablation_variants(),
              "r1_real_batch": check_r1_real_batch(metadata)}
    output = ROOT / "results_v2" / "b2_task_tests.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
