"""Preflight planned online batch 16 on training-fold images only."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import pandas as pd
import torch
from baselines_v2 import OnlineImages, OnlineProbe, target_loss
from chronoarch_tasks import task_folds
from chronoarch_train import OnlineFeatureStore, TrainConfig, make_lora_model
from pathing import PROCESSED


def main():
    assert torch.cuda.is_available()
    metadata = pd.read_parquet(PROCESSED / "metadata.parquet")
    fold = task_folds(metadata, "wc14")[0]
    idx = fold.train[:16]
    results = {}
    for kind in ("ChronoArch-L", "B-8", "B-9"):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        if kind == "ChronoArch-L":
            model, processor = make_lora_model(fold, TrainConfig())
            store = OnlineFeatureStore(metadata, processor)
            with torch.autocast("cuda", dtype=torch.float16):
                output = model(*store.online_batch(idx, "base"))
                loss = model.head.loss(output,
                    torch.as_tensor(fold.style[idx], device="cuda"),
                    torch.as_tensor(fold.fine[idx], device="cuda"),
                    torch.as_tensor(fold.year[idx], device="cuda"), 1, .3)
        else:
            model = OnlineProbe(kind, 14).to("cuda")
            store = OnlineImages(metadata, kind)
            with torch.autocast("cuda", dtype=torch.float16):
                output = model(store.batch(idx, "base"))
                loss = target_loss(output, fold, idx)
        assert torch.isfinite(loss)
        loss.backward()
        results[kind] = {"batch_size": len(idx), "backward_ok": True,
                         "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                         "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
                         "optimizer_step": False, "test_split_accessed": False}
        del model, store, output, loss
    path = ROOT / "results_v2" / "online_batch_preflight.json"
    path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(results, indent=2), flush=True)


if __name__ == "__main__":
    main()
