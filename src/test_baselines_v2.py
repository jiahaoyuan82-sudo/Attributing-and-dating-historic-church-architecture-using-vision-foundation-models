"""B3 GPU preflight on training/validation folds only; no optimizer step."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import torch

from baselines_v2 import (OnlineImages, OnlineProbe, Probe, frozen_input,
                          frozen_predictions, online_predictions, target_loss)
from chronoarch_tasks import task_folds
from chronoarch_train import FeatureStore
from pathing import PROCESSED
from run_baselines_v2 import LINEAR, matrix


def main():
    assert torch.cuda.is_available()
    metadata = pd.read_parquet(PROCESSED / "metadata.parquet")
    fold = task_folds(metadata, "wc14")[0]
    store = FeatureStore()
    train = fold.train[:2]
    report = {"device": "cuda", "test_split_accessed": False}
    report["linear_inputs"] = {}
    for kind in LINEAR:
        value = matrix(store, train, "base", kind)
        assert value.shape[0] == 2 and np.isfinite(value).all()
        report["linear_inputs"][kind] = list(value.shape)
    report["frozen_neural"] = {}
    for kind, width in (("B-3", 2048), ("B-5", 3584), ("B-7", 1024)):
        model = Probe("attn" if kind == "B-7" else "mlp", width, 14).to("cuda")
        with torch.autocast("cuda", dtype=torch.float16):
            loss = target_loss(model(frozen_input(store, train, "base", kind)), fold, train)
        assert torch.isfinite(loss)
        loss.backward()
        score = frozen_predictions(model, fold, store, fold.val[:1], kind)
        assert score.shape == (1, 14) and np.isfinite(score).all()
        report["frozen_neural"][kind] = {"backward": True, "validation_readout": list(score.shape)}
        del model
        torch.cuda.empty_cache()
    report["online_neural"] = {}
    for kind in ("B-8", "B-9"):
        images = OnlineImages(metadata, kind)
        model = OnlineProbe(kind, 14).to("cuda")
        with torch.autocast("cuda", dtype=torch.float16):
            loss = target_loss(model(images.batch(train[:1], "base")), fold, train[:1])
        assert torch.isfinite(loss)
        loss.backward()
        assert model.head.weight.grad is not None
        score = online_predictions(model, fold, images, fold.val[:1], batch_size=1)
        assert score.shape == (1, 14) and np.isfinite(score).all()
        report["online_neural"][kind] = {"backward": True, "validation_readout": list(score.shape)}
        del model, images
        torch.cuda.empty_cache()
    target = ROOT / "results_v2" / "b3_preflight.json"
    target.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
