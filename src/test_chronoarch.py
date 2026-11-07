"""B2 architecture tests and real CUDA forward/backward audit."""
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
os.environ["HF_HOME"] = str(ROOT / "env" / "hf_cache")
sys.path.insert(0, str(ROOT / "code" / "src"))

import numpy as np
import pandas as pd
import torch
from transformers import Dinov2Config, Dinov2Model, AutoModel

from chronoarch import ChronoArchF, ChronoArchL, LoRALinear, trainable_parameters
from pathing import PROCESSED


def synthetic_tests():
    assert torch.cuda.is_available(), "B2 requires CUDA"
    device = "cuda"
    model = ChronoArchF(14, 52, queries=4, token_dim=64, clip_dim=32, sig_dim=32,
                        class_counts=np.ones(14)).to(device)
    data = (torch.randn(2, 256, 64, device=device), torch.randn(2, 256, 64, device=device),
            torch.randn(2, 64, device=device), torch.randn(2, 64, device=device),
            torch.randn(2, 32, device=device), torch.randn(2, 32, device=device))
    out = model(*data, return_attention=True)
    assert out["style_logits"].shape == (2, 14)
    assert out["fine_logits"].shape == (2, 52)
    assert out["year_logits"].shape == (2, 25)
    assert len(out["attention"]) == 2
    assert torch.allclose(out["gates"].sum(dim=1), torch.ones(2, 384, device=device), atol=1e-6)
    assert out["year_prediction"].min() >= 825 and out["year_prediction"].max() <= 2025
    loss = model.loss(out, torch.tensor([0, 1], device=device),
                      torch.tensor([0, -1], device=device),
                      torch.tensor([1400., float("nan")], device=device))
    assert torch.isfinite(loss)
    loss.backward()
    assert model.query.grad is not None and torch.isfinite(model.query.grad).all()
    return {"synthetic_f_loss": float(loss.item()), "synthetic_f_backward": True}


def real_tests():
    metadata = pd.read_parquet(PROCESSED / "metadata.parquet")
    wc14 = metadata[metadata.wc14_style_id.notna()]
    fine_classes = wc14.all_styles.map(json.loads).map(lambda x: x[0]).nunique()
    style_classes = wc14.wc14_style_id.nunique()
    torch.cuda.reset_peak_memory_stats()
    head = ChronoArchF(style_classes, fine_classes, queries=4, dropout=0.1).to("cuda")
    count_f = trainable_parameters(head)
    args = (torch.randn(1, 256, 1024, device="cuda", dtype=torch.float16),
            torch.randn(1, 256, 1024, device="cuda", dtype=torch.float16),
            torch.randn(1, 1024, device="cuda", dtype=torch.float16),
            torch.randn(1, 1024, device="cuda", dtype=torch.float16),
            torch.randn(1, 768, device="cuda", dtype=torch.float16),
            torch.randn(1, 768, device="cuda", dtype=torch.float16))
    start = time.monotonic()
    with torch.autocast("cuda", dtype=torch.float16):
        output = head(*args)
        loss = head.loss(output, torch.tensor([0], device="cuda"),
                         torch.tensor([0], device="cuda"), torch.tensor([1400.], device="cuda"))
    loss.backward()
    assert torch.isfinite(loss) and head.query.grad is not None
    elapsed_f = time.monotonic() - start
    del head, output, loss
    torch.cuda.empty_cache()

    backbone = AutoModel.from_pretrained(str(ROOT / "env" / "dinov2-large"), local_files_only=True).to("cuda")
    head = ChronoArchF(style_classes, fine_classes, queries=4, dropout=0.1).to("cuda")
    model = ChronoArchL(backbone, head).to("cuda")
    count_l = trainable_parameters(model)
    lora_count = sum(p.numel() for name, p in model.named_parameters() if p.requires_grad and name.startswith("backbone."))
    assert count_l == count_f + lora_count
    assert sum(isinstance(layer.attention.q_proj, LoRALinear) for layer in model.backbone.encoder.layer) == 4
    pixel = torch.randn(1, 3, 224, 224, device="cuda")
    start = time.monotonic()
    with torch.autocast("cuda", dtype=torch.float16):
        output = model(pixel, args[1], args[3], args[4], args[5])
        loss = model.head.loss(output, torch.tensor([0], device="cuda"),
                               torch.tensor([0], device="cuda"), torch.tensor([1400.], device="cuda"))
    loss.backward()
    assert torch.isfinite(loss)
    assert all(layer.attention.q_proj.down.weight.grad is not None for layer in model.backbone.encoder.layer[-4:])
    elapsed_l = time.monotonic() - start
    result = {"device": "cuda", "wc14_style_classes": int(style_classes), "wc14_fine_classes": int(fine_classes),
              "chronoarch_f_trainable_parameters": int(count_f),
              "chronoarch_l_trainable_parameters": int(count_l), "lora_trainable_parameters": int(lora_count),
              "f_forward_backward_ok": True, "l_forward_backward_ok": True,
              "f_forward_backward_seconds": elapsed_f, "l_forward_backward_seconds": elapsed_l,
              "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
              "peak_reserved_bytes": torch.cuda.max_memory_reserved()}
    return result


if __name__ == "__main__":
    result = synthetic_tests()
    if "--real" in sys.argv:
        result.update(real_tests())
    target = ROOT / "results_v2" / "b2_unit_test.json"
    target.parent.mkdir(exist_ok=True)
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
