"""B5 validation-only gate summaries and descriptive attention maps."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from chronoarch_tasks import task_folds
from chronoarch_train import FeatureStore, TrainConfig, make_model
from pathing import PROCESSED
from run_chronoarch_v2 import RESULTS, stored_row


def main():
    assert torch.cuda.is_available()
    metadata = pd.read_parquet(PROCESSED / "metadata.parquet")
    store = FeatureStore()
    rows = []
    examples = []
    for task in ("wc6", "wc14"):
        fold = task_folds(metadata, task)[0]
        chosen = json.loads((RESULTS / f"selected_{task}_0.json").read_text(encoding="utf-8"))
        config = TrainConfig(**chosen["final"])
        for seed in range(3):
            record = stored_row("ChronoArch-F", fold, config, seed)
            assert record is not None
            model = make_model(fold, config)
            saved = torch.load(record["checkpoint"], map_location="cpu", weights_only=False)
            model.load_state_dict(saved["state_dict"])
            model.eval()
            gates = []
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
                for start in range(0, len(fold.val), 64):
                    part = fold.val[start:start + 64]
                    output = model(*store.batch(part, "base"))
                    weight = output["gates"].float().mean(dim=-1)
                    assert torch.allclose(weight.sum(-1), torch.ones(len(part), device="cuda"), atol=.01)
                    gates.append(weight.cpu().numpy())
                if seed == 0:
                    for position in fold.val[:3]:
                        result = model(*store.batch(np.array([position]), "base"),
                                       return_attention=True)
                        attention = result["attention"][-1][0].float().mean(dim=0).cpu().numpy()
                        examples.append((task, fold.image_ids[position],
                                         attention[:64].reshape(8, 8),
                                         attention[64:].reshape(8, 8)))
            values = np.concatenate(gates)
            for index, style in enumerate(fold.style_names):
                relevant = fold.style[fold.val] == index
                mean = values[relevant].mean(axis=0)
                rows.append({"task": task, "seed": seed, "style_id": style,
                             "images": int(relevant.sum()), "g_dino": mean[0],
                             "g_clip": mean[1], "g_siglip": mean[2],
                             "split": "validation"})
            print(f"Gate summary {task} seed {seed}", flush=True)
    pd.DataFrame(rows).to_csv(RESULTS / "gate_by_style.csv", index=False)
    figure, axes = plt.subplots(len(examples), 2, figsize=(8, 3 * len(examples)),
                                squeeze=False)
    for axis_pair, (task, image_id, low, high) in zip(axes, examples):
        for axis, value, scale in zip(axis_pair, (low, high), (224, 448)):
            image = axis.imshow(value, cmap="magma")
            axis.set_title(f"{task} {image_id} · {scale}px")
            axis.axis("off")
            figure.colorbar(image, ax=axis, fraction=.045)
    figure.suptitle("Descriptive query attention on validation images only")
    figure.tight_layout()
    target = ROOT / "figures_v2" / "attention_examples.pdf"
    figure.savefig(target, bbox_inches="tight")
    plt.close(figure)
    print(target, flush=True)


if __name__ == "__main__":
    main()
