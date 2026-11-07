"""Full-row integrity check of the R1 224-pixel DINO-B feature extension."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd
from pathing import PROCESSED


def main():
    features = ROOT / "features_v2"
    ids = pd.read_csv(features / "image_ids.csv")
    metadata = pd.read_parquet(PROCESSED / "metadata.parquet")
    assert ids.image_id.tolist() == metadata.image_id.tolist()
    rows = []
    for view in ("base", "flip", "crop0", "crop1"):
        stem = f"DINO-B_224_{view}"
        progress = json.loads((features / f"{stem}_progress.json").read_text(encoding="utf-8"))
        assert progress["completed"] == len(metadata) and progress["device"] == "cuda"
        for suffix, shape in (("tokens", (len(metadata), 256, 768)),
                              ("cls", (len(metadata), 768))):
            path = features / f"{stem}_{suffix}.npy"
            array = np.load(path, mmap_mode="r", allow_pickle=False)
            assert array.shape == shape and array.dtype == np.float16
            zeros = nan = inf = 0
            for start in range(0, len(array), 64):
                block = np.asarray(array[start:start + 64])
                axes = tuple(range(1, block.ndim))
                zeros += int(np.all(block == 0, axis=axes).sum())
                nan += int(np.any(np.isnan(block), axis=axes).sum())
                inf += int(np.any(np.isinf(block), axis=axes).sum())
            row = {"file": path.name, "rows": len(array), "zero_rows": zeros,
                   "nan_rows": nan, "inf_rows": inf, "bytes": path.stat().st_size,
                   "index_matches_metadata": True, "device": progress["device"]}
            assert zeros == nan == inf == 0, row
            rows.append(row)
            print(row, flush=True)
    pd.DataFrame(rows).to_csv(ROOT / "results_v2" / "dinob224_integrity.csv", index=False)


if __name__ == "__main__":
    main()
