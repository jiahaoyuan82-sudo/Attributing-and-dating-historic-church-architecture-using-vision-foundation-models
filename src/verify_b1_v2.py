"""Audit completed B1 arrays, indexes, storage, and recorded CUDA peaks."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pathing import ROOT, PROCESSED

FEATURES = ROOT / "features_v2"
RESULTS = ROOT / "results_v2"


def main():
    metadata = pd.read_parquet(PROCESSED / "metadata.parquet")
    image_ids = pd.read_csv(FEATURES / "image_ids.csv")
    assert image_ids.image_id.tolist() == metadata.image_id.tolist()
    expected = []
    for model, size, width in (("DINO-L", 224, 1024), ("DINO-L", 448, 1024), ("DINO-B", 448, 768)):
        for view in ("base", "flip", "crop0", "crop1"):
            expected.append((f"{model}_{size}_{view}", len(metadata), width))
        for view in ("base", "flip"):
            expected.append((f"{model}_{size}_{view}_rq4", 139 * 42, width))
    rows = []
    for stem, count, width in expected:
        token_path = FEATURES / f"{stem}_tokens.npy"
        cls_path = FEATURES / f"{stem}_cls.npy"
        progress_path = FEATURES / f"{stem}_progress.json"
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        assert progress["completed"] == count and progress["device"] == "cuda"
        tokens = np.load(token_path, mmap_mode="r")
        cls = np.load(cls_path, mmap_mode="r")
        assert tokens.dtype == cls.dtype == np.float16
        assert tokens.shape == (count, 256, width) and cls.shape == (count, width)
        for position in (0, count // 2, count - 1):
            assert np.isfinite(tokens[position]).all() and np.isfinite(cls[position]).all()
        rows.extend(({"name": path.name, "bytes": path.stat().st_size, "rows": count,
                      "dtype": "float16", "peak_allocated_bytes": progress["peak_allocated_bytes"],
                      "peak_reserved_bytes": progress.get("peak_reserved_bytes"),
                      "elapsed_seconds": progress["elapsed_seconds"]} for path in (token_path, cls_path)))
    for model in ("CLIP-L", "SIG2-B"):
        for suffix, count in (("flip", len(metadata)), ("rq4_base", 139 * 42), ("rq4_flip", 139 * 42)):
            path = FEATURES / f"{model}_{suffix}.npy"
            progress = json.loads((FEATURES / f"{model}_{suffix}_progress.json").read_text(encoding="utf-8"))
            assert progress["completed"] == count and progress["device"] == "cuda"
            array = np.load(path, mmap_mode="r")
            assert array.dtype == np.float16 and array.shape == (count, 768)
            assert np.isfinite(array[[0, count // 2, count - 1]]).all()
            rows.append({"name": path.name, "bytes": path.stat().st_size, "rows": count,
                         "dtype": "float16", "peak_allocated_bytes": progress["peak_allocated_bytes"],
                         "peak_reserved_bytes": progress.get("peak_reserved_bytes"),
                         "elapsed_seconds": progress["elapsed_seconds"]})
    rq4 = pd.read_csv(FEATURES / "rq4_index.csv")
    assert len(rq4) == 139 * 42
    assert rq4.groupby("image_id").size().eq(42).all()
    manifest = pd.DataFrame(rows)
    manifest.to_csv(RESULTS / "b1_feature_manifest.csv", index=False)
    report = {"image_rows": len(metadata), "rq4_rows": len(rq4), "feature_files": len(manifest),
              "feature_bytes": int(manifest.bytes.sum()),
              "peak_allocated_bytes": int(manifest.peak_allocated_bytes.max()),
              "peak_reserved_bytes": int(manifest.peak_reserved_bytes.max()),
              "all_cuda": True, "all_float16": True}
    (RESULTS / "b1_verification.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
