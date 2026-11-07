"""Read-only row-by-row integrity audit of v2 feature arrays and result CSVs.

Only after all checks finish, rename incomplete files with .incomplete suffix.
This script performs no model forward or experiment computation.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pathing import ROOT, PROCESSED, RAW

FEATURES = ROOT / "features_v2"
RESULTS = ROOT / "results_v2"


def expected_arrays():
    items = {}
    for model, size, width in (("DINO-L", 224, 1024), ("DINO-L", 448, 1024), ("DINO-B", 448, 768)):
        for view in ("base", "flip", "crop0", "crop1"):
            stem = f"{model}_{size}_{view}"
            items[f"{stem}_tokens.npy"] = (9485, 256, width, f"{stem}_progress.json")
            items[f"{stem}_cls.npy"] = (9485, width, f"{stem}_progress.json")
        for view in ("base", "flip"):
            stem = f"{model}_{size}_{view}_rq4"
            items[f"{stem}_tokens.npy"] = (5838, 256, width, f"{stem}_progress.json")
            items[f"{stem}_cls.npy"] = (5838, width, f"{stem}_progress.json")
    for model in ("CLIP-L", "SIG2-B"):
        items[f"{model}_flip.npy"] = (9485, 768, f"{model}_flip_progress.json")
        for view in ("base", "flip"):
            items[f"{model}_rq4_{view}.npy"] = (5838, 768, f"{model}_rq4_{view}_progress.json")
    return items


def audit_array(path, expected):
    expected_shape, progress_name = expected[:-1], expected[-1]
    row = {"file": path.name, "expected_rows": expected_shape[0], "actual_rows": None,
           "dtype": None, "zero_rows": None, "nan_rows": None, "inf_rows": None,
           "progress_completed": None, "index": "rq4_index.csv" if expected_shape[0] == 5838 else "image_ids.csv",
           "complete": False, "reason": ""}
    if not path.exists():
        row["reason"] = "file missing"
        return row
    progress_path = FEATURES / progress_name
    if progress_path.exists():
        try:
            progress = json.loads(progress_path.read_text(encoding="utf-8"))
            row["progress_completed"] = progress.get("completed")
            if progress.get("device") != "cuda":
                row["reason"] += "progress device is not cuda; "
        except Exception as error:
            row["reason"] += f"progress unreadable: {error}; "
    else:
        row["reason"] += "progress missing; "
    try:
        array = np.load(path, mmap_mode="r", allow_pickle=False)
        row["actual_rows"] = array.shape[0]
        row["dtype"] = str(array.dtype)
        if array.shape != expected_shape:
            row["reason"] += f"shape {array.shape} != {expected_shape}; "
        if array.dtype != np.float16:
            row["reason"] += "dtype is not float16; "
        zeros = nan = inf = 0
        for start in range(0, len(array), 64):
            block = np.asarray(array[start:start + 64])
            axes = tuple(range(1, block.ndim))
            zeros += int(np.all(block == 0, axis=axes).sum())
            nan += int(np.any(np.isnan(block), axis=axes).sum())
            inf += int(np.any(np.isinf(block), axis=axes).sum())
        row["zero_rows"], row["nan_rows"], row["inf_rows"] = zeros, nan, inf
        if zeros or nan or inf:
            row["reason"] += "zero/NaN/Inf row present; "
    except Exception as error:
        row["reason"] += f"array unreadable: {error}; "
    if row["progress_completed"] != expected_shape[0]:
        row["reason"] += "progress count mismatch; "
    row["complete"] = not row["reason"]
    print(f"AUDITED {path.name}: rows={row['actual_rows']} zero={row['zero_rows']} nan={row['nan_rows']} inf={row['inf_rows']} complete={row['complete']}", flush=True)
    return row


def index_checks():
    metadata = pd.read_parquet(PROCESSED / "metadata.parquet")
    ids = pd.read_csv(FEATURES / "image_ids.csv")
    original_ids = pd.read_csv(ROOT / "features" / "DINO-B_image_ids.csv")
    assert len(metadata) == len(ids) == len(original_ids) == 9485
    assert ids.image_id.is_unique
    assert ids.image_id.tolist() == metadata.image_id.tolist() == original_ids.image_id.tolist()
    assert ids.church_id.tolist() == metadata.church_id.tolist()
    annotations = json.loads((RAW / "building_parts.json").read_text(encoding="utf-8"))["annotations"]
    rq4 = pd.read_csv(FEATURES / "rq4_index.csv")
    assert len(rq4) == 5838 and rq4.image_id.nunique() == len(annotations) == 139
    assert list(dict.fromkeys(rq4.image_id)) == list(annotations)
    assert set(rq4.image_id).issubset(set(ids.image_id))
    for image_id, group in rq4.groupby("image_id", sort=False):
        assert len(group) == 42
        expected_conditions = ["E", "K"] + [value for _ in range(20) for value in ("R", "R_keep")]
        assert group.condition.tolist() == expected_conditions
        for draw in range(20):
            pair = group[(group.draw == draw) & group.condition.isin(("R", "R_keep"))]
            assert len(pair) == 2 and pair.boxes.iloc[0] == pair.boxes.iloc[1]
    return {"image_ids": len(ids), "metadata_order_match": True,
            "original_feature_index_match": True, "rq4_rows": len(rq4),
            "rq4_image_ids": len(annotations), "rq4_order_match": True}


def csv_checks():
    rows = []
    for path in sorted(RESULTS.glob("*.csv")):
        row = {"file": path.name, "rows": None, "columns": None,
               "complete": False, "reason": ""}
        try:
            frame = pd.read_csv(path)
            row["rows"], row["columns"] = frame.shape
            if frame.empty:
                row["reason"] = "empty CSV"
            elif frame.columns.duplicated().any():
                row["reason"] = "duplicate columns"
            else:
                row["complete"] = True
        except Exception as error:
            row["reason"] = str(error)
        rows.append(row)
    return rows


def main():
    indexes = index_checks()
    expected = expected_arrays()
    actual = {path.name for path in FEATURES.glob("*.npy")}
    assert actual == set(expected), {"unexpected": sorted(actual - set(expected)),
                                    "missing": sorted(set(expected) - actual)}
    arrays = [audit_array(FEATURES / name, spec) for name, spec in sorted(expected.items())]
    csvs = csv_checks()
    test_path = RESULTS / "b2_unit_test.json"
    b2 = json.loads(test_path.read_text(encoding="utf-8"))
    b2_ok = all(b2.get(name) is True for name in ("synthetic_f_backward", "f_forward_backward_ok", "l_forward_backward_ok")) and b2.get("device") == "cuda"
    # Preserve invalid artifacts; only rename after every file has been inspected.
    renamed = []
    for row in arrays:
        if not row["complete"] and (FEATURES / row["file"]).exists():
            source = FEATURES / row["file"]
            target = source.with_name(source.name + ".incomplete")
            assert not target.exists(), target
            source.rename(target)
            renamed.append(target.name)
    for row in csvs:
        if not row["complete"]:
            source = RESULTS / row["file"]
            target = source.with_name(source.name + ".incomplete")
            assert not target.exists(), target
            source.rename(target)
            renamed.append(target.name)
    report = {"indexes": indexes, "feature_files": len(arrays),
              "feature_failures": sum(not row["complete"] for row in arrays),
              "result_csv_files": len(csvs), "result_csv_failures": sum(not row["complete"] for row in csvs),
              "b2_unit_test_record_ok": b2_ok, "renamed": renamed}
    RESULTS.mkdir(exist_ok=True)
    pd.DataFrame(arrays).to_csv(RESULTS / "b1_integrity_audit.csv", index=False)
    (RESULTS / "b1_integrity_summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
