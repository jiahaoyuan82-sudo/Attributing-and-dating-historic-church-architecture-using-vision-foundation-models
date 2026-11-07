"""Verify exact Zenodo v1.0 file MD5 values."""
import datetime as dt
import hashlib
import json
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from pathing import ROOT, RAW
EXPECTED = {
    "building_parts.json": "47d678ae39ee7b63e6d64b18d87fb5e1",
    "churches.json": "7461f4752451f785d1d9340f79773393",
    "datasheet.pdf": "d4dd46daad863bf874ba07aa74875638",
    "image_meta.json": "ccf8749c6b4dd4aa5fbacf1170da7a95",
    "images.zip": "510367ec6dbac77d40f77c58673a1026",
    "labels.zip": "0f4bd9aa922c1e965c4a3fb1b02b0b31",
    "LICENSE": "4ca2fb9d5a10b1ce06910aef6a92e86e",
    "parent_child_rel.txt": "0dd9e676d350c0ca2193b31f9f9061f6",
    "README.md": "1054a7271a65bbeea5d82e0dbd2b165b",
    "style_names.txt": "d24dde41f3512229cd99cac47ee8d564",
}


def main():
    report = {"source": "https://zenodo.org/records/5166987", "checked_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "files": {}}
    for name, expected in EXPECTED.items():
        path = RAW / name
        if not path.exists():
            report["files"][name] = {"present": False, "expected_md5": expected, "valid": False}
            continue
        digest = hashlib.md5()
        with path.open("rb") as source:
            while chunk := source.read(16 * 1024 * 1024):
                digest.update(chunk)
        actual = digest.hexdigest()
        report["files"][name] = {"present": True, "size_bytes": path.stat().st_size, "expected_md5": expected, "actual_md5": actual, "valid": actual == expected}
        print(name, "OK" if actual == expected else "MISMATCH", flush=True)
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results/download_checksums.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    if not all(row["valid"] for row in report["files"].values()):
        raise SystemExit("Missing or mismatched download")


if __name__ == "__main__":
    main()
