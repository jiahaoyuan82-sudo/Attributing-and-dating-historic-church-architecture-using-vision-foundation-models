"""Extract the MD5-verified WikiChurches image archive."""
import zipfile
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from pathing import ROOT, RAW


def main():
    with zipfile.ZipFile(RAW / "images.zip") as archive:
        entries = archive.infolist()
        print(f"Archive entries: {len(entries)}", flush=True)
        for entry in entries:
            target = (RAW / entry.filename).resolve()
            if not target.is_relative_to(RAW.resolve()):
                raise RuntimeError(f"Unsafe archive path: {entry.filename}")
        archive.extractall(RAW)
    images = list((RAW / "images").rglob("*.jpg"))
    print(f"Extracted JPEG files: {len(images)}", flush=True)
    if not images:
        raise RuntimeError("No JPEG files extracted under data/raw/images")


if __name__ == "__main__":
    main()
