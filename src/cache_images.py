"""S2 image cache and original-image properties used by S5 controls."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import sys
import re

import pandas as pd
from PIL import Image, ImageOps

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pathing import ROOT, PROCESSED, image_path
OUT = PROCESSED / "img256"
PROPS = PROCESSED / "image_properties.parquet"


def one(row):
    source = image_path(row.image_path)
    target = OUT / row.image_id
    with Image.open(source) as original:
        width, height = original.size
        exif = original.getexif()
        raw_date = exif.get(36867) or exif.get(306) or exif.get(36868)
        match = re.match(r"^([12]\d{3})", str(raw_date)) if raw_date is not None else None
        exif_year = int(match.group(1)) if match else None
        if not target.exists():
            image = ImageOps.exif_transpose(original).convert("RGB")
            w, h = image.size
            scale = 256 / min(w, h)
            image = image.resize((max(1, round(w * scale)), max(1, round(h * scale))), Image.Resampling.LANCZOS)
            temporary = target.with_suffix(".tmp")
            image.save(temporary, format="JPEG", quality=90)
            temporary.replace(target)
    return {"image_id": row.image_id, "original_width": width, "original_height": height,
            "original_file_bytes": source.stat().st_size, "exif_year": exif_year,
            "cache_file_bytes": target.stat().st_size}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    metadata = pd.read_parquet(PROCESSED / "metadata.parquet")
    rows = []
    with ThreadPoolExecutor(max_workers=8) as pool:
        for index, result in enumerate(pool.map(one, metadata.itertuples(index=False)), 1):
            rows.append(result)
            if index % 500 == 0:
                pd.DataFrame(rows).to_parquet(PROPS, index=False)
                print(f"cached {index}/{len(metadata)}", flush=True)
    pd.DataFrame(rows).to_parquet(PROPS, index=False)
    print(f"cached {len(rows)} images; metadata rows {len(rows)}", flush=True)


if __name__ == "__main__":
    main()
