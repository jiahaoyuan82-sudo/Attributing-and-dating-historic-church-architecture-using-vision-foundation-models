"""Prepare WikiChurches metadata, group splits, duplicate audit and statistics."""
import csv
import hashlib
import json
import math
import zipfile
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import imagehash
import numpy as np
import pandas as pd
from PIL import Image

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from pathing import ROOT, RAW, PROCESSED, image_path
SPLITS = ROOT / "splits"
RESULTS = ROOT / "results"
SEED = 0


def read_labels(archive):
    out = {}
    for member in archive.namelist():
        if member.endswith(".txt"):
            name = Path(member).stem
            lines = [line.split() for line in archive.read(member).decode("utf-8").splitlines() if line.strip()]
            assert all(len(line) == 2 for line in lines), name
            assert len(lines) == len(set(line[0] for line in lines)), name
            out[name] = dict(lines)
    return out


def load_styles():
    names = {}
    for line in (RAW / "style_names.txt").read_text(encoding="utf-8").splitlines():
        qid, name = line.split(" ", 1)
        names[qid] = name
    parents = {}
    for line in (RAW / "parent_child_rel.txt").read_text(encoding="utf-8").splitlines():
        parent, child = line.split()
        parents[child] = parent
    return names, parents


def ancestry(qid, parents):
    path = [qid]
    while qid in parents:
        qid = parents[qid]
        assert qid not in path, "Cycle in style hierarchy"
        path.insert(0, qid)
    return path


def split_wc14(labels):
    by_class = defaultdict(set)
    for image_id, style in labels["wc14"].items():
        by_class[style].add(image_id.split("_", 1)[0])
    rng = np.random.default_rng(SEED)
    church_split = {}
    for style, churches in sorted(by_class.items()):
        items = np.array(sorted(churches))
        rng.shuffle(items)
        n_test = max(5, math.ceil(0.25 * len(items)))
        n_val = max(1, round(0.10 * len(items)))
        assert n_test + n_val < len(items), style
        for church, split in [(x, "test") for x in items[:n_test]] + [(x, "val") for x in items[n_test:n_test+n_val]] + [(x, "train") for x in items[n_test+n_val:]]:
            assert church not in church_split
            church_split[church] = split
    return church_split


def make_phashes(frame):
    path = PROCESSED / "phash.csv"
    existing = {}
    if path.exists():
        existing = dict(pd.read_csv(path, dtype=str)[["image_id", "phash"]].itertuples(index=False, name=None))
    pending = [(row.image_id, image_path(row.image_path)) for row in frame[frame["wc14_style_id"].notna()].itertuples(index=False) if row.image_id not in existing]
    def hash_one(item):
        image_id, image_path = item
        with Image.open(image_path) as image:
            return image_id, str(imagehash.phash(image.convert("RGB")))
    with ThreadPoolExecutor(max_workers=8) as pool:
      for index, (image_id, value) in enumerate(pool.map(hash_one, pending), 1):
        existing[image_id] = value
        if index % 500 == 0:
            pd.DataFrame(sorted(existing.items()), columns=["image_id", "phash"]).to_csv(path, index=False)
            print(f"pHash {index}/{len(pending)}", flush=True)
    pd.DataFrame(sorted(existing.items()), columns=["image_id", "phash"]).to_csv(path, index=False)
    return {key: int(value, 16) for key, value in existing.items()}


def duplicate_audit(frame, hashes):
    # Five disjoint chunks guarantee a matching chunk for Hamming distance <= 4.
    buckets = defaultdict(set)
    train = frame.loc[frame["split_wc14"] == "train", "image_id"]
    test = frame.loc[frame["split_wc14"] == "test", "image_id"]
    widths = [13, 13, 13, 13, 12]
    offsets = np.cumsum([0] + widths[:-1]).tolist()
    def keys(value):
        return [(i, (value >> offset) & ((1 << width) - 1)) for i, (offset, width) in enumerate(zip(offsets, widths))]
    for image_id in train:
        for key in keys(hashes[image_id]):
            buckets[key].add(image_id)
    pairs = []
    excluded = set()
    for image_id in test:
        candidates = set().union(*(buckets[key] for key in keys(hashes[image_id])))
        for train_id in candidates:
            distance = (hashes[image_id] ^ hashes[train_id]).bit_count()
            if distance <= 4:
                pairs.append((image_id, train_id, distance))
                excluded.add(image_id)
    pd.DataFrame(sorted(pairs), columns=["test_image_id", "train_image_id", "hamming_distance"]).to_csv(PROCESSED / "near_duplicate_pairs.csv", index=False)
    frame.loc[frame["image_id"].isin(excluded), "split_wc14"] = "excluded_duplicate"
    return len(excluded), len(pairs)


def stats(frame):
    rows = []
    def add(section, subset, style, country, part):
        known = int(part["construction_year"].notna().sum())
        rows.append({"section": section, "subset": subset, "style": style or "", "country": country or "", "images": len(part), "churches": part["church_id"].nunique(), "year_known_images": known, "year_coverage_pct": round(100 * known / len(part), 4) if len(part) else 0})
    subsets = {
        "all": frame,
        "wc14": frame[frame["wc14_style_id"].notna()],
        "wc14_train": frame[frame["split_wc14"] == "train"],
        "wc14_val": frame[frame["split_wc14"] == "val"],
        "wc14_test": frame[frame["split_wc14"] == "test"],
        "wc6": frame[frame["wc6_style_id"].notna()],
        "wc6_train": frame[frame["split_wc6"] == "train"],
        "wc6_val": frame[frame["split_wc6"] == "val"],
        "wc6_test": frame[frame["split_wc6"] == "test"],
        "wc4": frame[frame["wc4_style_id"].notna()],
        "wc4_train": frame[frame["split_wc4"] == "train"],
        "wc4_val": frame[frame["split_wc4"] == "val"],
        "wc4_test": frame[frame["split_wc4"] == "test"],
    }
    for subset, part in subsets.items():
        add("subset_total", subset, None, None, part)
        style_col = "wc14_style_id" if subset.startswith("wc14") else "wc6_style_id" if subset.startswith("wc6") else "wc4_style_id" if subset.startswith("wc4") else "level1_style"
        for style, group in part.groupby(style_col, dropna=False):
            label = group["level1_style"].iloc[0] if style_col != "level1_style" else style
            add("class", subset, label if pd.notna(label) else "(unassigned_or_multilabel)", None, group)
    for country, part in frame.groupby("country", dropna=False):
        add("country", "all", None, country if pd.notna(country) else "Unknown", part)
    rows = [r for r in rows if r["section"] != "country"] + sorted((r for r in rows if r["section"] == "country"), key=lambda x: -x["images"])[:10]
    pd.DataFrame(rows).to_csv(RESULTS / "dataset_stats.csv", index=False, float_format="%.4f")


def main():
    PROCESSED.mkdir(parents=True, exist_ok=True)
    SPLITS.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    churches = json.loads((RAW / "churches.json").read_text(encoding="utf-8"))
    image_meta = json.loads((RAW / "image_meta.json").read_text(encoding="utf-8"))
    cache_path = PROCESSED / "wikidata_cache.json"
    cache = json.loads(cache_path.read_text(encoding="utf-8"))["records"] if cache_path.exists() else {}
    names, parents = load_styles()
    with zipfile.ZipFile(RAW / "labels.zip") as z:
        labels = read_labels(z)
    images = sorted((RAW / "images").rglob("*.jpg"))
    assert images, "Extract images.zip first"
    image_paths = {path.name: (Path("data/raw/images") / path.name).as_posix() for path in images}
    assert len(image_paths) == len(images), "Duplicate image filenames"
    orphan_images = sorted(image_id for image_id in image_paths if image_id.split("_", 1)[0] not in churches)
    missing_labels = set(labels["wc14"]) - set(image_paths)
    assert not missing_labels, f"Missing WC14 images: {len(missing_labels)}"
    own_styles = {qid: set(c["styles"]) for qid, c in churches.items()}
    church_split = split_wc14(labels)
    rows = []
    for image_id, image_path in image_paths.items():
        qid = image_id.split("_", 1)[0]
        if qid not in churches:
            continue
        church = churches[qid]
        own_year = church.get("built_year")
        supplement = cache.get(qid, {}).get("year")
        if isinstance(own_year, int):
            year, year_source = own_year, "churches.json"
        elif isinstance(own_year, list):
            year, year_source = None, "churches.json ambiguous multiple years"
        else:
            year = supplement
            year_source = "Wikidata P571" if supplement is not None else None
        wc14_id = labels["wc14"].get(image_id)
        paths = [ancestry(style, parents) for style in own_styles[qid]]
        top = wc14_id or (paths[0][0] if len({p[0] for p in paths}) == 1 else None)
        level2 = {p[1] for p in paths if len(p) > 1 and p[0] == top}
        rows.append({
            "image_id": image_id, "church_id": qid, "image_path": image_path,
            "level1_style": names.get(top), "level2_style": names.get(next(iter(level2))) if len(level2) == 1 else None,
            "all_styles": json.dumps(sorted(own_styles[qid])), "country": church["country"]["name"] if church.get("country") else None,
            "construction_year": year, "construction_year_raw": json.dumps(own_year), "year_source": year_source,
            "split_wc6": next((split for split in ("train", "val", "test") if image_id in labels[f"wc6_{split}"]), None),
            "split_wc4": next((split for split in ("train", "val", "test") if image_id in labels[f"wc4_{split}"]), None),
            "split_wc14": church_split.get(qid) if wc14_id else None,
            "wc14_style_id": wc14_id, "wc6_style_id": labels["wc6"].get(image_id), "wc4_style_id": labels["wc4"].get(image_id),
            "image_metadata_present": image_id in image_meta,
        })
    frame = pd.DataFrame(rows)
    assert frame["image_id"].is_unique
    hashes = make_phashes(frame)
    excluded, pairs = duplicate_audit(frame, hashes)
    frame.to_parquet(PROCESSED / "metadata.parquet", index=False)
    frame.loc[frame["wc14_style_id"].notna(), ["image_id", "church_id", "wc14_style_id", "split_wc14"]].to_csv(SPLITS / "wc14.csv", index=False)
    for name in ("wc14", "wc6", "wc4"):
        subset = frame[frame[f"wc{14 if name == 'wc14' else 6 if name == 'wc6' else 4}_style_id"].notna()]
        split_col = f"split_{name}"
        assert subset.groupby("church_id")[split_col].nunique().max() == 1, f"{name} church leakage"
    stats(frame)
    report = {"wc14_test_images_excluded_near_duplicate": excluded, "wc14_test_train_near_duplicate_pairs": pairs,
              "phash_hamming_threshold": 4, "images_in_archive": len(image_paths), "images_with_church_metadata": len(frame),
              "archive_images_without_church_metadata": orphan_images,
              "wc14_images": int(frame["wc14_style_id"].notna().sum()),
              "archive_images_without_image_meta": sorted(set(image_paths) - set(image_meta)),
              "image_meta_keys_without_archive": sorted(set(image_meta) - set(image_paths))}
    (PROCESSED / "dedup_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
