"""S2 frozen image and zero-shot text feature extraction."""
import argparse
import json
import time
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent))
from models import FrozenModel, ROOT
from pathing import RAW, PROCESSED

FEATURES = ROOT / "features"
TEMPLATES = [
    "a photo of a {style} church",
    "a church built in {style} style",
    "{style} architecture",
    "the facade of a {style} church",
    "a historic {style} church building",
]


class ImageDataset(Dataset):
    def __init__(self, metadata, transform):
        self.ids = metadata.image_id.tolist()
        self.paths = [PROCESSED / "img256" / image_id for image_id in self.ids]
        self.transform = transform

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        with Image.open(self.paths[index]) as image:
            return self.transform(image.convert("RGB")), self.ids[index]


def usable_batch(model, dataset, requested=64):
    size = requested
    while size >= 1:
        try:
            batch = torch.stack([dataset[index][0] for index in range(size)])
            output = model.image_features(batch)
            assert output.ndim == 2 and output.shape[0] == size
            return size, output.shape[1]
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            size //= 2
    raise RuntimeError("Model cannot process one image on the CUDA device")


def save_text_features(model, metadata):
    names = {}
    for line in (RAW / "style_names.txt").read_text(encoding="utf-8").splitlines():
        qid, name = line.split(" ", 1)
        names[qid] = name
    for task in ("wc14", "wc6"):
        style_ids = sorted(metadata[f"{task}_style_id"].dropna().unique())
        prompts = [template.format(style=names[style_id].lower()) for style_id in style_ids for template in TEMPLATES]
        vectors = model.text_features(prompts).numpy().reshape(len(style_ids), len(TEMPLATES), -1).mean(axis=1)
        vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
        np.save(FEATURES / f"{model.name}_text_{task}.npy", vectors.astype(np.float32))
        (FEATURES / f"{model.name}_text_{task}_styles.json").write_text(json.dumps(style_ids, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=["R50", "DINO-B", "CLIP-L", "SIG2-B"])
    args = parser.parse_args()
    start = time.time()
    torch.set_num_threads(4)
    if not torch.cuda.is_available():
        raise RuntimeError("NVIDIA CUDA GPU is required for S2 feature extraction")
    FEATURES.mkdir(exist_ok=True)
    metadata = pd.read_parquet(PROCESSED / "metadata.parquet")
    feature_file = FEATURES / f"{args.model}.npy"
    ids_file = FEATURES / f"{args.model}_image_ids.csv"
    if feature_file.exists() and ids_file.exists():
        existing_ids = pd.read_csv(ids_file).image_id.tolist()
        if existing_ids == metadata.image_id.tolist() and np.load(feature_file, mmap_mode="r").shape[0] == len(metadata):
            print(f"Verified existing {args.model} features; skip", flush=True)
            return
    model = FrozenModel(args.model, device="cuda")
    dataset = ImageDataset(metadata, model.transform)
    batch_size, dimension = usable_batch(model, dataset, requested=64)
    print(f"{args.model}: device={model.device}, batch={batch_size}, dim={dimension}", flush=True)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True, persistent_workers=True)
    vectors = np.empty((len(dataset), dimension), dtype=np.float32)
    seen_ids = []
    offset = 0
    for batch_index, (images, image_ids) in enumerate(loader, 1):
        features = model.image_features(images).numpy()
        vectors[offset:offset + len(features)] = features
        seen_ids.extend(image_ids)
        offset += len(features)
        if batch_index % 20 == 0 or offset == len(dataset):
            print(f"{args.model}: {offset}/{len(dataset)}", flush=True)
    assert seen_ids == metadata.image_id.tolist()
    assert np.isfinite(vectors).all()
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-3)
    np.save(feature_file, vectors)
    pd.DataFrame({"image_id": seen_ids}).to_csv(ids_file, index=False)
    if args.model in ("CLIP-L", "SIG2-B"):
        save_text_features(model, metadata)
    (FEATURES / f"{args.model}_extraction.json").write_text(json.dumps({"model": args.model, "device": model.device, "batch_size": batch_size, "num_workers": 8, "feature_dimension": dimension, "image_count": len(dataset), "elapsed_seconds": round(time.time() - start, 2)}, indent=2), encoding="utf-8")
    print(f"{args.model}: completed in {time.time() - start:.1f}s", flush=True)


if __name__ == "__main__":
    main()
