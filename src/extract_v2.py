"""Resumable fp16 feature extraction for Part B1; all model forwards require CUDA."""
import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
os.environ["HF_HOME"] = str(ROOT / "env" / "hf_cache")
os.environ["TORCH_HOME"] = str(ROOT / "env" / "torch_cache")
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageOps, ImageDraw, ImageFilter
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms as T
from torchvision.transforms import InterpolationMode
from transformers import AutoImageProcessor, AutoModel

from models import FrozenModel
from pathing import RAW, PROCESSED, image_path
from rq4 import boxes_for, occlude, pixel_box, random_boxes

OUT = ROOT / "features_v2"
SEED = 0
VIEWS = ("base", "flip", "crop0", "crop1")


class LocalClip:
    """Load the already verified OpenAI ViT-L/14 safetensors without a Hub request."""
    def __init__(self):
        import open_clip
        cache = ROOT / "env" / "hf_cache" / "hub" / "models--timm--vit_large_patch14_clip_224.openai" / "snapshots"
        weights = list(cache.glob("*/open_clip_model.safetensors"))
        assert len(weights) == 1, "Existing CLIP-L weight file not found"
        weight_path = "\\\\?\\" + str(weights[0].resolve()) if os.name == "nt" else str(weights[0])
        self.model, _, self.transform = open_clip.create_model_and_transforms(
            "ViT-L-14", pretrained=weight_path, force_quick_gelu=True)
        self.model = self.model.eval().to("cuda")
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    @torch.inference_mode()
    def image_features(self, images):
        with torch.autocast("cuda", dtype=torch.float16):
            vectors = self.model.encode_image(images.to("cuda"))
        return torch.nn.functional.normalize(vectors.float(), dim=-1).cpu()


def save_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def resize_crop(image, size, mean, std, view, image_id):
    if view.startswith("crop"):
        digest = hashlib.sha256(f"{SEED}:{image_id}:{view}".encode()).digest()
        generator = torch.Generator().manual_seed(int.from_bytes(digest[:8], "little"))
        # RandomResizedCrop.get_params uses the global torch RNG; isolate its state.
        with torch.random.fork_rng(devices=[]):
            torch.set_rng_state(generator.get_state())
            top, left, height, width = T.RandomResizedCrop.get_params(image, (0.6, 1.0), (0.75, 4 / 3))
        image = T.functional.resized_crop(image, top, left, height, width,
                                          (size, size), InterpolationMode.BICUBIC)
    else:
        if view == "flip":
            image = ImageOps.mirror(image)
        image = T.Resize(size, interpolation=InterpolationMode.BICUBIC)(image)
        image = T.CenterCrop(size)(image)
    return T.Normalize(mean, std)(T.ToTensor()(image))


class ImageViewDataset(Dataset):
    def __init__(self, metadata, size, view, mean, std):
        self.ids = metadata.image_id.tolist()
        source_raw = size == 448 or view.startswith("crop")
        self.paths = [image_path(path) if source_raw else PROCESSED / "img256" / image_id
                      for image_id, path in zip(self.ids, metadata.image_path)]
        self.size, self.view, self.mean, self.std = size, view, mean, std

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        with Image.open(self.paths[index]) as opened:
            image = ImageOps.exif_transpose(opened).convert("RGB")
            return resize_crop(image, self.size, self.mean, self.std,
                               self.view, self.ids[index])


class Dino:
    def __init__(self, name):
        assert torch.cuda.is_available(), "CUDA required"
        self.name = name
        identifier = str(ROOT / "env" / "dinov2-large") if name == "DINO-L" else "facebook/dinov2-base"
        self.processor = AutoImageProcessor.from_pretrained(identifier, local_files_only=True)
        self.model = AutoModel.from_pretrained(identifier, local_files_only=True).eval().to("cuda")
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.width = self.model.config.hidden_size
        self.mean, self.std = self.processor.image_mean, self.processor.image_std

    @torch.inference_mode()
    def features(self, images, size):
        batch = images.to("cuda", non_blocking=True)
        with torch.autocast("cuda", dtype=torch.float16):
            hidden = self.model(pixel_values=batch).last_hidden_state
        cls = hidden[:, 0].to(dtype=torch.float16)
        patches = hidden[:, 1:]
        expected = size // 14
        assert patches.shape[1] == expected ** 2, (size, patches.shape)
        if size == 448:
            patches = patches.reshape(len(images), expected, expected, self.width)
            patches = patches.reshape(len(images), 16, 2, 16, 2, self.width).mean(dim=(2, 4))
            patches = patches.reshape(len(images), 256, self.width)
        assert patches.shape[1:3] == (256, self.width)
        return patches.to(dtype=torch.float16).cpu().numpy(), cls.cpu().numpy()


def paths_for(model_name, size, view, occlusion=False):
    stem = f"{model_name}_{size}_{view}" + ("_rq4" if occlusion else "")
    return OUT / f"{stem}_tokens.npy", OUT / f"{stem}_cls.npy", OUT / f"{stem}_progress.json"


def open_output(path, shape, completed):
    if path.exists():
        array = np.load(path, mmap_mode="r+")
        assert array.shape == shape and array.dtype == np.float16
        return array
    assert completed == 0, "Progress exists but output file missing"
    return np.lib.format.open_memmap(path, mode="w+", dtype=np.float16, shape=shape)


def extract_dino_view(model, metadata, size, view, batch_size):
    assert view in VIEWS
    n = len(metadata)
    token_path, cls_path, progress_path = paths_for(model.name, size, view)
    progress = json.loads(progress_path.read_text(encoding="utf-8")) if progress_path.exists() else {"completed": 0, "peak_allocated_bytes": 0, "elapsed_seconds": 0.0}
    completed = progress["completed"]
    tokens = open_output(token_path, (n, 256, model.width), completed)
    cls = open_output(cls_path, (n, model.width), completed)
    dataset = ImageViewDataset(metadata, size, view, model.mean, model.std)
    start = time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    while completed < n:
        remaining = Subset(dataset, range(completed, n))
        loader = DataLoader(remaining, batch_size=batch_size, num_workers=2,
                            pin_memory=True, persistent_workers=True)
        out_of_memory = False
        for prepared in loader:
            step = len(prepared)
            try:
                block, head = model.features(prepared, size)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                out_of_memory = True
                break
            tokens[completed:completed + step] = block
            cls[completed:completed + step] = head
            completed += step
            if completed % (batch_size * 10) < batch_size or completed == n:
                tokens.flush(); cls.flush()
                progress.update(completed=completed, batch_size=batch_size, device="cuda",
                                peak_allocated_bytes=max(progress["peak_allocated_bytes"], torch.cuda.max_memory_allocated()),
                                peak_reserved_bytes=max(progress.get("peak_reserved_bytes", 0), torch.cuda.max_memory_reserved()),
                                elapsed_seconds=progress["elapsed_seconds"] + time.monotonic() - start)
                save_json(progress_path, progress)
                start = time.monotonic()
                print(f"{model.name} {size} {view}: {completed}/{n}", flush=True)
        del loader
        if out_of_memory:
            if batch_size == 1:
                raise RuntimeError("DINO extraction OOM at batch 1")
            batch_size = max(1, batch_size // 2)
            print(f"{model.name} {size} {view}: OOM, batch reduced to {batch_size}", flush=True)
    return progress


def extract_global_flip(name, metadata, batch_size=32):
    n = len(metadata)
    output = OUT / f"{name}_flip.npy"
    progress_path = OUT / f"{name}_flip_progress.json"
    progress = json.loads(progress_path.read_text(encoding="utf-8")) if progress_path.exists() else {"completed": 0, "peak_allocated_bytes": 0, "elapsed_seconds": 0.0}
    model = LocalClip() if name == "CLIP-L" else FrozenModel(name, device="cuda")
    dimension = 768
    array = open_output(output, (n, dimension), progress["completed"])
    completed = progress["completed"]
    start = time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    while completed < n:
        step = min(batch_size, n - completed)
        images = []
        for image_id in metadata.image_id.iloc[completed:completed + step]:
            with Image.open(PROCESSED / "img256" / image_id) as opened:
                images.append(model.transform(ImageOps.mirror(opened.convert("RGB"))))
        try:
            vectors = model.image_features(torch.stack(images)).numpy()
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if batch_size == 1: raise
            batch_size = max(1, batch_size // 2)
            print(f"{name} flip: OOM, batch reduced to {batch_size}", flush=True)
            continue
        array[completed:completed + step] = vectors.astype(np.float16)
        completed += step
        if completed % (batch_size * 20) < batch_size or completed == n:
            array.flush()
            progress.update(completed=completed, batch_size=batch_size, device="cuda",
                            peak_allocated_bytes=max(progress["peak_allocated_bytes"], torch.cuda.max_memory_allocated()),
                            peak_reserved_bytes=max(progress.get("peak_reserved_bytes", 0), torch.cuda.max_memory_reserved()),
                            elapsed_seconds=progress["elapsed_seconds"] + time.monotonic() - start)
            save_json(progress_path, progress)
            start = time.monotonic()
            print(f"{name} flip: {completed}/{n}", flush=True)


def rq4_records():
    annotations = json.loads((RAW / "building_parts.json").read_text(encoding="utf-8"))["annotations"]
    rng = np.random.default_rng(SEED)
    records = []
    for image_id, annotation in annotations.items():
        expert = boxes_for(annotation)
        records.append((image_id, "E", 0, expert, False))
        records.append((image_id, "K", 0, expert, True))
        random_sets = [random_boxes(expert, rng) for _ in range(20)]
        for draw, boxes in enumerate(random_sets):
            records.append((image_id, "R", draw, boxes, False))
            records.append((image_id, "R_keep", draw, boxes, True))
    path = OUT / "rq4_index.csv"
    frame = pd.DataFrame([{"image_id": x[0], "condition": x[1], "draw": x[2],
                           "boxes": json.dumps(x[3]), "keep_only": x[4]} for x in records])
    if path.exists():
        assert pd.read_csv(path).equals(frame)
    else:
        frame.to_csv(path, index=False)
    return records


def occlude_with_cached_fill(image, filled, boxes, keep_only):
    """Same RQ4 mask and radius as occlude(), with the per-image mean reused."""
    mask = Image.new("L", image.size, color=0)
    draw = ImageDraw.Draw(mask)
    for box in boxes:
        draw.rectangle(pixel_box(box, *image.size), fill=255)
    mask = mask.filter(ImageFilter.GaussianBlur(radius=3))
    return Image.composite(image, filled, mask) if keep_only else Image.composite(filled, image, mask)


def extract_rq4(model_name, size, view, batch_size):
    assert torch.cuda.is_available() and view in ("base", "flip")
    records = rq4_records()
    n = len(records)
    assert n == 139 * 42
    dino = model_name.startswith("DINO")
    model = Dino(model_name) if dino else (LocalClip() if model_name == "CLIP-L" else FrozenModel(model_name, device="cuda"))
    width = model.width if dino else 768
    if dino:
        token_path, cls_path, progress_path = paths_for(model_name, size, view, occlusion=True)
        progress = json.loads(progress_path.read_text(encoding="utf-8")) if progress_path.exists() else {"completed": 0, "peak_allocated_bytes": 0, "elapsed_seconds": 0.0}
        tokens = open_output(token_path, (n, 256, width), progress["completed"])
        cls = open_output(cls_path, (n, width), progress["completed"])
    else:
        stem = f"{model_name}_rq4_{view}"
        output_path, progress_path = OUT / f"{stem}.npy", OUT / f"{stem}_progress.json"
        progress = json.loads(progress_path.read_text(encoding="utf-8")) if progress_path.exists() else {"completed": 0, "peak_allocated_bytes": 0, "elapsed_seconds": 0.0}
        vectors = open_output(output_path, (n, width), progress["completed"])
    completed = progress["completed"]
    lookup = pd.read_parquet(PROCESSED / "metadata.parquet").set_index("image_id")
    start = time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    cached_id, cached_image, cached_fill = None, None, None
    while completed < n:
        step = min(batch_size, n - completed)
        prepared = []
        for image_id, condition, draw, boxes, keep_only in records[completed:completed + step]:
            if image_id != cached_id:
                source = image_path(lookup.loc[image_id, "image_path"]) if dino and size == 448 else PROCESSED / "img256" / image_id
                with Image.open(source) as opened:
                    cached_image = ImageOps.exif_transpose(opened).convert("RGB")
                average = tuple(np.asarray(cached_image).reshape(-1, 3).mean(axis=0).astype(np.uint8))
                cached_fill = Image.new("RGB", cached_image.size, color=average)
                cached_id = image_id
            obscured = occlude_with_cached_fill(cached_image, cached_fill, boxes, keep_only)
            if dino:
                prepared.append(resize_crop(obscured, size, model.mean, model.std, view, image_id))
            else:
                if view == "flip":
                    obscured = ImageOps.mirror(obscured)
                prepared.append(model.transform(obscured))
        try:
            if dino:
                block, head = model.features(torch.stack(prepared), size)
            else:
                block = model.image_features(torch.stack(prepared)).numpy().astype(np.float16)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if batch_size == 1: raise
            batch_size = max(1, batch_size // 2)
            print(f"RQ4 {model_name} {size} {view}: OOM, batch {batch_size}", flush=True)
            continue
        if dino:
            tokens[completed:completed + step] = block
            cls[completed:completed + step] = head
        else:
            vectors[completed:completed + step] = block
        completed += step
        if completed % (batch_size * 20) < batch_size or completed == n:
            if dino:
                tokens.flush(); cls.flush()
            else:
                vectors.flush()
            progress.update(completed=completed, batch_size=batch_size, device="cuda",
                            peak_allocated_bytes=max(progress["peak_allocated_bytes"], torch.cuda.max_memory_allocated()),
                            peak_reserved_bytes=max(progress.get("peak_reserved_bytes", 0), torch.cuda.max_memory_reserved()),
                            elapsed_seconds=progress["elapsed_seconds"] + time.monotonic() - start)
            save_json(progress_path, progress)
            start = time.monotonic()
            print(f"RQ4 {model_name} {size} {view}: {completed}/{n}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["DINO-L", "DINO-B", "CLIP-L", "SIG2-B"], required=True)
    parser.add_argument("--size", type=int, choices=[224, 448])
    parser.add_argument("--view", choices=VIEWS)
    parser.add_argument("--rq4", action="store_true")
    args = parser.parse_args()
    assert torch.cuda.is_available(), "CUDA required"
    OUT.mkdir(exist_ok=True)
    metadata = pd.read_parquet(PROCESSED / "metadata.parquet")
    ids_file = OUT / "image_ids.csv"
    if ids_file.exists():
        assert pd.read_csv(ids_file).image_id.tolist() == metadata.image_id.tolist()
    else:
        metadata[["image_id", "church_id"]].to_csv(ids_file, index=False)
    if args.rq4:
        assert args.view in ("base", "flip")
        assert (args.model == "DINO-L" and args.size in (224, 448)) or (args.model == "DINO-B" and args.size == 448) or (args.model in ("CLIP-L", "SIG2-B") and args.size is None)
        progress_path = (paths_for(args.model, args.size, args.view, occlusion=True)[2]
                         if args.model.startswith("DINO") else OUT / f"{args.model}_rq4_{args.view}_progress.json")
        if progress_path.exists() and json.loads(progress_path.read_text(encoding="utf-8"))["completed"] == 139 * 42:
            print(f"Verified complete: {progress_path.name}", flush=True)
            return
        extract_rq4(args.model, args.size, args.view,
                    batch_size=(32 if args.size == 224 else 8 if args.model == "DINO-L" else 16)
                    if args.model.startswith("DINO") else 32)
    elif args.model.startswith("DINO"):
        assert args.size and args.view
        progress_path = paths_for(args.model, args.size, args.view)[2]
        if progress_path.exists() and json.loads(progress_path.read_text(encoding="utf-8"))["completed"] == len(metadata):
            print(f"Verified complete: {progress_path.name}", flush=True)
            return
        model = Dino(args.model)
        extract_dino_view(model, metadata, args.size, args.view,
                          batch_size=32 if args.size == 224 else 8 if args.model == "DINO-L" else 16)
    else:
        assert args.size is None and args.view is None
        progress_path = OUT / f"{args.model}_flip_progress.json"
        if progress_path.exists() and json.loads(progress_path.read_text(encoding="utf-8"))["completed"] == len(metadata):
            print(f"Verified complete: {progress_path.name}", flush=True)
            return
        extract_global_flip(args.model, metadata)


if __name__ == "__main__":
    main()
