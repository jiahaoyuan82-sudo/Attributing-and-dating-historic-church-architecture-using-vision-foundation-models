"""Frozen backbones and common 224-pixel image preprocessing."""
import os
from pathlib import Path

from pathing import ROOT
os.environ.setdefault("HF_HOME", str(ROOT / "env/hf_cache"))
os.environ.setdefault("TORCH_HOME", str(ROOT / "env/torch_cache"))
os.environ.setdefault("XDG_CACHE_HOME", str(ROOT / "env/xdg_cache"))

import torch
from torchvision import transforms as T
from torchvision.transforms import InterpolationMode


def standard_transform(mean, std):
    return T.Compose([
        T.Resize(224, interpolation=InterpolationMode.BICUBIC),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=mean, std=std),
    ])


class FrozenModel:
    def __init__(self, name, device="cuda"):
        self.name = name
        if device != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("NVIDIA CUDA GPU is required for all deep-model forward passes")
        self.device = "cuda"
        self.processor = None
        self.tokenizer = None
        if name == "R50":
            import timm
            self.model = timm.create_model("resnet50.a1_in1k", pretrained=True, num_classes=0, global_pool="avg")
            cfg = self.model.default_cfg
            self.transform = standard_transform(cfg["mean"], cfg["std"])
        elif name == "DINO-B":
            from transformers import AutoImageProcessor, AutoModel
            identifier = "facebook/dinov2-base"
            self.processor = AutoImageProcessor.from_pretrained(identifier)
            self.model = AutoModel.from_pretrained(identifier)
            self.transform = standard_transform(self.processor.image_mean, self.processor.image_std)
        elif name == "CLIP-L":
            import open_clip
            self.model, _, self.transform = open_clip.create_model_and_transforms("ViT-L-14", pretrained="openai", force_quick_gelu=True)
            self.tokenizer = open_clip.get_tokenizer("ViT-L-14")
        elif name == "SIG2-B":
            from transformers import AutoProcessor, AutoModel
            identifier = "google/siglip2-base-patch16-224"
            self.processor = AutoProcessor.from_pretrained(identifier)
            self.model = AutoModel.from_pretrained(identifier)
            image_processor = self.processor.image_processor
            self.transform = standard_transform(image_processor.image_mean, image_processor.image_std)
        else:
            raise ValueError(name)
        self.model.eval().to(self.device)
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    @torch.inference_mode()
    def image_features(self, batch):
        batch = batch.to(self.device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            if self.name == "DINO-B":
                features = self.model(pixel_values=batch).last_hidden_state[:, 0]
            elif self.name == "CLIP-L":
                features = self.model.encode_image(batch)
            elif self.name == "SIG2-B":
                features = self.model.get_image_features(pixel_values=batch)
            else:
                features = self.model(batch)
        if hasattr(features, "pooler_output"):
            features = features.pooler_output
        elif isinstance(features, (tuple, list)):
            features = features[0]
        features = features.float()
        return torch.nn.functional.normalize(features, p=2, dim=-1).cpu()

    @torch.inference_mode()
    def text_features(self, texts):
        if self.name == "CLIP-L":
            tokens = self.tokenizer(texts).to(self.device)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                output = self.model.encode_text(tokens)
        elif self.name == "SIG2-B":
            inputs = self.processor(text=texts, padding="max_length", return_tensors="pt")
            inputs = {key: value.to(self.device) for key, value in inputs.items()}
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                output = self.model.get_text_features(**inputs)
        else:
            raise ValueError(f"No text encoder for {self.name}")
        if hasattr(output, "pooler_output"):
            output = output.pooler_output
        elif isinstance(output, (tuple, list)):
            output = output[0]
        return torch.nn.functional.normalize(output.float(), p=2, dim=-1).cpu()
