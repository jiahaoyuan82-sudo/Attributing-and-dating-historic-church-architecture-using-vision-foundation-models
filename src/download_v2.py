"""Download the exact B1 DINOv2-L checkpoint into the experiment cache."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
os.environ["HF_HOME"] = str(ROOT / "env" / "hf_cache")
os.environ["HF_HUB_DISABLE_XET"] = "1"
from huggingface_hub import snapshot_download

if __name__ == "__main__":
    path = snapshot_download(
        "facebook/dinov2-large",
        allow_patterns=["config.json", "model.safetensors", "preprocessor_config.json"],
    )
    print(path, flush=True)
