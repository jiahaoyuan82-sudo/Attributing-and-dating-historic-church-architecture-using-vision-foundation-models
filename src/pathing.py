"""Resolve experiment and dataset paths before and after the S8 desktop move."""
from pathlib import Path

import yaml


CODE_ROOT = Path(__file__).resolve().parents[1]
ROOT = CODE_ROOT.parent if CODE_ROOT.name == "code" else CODE_ROOT
_config = yaml.safe_load((ROOT / "configs/config.yaml").read_text(encoding="utf-8"))
_dataset = _config.get("dataset", {})
RAW = (ROOT / _dataset.get("raw_path", "data/raw")).resolve()
PROCESSED = (ROOT / _dataset.get("processed_path", "data/processed")).resolve()


def image_path(recorded_path):
    """Map an S1 metadata path to the current raw-data location."""
    recorded = Path(recorded_path)
    if recorded.parts[:2] == ("data", "raw"):
        return RAW.joinpath(*recorded.parts[2:])
    if recorded.is_absolute():
        return recorded
    return ROOT / recorded
