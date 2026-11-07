"""Illustrative path settings for a fresh reproduction workspace.

The copied experiment scripts do not import this file. They resolve the project
root from a ``code/src`` layout and read ``configs/config.yaml``. Copy the
published ``src`` directory to ``code/src`` and edit the YAML paths for your
own data location before running the scripts.
"""

import os
from pathlib import Path


PROJECT_ROOT = Path(os.environ.get("CHURCH_PROJECT_ROOT", ".")).resolve()
DATA_ROOT = Path(os.environ.get("WIKICHURCHES_DATA_ROOT", PROJECT_ROOT / "data")).resolve()
RAW_PATH = DATA_ROOT / "raw"
PROCESSED_PATH = DATA_ROOT / "processed"
CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"
FEATURES_PATH = PROJECT_ROOT / "features"
FEATURES_V2_PATH = PROJECT_ROOT / "features_v2"
RESULTS_PATH = PROJECT_ROOT / "results"
RESULTS_V2_PATH = PROJECT_ROOT / "results_v2"
