"""Church-level separation in official and generated task splits."""
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pathing import PROCESSED


def test_no_church_leakage():
    metadata = pd.read_parquet(PROCESSED / "metadata.parquet")
    for task in ("wc14", "wc6", "wc4"):
        column = f"split_{task}"
        used = metadata[metadata[column].isin(["train", "val", "test"])]
        assert used.groupby("church_id")[column].nunique().max() == 1, task
    duplicates = pd.read_csv(PROCESSED / "near_duplicate_pairs.csv")
    if len(duplicates):
        excluded = set(metadata.loc[metadata["split_wc14"] == "excluded_duplicate", "image_id"])
        assert set(duplicates["test_image_id"]).issubset(excluded)


if __name__ == "__main__":
    test_no_church_leakage()
    print("No church leakage in WC14, WC6, or WC4 splits")
