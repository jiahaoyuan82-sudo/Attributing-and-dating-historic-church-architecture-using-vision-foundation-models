"""Mark final validation-selected configurations after all training has finished."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd
from chronoarch_train import TrainConfig
from run_chronoarch_v2 import config_id


def main():
    results = ROOT / "results_v2"
    path = results / "selection_log.csv"
    frame = pd.read_csv(path)
    selected = []
    for row in frame.itertuples(index=False):
        marker = False
        if row.model in ("ChronoArch-F", "ChronoArch-L"):
            chosen_path = results / f"selected_{row.task}_{row.fold}.json"
            if chosen_path.exists():
                config = TrainConfig(**json.loads(chosen_path.read_text(
                    encoding="utf-8"))["final"])
                marker = (str(row.config_id) == config_id(config) and
                          not str(row.stage).startswith("ablation_"))
        elif row.model in ("B-2_224", "B-2_448", "B-4",
                           "B-6_DINO", "B-6_CLIP", "B-6_SIG"):
            chosen_path = results / f"selected_{row.model}_{row.task}_{row.fold}.json"
            if chosen_path.exists():
                chosen = json.loads(chosen_path.read_text(encoding="utf-8"))["C"]
                marker = np.isclose(float(row.config_id), chosen)
        else:
            marker = str(row.stage) == "early_stop"
        selected.append(marker)
    frame["selected_final_by_validation"] = selected
    temporary = path.with_suffix(".selected.tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)
    print({"rows": len(frame), "selected_rows": int(frame.selected_final_by_validation.sum())},
          flush=True)


if __name__ == "__main__":
    main()
