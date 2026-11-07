"""Three-seed WC6/Gothic ablations using the full model's validation-selected settings."""
import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from chronoarch_tasks import task_folds
from chronoarch_train import FeatureStore, RESULTS, TrainConfig, evaluate_checkpoint
from pathing import PROCESSED
from run_chronoarch_v2 import claim_test, run_config, save_test, stored_row

VARIANTS = ("A0", "A1", "A2", "A3", "A4", "R1", "R2", "R4")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=("wc6", "gothic"), required=True)
    args = parser.parse_args()
    assert torch.cuda.is_available()
    metadata = pd.read_parquet(PROCESSED / "metadata.parquet")
    store = FeatureStore(include_b=args.task == "wc6")
    for fold in task_folds(metadata, args.task):
        selected = RESULTS / f"selected_{fold.task}_{fold.fold}.json"
        assert selected.exists(), "Full model must be validation-selected first"
        base = TrainConfig(**json.loads(selected.read_text(encoding="utf-8"))["final"])
        for variant in VARIANTS:
            if variant == "R1" and fold.task != "wc6":
                continue
            if variant == "A0":
                assert all((RESULTS / f"pred_ChronoArch-F_{fold.task}_{fold.fold}_{seed}.csv").exists()
                           for seed in range(3))
                continue
            config = replace(base, variant=variant,
                             tau=0 if variant == "R2" else base.tau)
            for seed in range(3):
                target = RESULTS / f"pred_{variant}_{fold.task}_{fold.fold}_{seed}.csv"
                if target.exists():
                    continue
                if variant == "R4":
                    row = stored_row("ChronoArch-F", fold, base, seed)
                    assert row is not None
                else:
                    row = stored_row("ChronoArch-F", fold, config, seed)
                    if row is None:
                        row = run_config(fold, store, config, seed, f"ablation_{variant}")
                claim_test(fold, variant, seed)
                pred, seconds = evaluate_checkpoint(fold, store, Path(row["checkpoint"]),
                                                    tta=(variant != "R4"))
                save_test(fold, variant, seed, pred, seconds)
            print(f"Completed {args.task} fold {fold.fold} {variant}", flush=True)


if __name__ == "__main__":
    main()
