"""Audit and reformat first-round B-1 predictions without retraining."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd
from chronoarch_tasks import task_folds
from pathing import PROCESSED

OLD = ROOT / "results"
NEW = ROOT / "results_v2"
MODELS = ("R50", "DINO-B", "CLIP-L", "SIG2-B")


def main():
    metadata = pd.read_parquet(PROCESSED / "metadata.parquet")
    rows = []
    for task in ("wc6", "wc14"):
        fold = task_folds(metadata, task)[0]
        labels = {qid: i for i, qid in enumerate(fold.style_names)}
        for model in MODELS:
            path = OLD / f"rq1_prob_{model}_{task}.npz"
            with np.load(path, allow_pickle=True) as source:
                assert source["image_id"].tolist() == fold.image_ids[fold.test].tolist()
                assert set(source["classes"].tolist()) == set(fold.style_names)
                class_order = [list(source["classes"]).index(qid) for qid in fold.style_names]
                probabilities = source["probabilities"][:, class_order]
                truth = np.array([labels[qid] for qid in source["true"]])
                assert np.array_equal(truth, fold.style[fold.test])
                part = pd.DataFrame({"task": task, "model": model,
                                     "image_id": source["image_id"],
                                     "church_id": source["church_id"],
                                     "truth": truth})
                for i in range(len(class_order)):
                    part[f"prob_{i}"] = probabilities[:, i]
                rows.append(part)
    auth = pd.read_csv(OLD / "rq3a_oof.csv")
    for task in ("gothic", "romanesque"):
        fold = task_folds(metadata, task)[0]
        for model in MODELS:
            part = auth.loc[(auth["style"].str.lower() == task) &
                            auth.rule.eq("mixed") & auth.model.eq(model)].copy()
            assert set(part.image_id) == set(fold.image_ids[fold.eligible])
            rows.append(pd.DataFrame({"task": task, "model": model,
                                      "image_id": part.image_id,
                                      "church_id": part.church_id,
                                      "truth": part["true"].astype(int),
                                      "prediction": part.score}))
    date = pd.read_csv(OLD / "rq3b_oof.csv")
    for model in MODELS:
        part = date.loc[date.dataset.eq("main") & date.model.eq(model) &
                        date.method.eq("image_only")].copy()
        fold = task_folds(metadata, "dating")[0]
        assert set(part.image_id) == set(fold.image_ids[fold.eligible])
        rows.append(pd.DataFrame({"task": "dating", "model": model,
                                  "image_id": part.image_id,
                                  "church_id": part.church_id,
                                  "truth": part.true_year,
                                  "prediction": part.predicted_year}))
    output = pd.concat(rows, ignore_index=True)
    assert not output.duplicated(["task", "model", "image_id"]).any()
    output.to_csv(NEW / "b1_reused_oof.csv", index=False)
    print({"rows": len(output), "tasks": output.task.nunique(),
           "models": output.model.nunique()}, flush=True)


if __name__ == "__main__":
    main()
