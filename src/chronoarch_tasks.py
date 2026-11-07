"""Leakage-safe task labels, fixed splits, losses and readouts for v2."""
import json
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from pathing import RAW
from rq3 import authenticity_groups, dating_splits

SEED = 0


@dataclass
class TaskFold:
    task: str
    fold: int
    train: np.ndarray
    val: np.ndarray
    test: np.ndarray
    eligible: np.ndarray
    style: np.ndarray
    fine: np.ndarray
    year: np.ndarray
    style_names: list
    fine_names: list
    churches: np.ndarray
    image_ids: np.ndarray


def grouped_inner_validation(outer_train, labels, churches, seed):
    """Take 15% of churches per label; outer-test churches remain untouched."""
    rng = np.random.default_rng(seed)
    table = pd.DataFrame({"church": churches[outer_train], "label": labels[outer_train]})
    assert table.groupby("church").label.nunique().max() == 1
    holdout = set()
    for label, group in table.drop_duplicates("church").groupby("label"):
        items = np.array(sorted(group.church.tolist()))
        rng.shuffle(items)
        count = max(1, min(len(items) - 1, round(0.15 * len(items))))
        holdout.update(items[:count])
    is_val = np.isin(churches[outer_train], list(holdout))
    train, val = outer_train[~is_val], outer_train[is_val]
    assert len(train) and len(val)
    assert not set(churches[train]) & set(churches[val])
    return train, val


def task_folds(metadata, task):
    ids = metadata.image_id.to_numpy()
    churches = metadata.church_id.to_numpy()
    years = metadata.construction_year.to_numpy(dtype=np.float32)
    fine_qids = metadata.all_styles.map(json.loads).map(lambda styles: styles[0]).to_numpy()
    names = dict(line.split(" ", 1) for line in (RAW / "style_names.txt").read_text(encoding="utf-8").splitlines())
    if task == "wc4_rq4":
        annotations = json.loads((RAW / "building_parts.json").read_text(encoding="utf-8"))["annotations"]
        annotation_ids = set(annotations)
        assert len(annotation_ids) == 139 and annotation_ids.issubset(set(ids))
        expert_churches = set(churches[np.isin(ids, list(annotation_ids))])
        valid_style = metadata.wc4_style_id.notna().to_numpy()
        train_mask = (metadata.split_wc4.eq("train").to_numpy() & valid_style &
                      ~np.isin(churches, list(expert_churches)))
        val_mask = (metadata.split_wc4.eq("val").to_numpy() & valid_style &
                    ~np.isin(churches, list(expert_churches)))
        test_mask = np.isin(ids, list(annotation_ids))
        eligible = np.flatnonzero(train_mask | val_mask | test_mask)
        style_qids = metadata.wc4_style_id.to_numpy()
        style_names = sorted(set(style_qids[train_mask | val_mask]))
        assert len(style_names) == 4
        fixed_train, fixed_val = np.flatnonzero(train_mask), np.flatnonzero(val_mask)
        test = np.flatnonzero(test_mask)
        outer = [(np.concatenate((fixed_train, fixed_val)), test)]
        mode = "style"
    elif task in ("wc6", "wc14"):
        label_col, split_col = f"{task}_style_id", f"split_{task}"
        eligible = np.flatnonzero(metadata[label_col].notna().to_numpy())
        style_qids = metadata[label_col].to_numpy()
        style_names = sorted(set(style_qids[eligible]))
        outer = [(np.flatnonzero(metadata[split_col].isin(("train", "val")).to_numpy()),
                  np.flatnonzero(metadata[split_col].eq("test").to_numpy()))]
        fixed_train = np.flatnonzero(metadata[split_col].eq("train").to_numpy())
        fixed_val = np.flatnonzero(metadata[split_col].eq("val").to_numpy())
        mode = "style"
    elif task in ("gothic", "romanesque"):
        style_name = task.capitalize()
        original, revival = authenticity_groups(metadata)[style_name]["mixed"]
        eligible = np.flatnonzero(original | revival)
        binary = revival.astype(int)
        style_qids = binary
        style_names = [0, 1]
        local = list(StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
                     .split(np.zeros(len(eligible)), binary[eligible], churches[eligible]))
        outer = [(eligible[train], eligible[test]) for train, test in local]
        fixed_train = fixed_val = None
        mode = "authenticity"
    elif task == "dating":
        eligible = np.flatnonzero((metadata.wc14_style_id.notna() &
                                   metadata.construction_year.between(800, 2020)).to_numpy())
        style_qids = metadata.wc14_style_id.to_numpy()
        style_names = sorted(set(style_qids[eligible]))
        local = dating_splits(style_qids[eligible], churches[eligible])
        outer = [(eligible[train], eligible[test]) for train, test in local]
        fixed_train = fixed_val = None
        mode = "dating"
    else:
        raise ValueError(task)
    fine_source = (np.concatenate((fixed_train, fixed_val))
                   if task == "wc4_rq4" else eligible)
    if mode == "authenticity":
        fine_names = sorted({qid for qid in fine_qids[eligible] if "revival" not in names.get(qid, "").lower()})
    else:
        fine_names = sorted(set(fine_qids[fine_source]))
    style_map = {value: i for i, value in enumerate(style_names)}
    fine_map = {value: i for i, value in enumerate(fine_names)}
    style = np.full(len(metadata), -1, dtype=np.int64)
    fine = np.full(len(metadata), -1, dtype=np.int64)
    labelled = (np.concatenate((fixed_train, fixed_val))
                if task == "wc4_rq4" else eligible)
    style[labelled] = [style_map[value] for value in style_qids[labelled]]
    fine[fine_source] = [fine_map.get(value, -1) for value in fine_qids[fine_source]]
    if mode == "authenticity":
        assert all("revival" not in names.get(fine_names[index], "").lower()
                   for index in fine[eligible] if index >= 0)
    folds = []
    for number, (outer_train, test) in enumerate(outer):
        if fixed_train is not None:
            train, val = fixed_train, fixed_val
        else:
            train, val = grouped_inner_validation(outer_train, style, churches, SEED + number)
        assert not set(churches[train]) & set(churches[val])
        assert not set(churches[train]) & set(churches[test])
        assert not set(churches[val]) & set(churches[test])
        assert set(np.concatenate((train, val, test))) == set(eligible)
        folds.append(TaskFold(task, number, train, val, test, eligible, style, fine, years,
                              style_names, fine_names, churches, ids))
    return folds


def validation_metric(task, truth, prediction):
    from sklearn.metrics import balanced_accuracy_score, mean_absolute_error, roc_auc_score
    if task in ("wc4_rq4", "wc6", "wc14"):
        return balanced_accuracy_score(truth, prediction.argmax(axis=1))
    if task in ("gothic", "romanesque"):
        return roc_auc_score(truth, prediction)
    return -mean_absolute_error(truth, prediction)
