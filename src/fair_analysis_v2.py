"""Summarize isolated fair baseline predictions and append a report section."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from chronoarch_tasks import task_folds, validation_metric
from chronoarch_train import FeatureStore
from fair_baselines_v2 import OUT, linear_predict
from pathing import PROCESSED, ROOT
from significance_v2 import bootstrap_difference, delong_p, holm, old_dinob
from summarize_v2 import church_bootstrap, metric, model_predictions, prediction_values, primary, stack_parts

TASKS = ("wc6", "wc14", "gothic", "romanesque", "dating")
MODELS = ("B-2_224", "B-2_448", "B-3", "B-4", "B-5", "B-6", "B-7")
NEURAL = {"B-3", "B-5", "B-7"}


def fair_predictions(task, model, folds):
    seeds = range(3) if model in NEURAL else range(1)
    per_seed = {seed: [] for seed in seeds}
    ensemble = []
    for fold in folds:
        frames = [pd.read_csv(OUT / f"pred_{model}_{task}_{fold.fold}_{seed}.csv") for seed in seeds]
        assert all(frame.image_id.tolist() == fold.image_ids[fold.test].tolist() for frame in frames)
        values = [prediction_values(frame) for frame in frames]
        for seed, frame, value in zip(seeds, frames, values):
            per_seed[seed].append((frame, value))
        ensemble.append((frames[0], np.mean(values, axis=0)))
    return per_seed, ensemble


def validation_scores(task, folds):
    selected_fold = folds[0]
    scores = {}
    for model in MODELS:
        if model == "B-6":
            store = FeatureStore(include_tokens=False)
            values = []
            for kind in ("B-6_DINO", "B-6_CLIP", "B-6_SIG"):
                chosen = json.loads((OUT / f"selected_{kind}_{task}_0.json").read_text(encoding="utf-8"))["value"]
                import pickle
                with (OUT / "checkpoints" / f"linear_{kind}_{task}_0_{chosen:.12g}.pkl").open("rb") as stream:
                    model_fit = pickle.load(stream)
                values.append(linear_predict(model_fit, selected_fold, store, selected_fold.val, kind))
            pred = np.mean(values, axis=0)
            truth = (selected_fold.year[selected_fold.val] if task == "dating" else
                     selected_fold.style[selected_fold.val])
            scores[model] = validation_metric(task, truth, pred)
            del store
        elif model in NEURAL:
            record = json.loads((OUT / f"selected_{model}_{task}_0.json").read_text(encoding="utf-8"))
            scores[model] = float(record["val_metric"])
        else:
            record = json.loads((OUT / f"selected_{model}_{task}_0.json").read_text(encoding="utf-8"))
            scores[model] = float(record["validation_metric"])
    return scores


def fmt(value, digits=4):
    if pd.isna(value):
        return "—"
    return f"{value:.{digits}f}"


def p_fmt(value):
    if value < 1e-16:
        return "p < 1e-16"
    return f"{value:.3g}"


def main():
    report = ROOT / "RESULTS_v2.md"
    original = report.read_text(encoding="utf-8")
    assert "## 公平基线重跑" not in original, "Report section already exists"
    metadata = pd.read_parquet(PROCESSED / "metadata.parquet")
    summary, tests, selections = [], [], []
    for task in TASKS:
        folds = task_folds(metadata, task)
        scores = validation_scores(task, folds)
        winner = max(MODELS, key=lambda name: (scores[name], name))
        selections.append(dict(task=task, selected=winner, **{f"val_{k}": v for k, v in scores.items()}))
        fair = {}
        for model in MODELS:
            per_seed, parts = fair_predictions(task, model, folds)
            frame, ensemble = stack_parts(parts)
            seed_values = [metric(task, frame.truth.to_numpy(), stack_parts(items)[1])[primary(task)]
                           for items in per_seed.values()]
            value = metric(task, frame.truth.to_numpy(), ensemble)[primary(task)]
            ci = church_bootstrap(task, frame, ensemble)
            summary.append(dict(task=task, model=model, images=len(frame), seeds=len(seed_values),
                                metric=primary(task), seed_mean=float(np.mean(seed_values)),
                                seed_sd=float(np.std(seed_values, ddof=1)) if len(seed_values)>1 else np.nan,
                                ensemble=value, ci_low=ci[0], ci_high=ci[1],
                                validation_score=scores[model], validation_best=model==winner))
            fair[model] = (frame, ensemble)
        left_meta, left = stack_parts(model_predictions(task, "ChronoArch-F", folds)[1])
        right_meta, right = fair[winner]
        assert left_meta[["image_id", "church_id", "truth"]].equals(
               right_meta[["image_id", "church_id", "truth"]])
        old = old_dinob(task, left_meta)
        for name, comparison in ((winner, right), ("B-1 DINO-B", old)):
            truth = left_meta.truth.to_numpy()
            direction = -1 if task == "dating" else 1
            delta = direction*(metric(task, truth, left)[primary(task)] -
                               metric(task, truth, comparison)[primary(task)])
            ci, samples = bootstrap_difference(task, left_meta, left, comparison)
            if task in ("wc6", "wc14"):
                a, b = left.argmax(-1)==truth, comparison.argmax(-1)==truth
                discordant = int(np.count_nonzero(a!=b))
                p = float(binomtest(int(np.count_nonzero(a & ~b)), discordant, .5).pvalue) if discordant else 1.
                test = "exact_McNemar"
            elif task in ("gothic", "romanesque"):
                p, test = delong_p(truth, left, comparison), "DeLong"
            else:
                p, test = float(2*min((samples<=0).mean(),(samples>=0).mean())), "paired_church_bootstrap"
            tests.append(dict(task=task, baseline=name, metric=primary(task), improvement=delta,
                              ci_low=ci[0], ci_high=ci[1], test=test, p=p,
                              proposed_metric=metric(task, truth, left)[primary(task)],
                              baseline_metric=metric(task, truth, comparison)[primary(task)]))
        print("summarized",task,"winner",winner,flush=True)
    summary = pd.DataFrame(summary)
    tests = pd.DataFrame(tests)
    tests["p_holm"] = holm(tests.p.to_numpy())
    pd.DataFrame(selections).to_csv(OUT / "validation_best.csv", index=False)
    summary.to_csv(OUT / "main_comparison_fair.csv", index=False)
    tests.to_csv(OUT / "significance_fair.csv", index=False)
    lines = ["## 公平基线重跑", "",
             "仅重跑 B-2 至 B-7；ChronoArch-F 的配置、训练、检查点和测试预测均沿用原结果。旧基线文件保留。本节所有新文件位于 `results_v2/baselines_fair/`。",
             "",
             "B-2、B-4、B-6 对各编码器特征分别做 L2 归一化后拼接；分类 C 与断代 Ridge alpha 的初始网格均为 {1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 0.1, 0.3, 1, 3, 10}，若验证最优落在边缘则按十倍向外扩展。B-3、B-5、B-7 在 seed 0 上搜索 lr × τ × dropout 共 8 组，训练使用 base/flip 增强、fp16 CUDA、40 epoch 上限与 patience 6；选中配置使用 seeds 0–2 并做翻转测试增强和三 seed 集成。断代任务不使用风格 logit adjustment，τ 对断代损失无作用；该网格仍按要求记录。",
             "",
             "所有配置只用验证集选出。Gothic、Romanesque、断代只在第 1 折内部验证集选超参数和比较基线，后四折沿用；最终测试预测每个配置仅评估一次。B-2/B-4/B-6 为确定性模型，表中只有一个独立 seed；其额外 seed 不会提供新的随机重复。95% CI 使用 1,000 次教堂级 bootstrap。正差值表示 ChronoArch-F 更好，断代表示 MAE 降低年数。Holm 校正涵盖本节十项预定配对检验。",
             "", "### 验证集选出的最强公平基线", "",
             "| 任务 | 基线 | 验证分数 |", "|---|---|---:|"]
    for item in selections:
        lines.append(f"| {item['task']} | {item['selected']} | {fmt(item['val_'+item['selected']])} |")
    lines += ["", "### 全部公平基线主结果", "",
              "| 任务 | 方法 | 图像数 | seed | seed 均值 ± SD | 集成/单次 | 95% CI |",
              "|---|---|---:|---:|---:|---:|---|"]
    for row in summary.itertuples():
        mark = " ★" if row.validation_best else ""
        lines.append(f"| {row.task} | {row.model}{mark} | {row.images} | {row.seeds} | "
                     f"{fmt(row.seed_mean)} ± {fmt(row.seed_sd)} | {fmt(row.ensemble)} | "
                     f"[{fmt(row.ci_low)}, {fmt(row.ci_high)}] |")
    lines += ["", "### 与 ChronoArch-F 的配对比较", "",
              "分类差值和区间基于平衡准确率；McNemar 检验逐图正确与否。AUROC 使用 DeLong，断代使用配对教堂 bootstrap。",
              "", "| 任务 | 对照 | 差值 | 95% CI | 检验 | p | Holm p |",
              "|---|---|---:|---|---|---:|---:|"]
    for row in tests.itertuples():
        lines.append(f"| {row.task} | {row.baseline} | {fmt(row.improvement)} | "
                     f"[{fmt(row.ci_low)}, {fmt(row.ci_high)}] | {row.test} | "
                     f"{p_fmt(row.p)} | {p_fmt(row.p_holm)} |")
    lines += ["", "代码修改前后的全量 `code/src/*.py` SHA-256 见 `baselines_fair/code_src_sha256_before.csv`、`code_src_sha256_after.csv`。新训练与分析代码独立于原结果。", ""]
    report.write_text(original.rstrip()+"\n\n"+"\n".join(lines),encoding="utf-8")
    print("appended fair section",len(lines),"lines",flush=True)


if __name__ == "__main__":
    main()
