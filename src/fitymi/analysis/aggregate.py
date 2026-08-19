"""Turn a directory of run records into the tables the paper reports.

Deliberately dumb about where runs came from: it globs JSON records, groups by arm,
and computes across-seed intervals. That way a sweep can be run in one process, in
250 SLURM jobs, or across three machines, and the analysis is identical.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from ..eval.stats import Interval, exchange_rate, mean_ci, trend_test

PRIMARY_METRIC = "balanced_accuracy"


def load_runs(root: str | Path, pattern: str = "**/run_*.json") -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for path in sorted(Path(root).glob(pattern)):
        with open(path) as fh:
            record = json.load(fh)
        split = record.get("test") or record.get("val") or {}
        config = record.get("config", {})
        rows.append(
            {
                "run_id": record.get("run_id"),
                "arm": record.get("arm"),
                "kind": config.get("mix_kind"),
                "synthetic_fraction": config.get("synthetic_fraction"),
                "real_budget_fraction": config.get("real_budget_fraction"),
                "synthetic_multiplier": config.get("synthetic_multiplier"),
                "generator": config.get("generator"),
                "arch": config.get("arch"),
                "init": config.get("init"),
                "seed": config.get("seed"),
                "n_train": record.get("train_summary", {}).get("n"),
                "evaluated_on": "test" if record.get("test") else "val",
                "balanced_accuracy": split.get("balanced_accuracy"),
                "accuracy": split.get("accuracy"),
                "macro_f1": split.get("macro_f1"),
                "qwk": split.get("qwk"),
                "mae": split.get("mae"),
                "ece": split.get("ece"),
                "tail_recall": _tail_recall(split),
                "path": str(path),
            }
        )
    df = pd.DataFrame(rows)
    if not df.empty and df["run_id"].duplicated().any():
        # The real-subset controls are identical across generator arms and may be
        # written under more than one output directory. Counting them twice would
        # not move the mean but would halve the confidence intervals.
        n_dupes = int(df["run_id"].duplicated().sum())
        df = df.drop_duplicates(subset="run_id", keep="first").reset_index(drop=True)
        logging.getLogger(__name__).info(
            "dropped %d duplicate run records (same run_id under multiple paths)", n_dupes
        )
    return df


def _tail_recall(split: dict) -> float:
    recalls = split.get("per_class_recall") or {}
    vals = [recalls.get("severe"), recalls.get("very_severe")]
    vals = [v for v in vals if v is not None and np.isfinite(v)]
    return float(np.mean(vals)) if vals else float("nan")


@dataclass
class ArmSummary:
    arm: str
    generator: str | None
    init: str | None
    n_seeds: int
    metric: str
    interval: Interval
    synthetic_fraction: float | None = None

    def to_row(self) -> dict:
        return {
            "arm": self.arm,
            "generator": self.generator,
            "init": self.init,
            "synthetic_fraction": self.synthetic_fraction,
            "n_seeds": self.n_seeds,
            "metric": self.metric,
            "mean": self.interval.point,
            "ci_lo": self.interval.lo,
            "ci_hi": self.interval.hi,
        }


#: Generator label used for arms that contain no synthetic images. Such an arm is
#: run once, not once per generator, so it has to be attached to every generator's
#: curve at analysis time.
SHARED_GENERATOR = "none"


def expand_shared_arms(df: pd.DataFrame) -> pd.DataFrame:
    """Attach generator-independent arms to every generator's curve.

    The p=0 arm holds no synthetic images and is therefore identical whichever
    generator arm asked for it, so it is trained once. Every curve still needs its
    anchor at zero, and the trend test needs the point to fit a slope through, so the
    row is copied to each generator here rather than being trained repeatedly.
    """
    if df.empty or "generator" not in df:
        return df
    shared = df[df["generator"] == SHARED_GENERATOR]
    if shared.empty:
        return df
    generators = [
        g for g in df["generator"].dropna().unique()
        if g not in (SHARED_GENERATOR, "real_subset")
    ]
    if not generators:
        return df
    copies = [df[df["generator"] != SHARED_GENERATOR]]
    for generator in generators:
        block = shared.copy()
        block["generator"] = generator
        copies.append(block)
    return pd.concat(copies, ignore_index=True)


def summarise_arms(
    df: pd.DataFrame,
    metric: str = PRIMARY_METRIC,
    group_keys: Iterable[str] = ("arm", "generator", "init"),
) -> pd.DataFrame:
    keys = [k for k in group_keys if k in df.columns]
    summaries: list[dict] = []
    for values, group in df.groupby(keys, dropna=False):
        values = values if isinstance(values, tuple) else (values,)
        info = dict(zip(keys, values))
        summary = ArmSummary(
            arm=str(info.get("arm")),
            generator=info.get("generator"),
            init=info.get("init"),
            n_seeds=int(group["seed"].nunique()),
            metric=metric,
            interval=mean_ci(group[metric].tolist()),
            synthetic_fraction=(
                float(group["synthetic_fraction"].iloc[0])
                if group["synthetic_fraction"].notna().any()
                else None
            ),
        )
        summaries.append(summary.to_row())
    out = pd.DataFrame(summaries)
    if "synthetic_fraction" in out:
        out = out.sort_values(["generator", "init", "synthetic_fraction"], na_position="first")
    return out.reset_index(drop=True)


def substitution_curve(df: pd.DataFrame, metric: str = PRIMARY_METRIC) -> pd.DataFrame:
    sub = df[df["kind"] == "substitution"] if "kind" in df else df
    return summarise_arms(expand_shared_arms(sub), metric)


def test_h2(df: pd.DataFrame, metric: str = PRIMARY_METRIC) -> dict:
    """Trend of `metric` against synthetic fraction, per generator/init cell."""
    sub = expand_shared_arms(df[df["kind"] == "substitution"] if "kind" in df else df)
    out: dict[str, dict] = {}
    for (generator, init), group in sub.groupby(["generator", "init"], dropna=False):
        fractions = sorted(group["synthetic_fraction"].dropna().unique())
        scores_by_seed: dict[int, list[float]] = {}
        for seed, seed_group in group.groupby("seed"):
            per_fraction = (
                seed_group.set_index("synthetic_fraction")[metric].reindex(fractions).tolist()
            )
            scores_by_seed[int(seed)] = per_fraction
        if not scores_by_seed:
            continue
        result = trend_test(fractions, scores_by_seed)
        out[f"{generator}|{init}"] = {
            "fractions": [float(f) for f in fractions],
            **result.to_dict(),
        }
    return out


def exchange_rates(df: pd.DataFrame, metric: str = PRIMARY_METRIC) -> pd.DataFrame:
    """Protocol §5.2: how many synthetic images buy one real image, per real budget."""
    add = expand_shared_arms(df[df["kind"] == "additive"] if "kind" in df else df)
    if add.empty:
        return pd.DataFrame(columns=["generator", "init", "real_budget", "target", "k"])

    rows: list[dict] = []
    for (generator, init), group in add.groupby(["generator", "init"], dropna=False):
        budgets = sorted(group["real_budget_fraction"].dropna().unique())
        # Real-only performance at each budget is the k=0 point.
        baseline = {
            b: float(
                group[(group["real_budget_fraction"] == b) & (group["synthetic_multiplier"] == 0)][
                    metric
                ].mean()
            )
            for b in budgets
        }
        for i, budget in enumerate(budgets[:-1]):
            target_budget = budgets[i + 1]
            target = baseline.get(target_budget, float("nan"))
            cell = group[group["real_budget_fraction"] == budget]
            curve = cell.groupby("synthetic_multiplier")[metric].mean().sort_index()
            if curve.empty or not np.isfinite(target):
                continue
            k = exchange_rate(curve.index.tolist(), curve.tolist(), target)
            rows.append(
                {
                    "generator": generator,
                    "init": init,
                    "real_budget": float(budget),
                    "target_budget": float(target_budget),
                    "target": target,
                    "k": k,
                    "reached": np.isfinite(k),
                }
            )
    return pd.DataFrame(rows)


def write_tables(df: pd.DataFrame, out_dir: str | Path, metric: str = PRIMARY_METRIC) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    curve = substitution_curve(df, metric)
    curve.to_csv(out_dir / "substitution_curve.csv", index=False)

    rates = exchange_rates(df, metric)
    rates.to_csv(out_dir / "exchange_rates.csv", index=False)

    trends = test_h2(df, metric)
    with open(out_dir / "h2_trend.json", "w") as fh:
        json.dump(trends, fh, indent=2, default=float)

    df.to_csv(out_dir / "all_runs.csv", index=False)
    return {"curve": curve, "exchange_rates": rates, "trends": trends}
