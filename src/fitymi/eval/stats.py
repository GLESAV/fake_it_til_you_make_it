"""Statistics.

Protocol §9. Three jobs: put confidence intervals on every arm, compare arms in a
way that respects the fact that all arms are scored on the *same* test images, and
turn the additive sweep into an exchange rate.

Effect sizes and intervals are the output. p-values are reported but are not the
argument, because with a ~290-image test set the binomial standard error alone is
around 2.7 points and a study this size is not entitled to claim 1-point effects.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Callable, Sequence

import numpy as np
from scipy import stats as sps

from ..utils.seeding import rng
from .metrics import balanced_accuracy


@dataclass
class Interval:
    point: float
    lo: float
    hi: float
    level: float = 0.95

    def __str__(self) -> str:
        return f"{self.point:.4f} [{self.lo:.4f}, {self.hi:.4f}]"

    def to_dict(self) -> dict:
        return asdict(self)


def mean_ci(values: Sequence[float], level: float = 0.95) -> Interval:
    """Across-seed mean with a t interval. Never report a single seed."""
    v = np.asarray([x for x in values if np.isfinite(x)], dtype=float)
    if v.size == 0:
        return Interval(float("nan"), float("nan"), float("nan"), level)
    if v.size == 1:
        return Interval(float(v[0]), float("nan"), float("nan"), level)
    mean = float(v.mean())
    sem = float(v.std(ddof=1) / np.sqrt(v.size))
    half = sps.t.ppf(0.5 + level / 2, v.size - 1) * sem
    return Interval(mean, mean - half, mean + half, level)


def bootstrap_metric(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric: Callable = balanced_accuracy,
    n_boot: int = 10_000,
    level: float = 0.95,
    seed: int = 0,
) -> Interval:
    """Percentile bootstrap over test images for a single model."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    gen = rng(seed)
    n = len(y_true)
    stats_ = np.empty(n_boot)
    for b in range(n_boot):
        idx = gen.integers(0, n, size=n)
        stats_[b] = metric(y_true[idx], y_pred[idx])
    lo, hi = np.percentile(stats_, [(1 - level) / 2 * 100, (1 + level) / 2 * 100])
    return Interval(float(metric(y_true, y_pred)), float(lo), float(hi), level)


@dataclass
class PairedResult:
    delta: Interval
    p_value: float
    n_boot: int

    def to_dict(self) -> dict:
        return {"delta": self.delta.to_dict(), "p_value": self.p_value, "n_boot": self.n_boot}


def paired_bootstrap(
    y_true: np.ndarray,
    y_pred_a: np.ndarray,
    y_pred_b: np.ndarray,
    metric: Callable = balanced_accuracy,
    n_boot: int = 10_000,
    level: float = 0.95,
    seed: int = 0,
) -> PairedResult:
    """Compare two models on the same test set, resampling *images*, not predictions.

    Pairing on image removes the shared difficulty of the test set from the
    comparison; treating the two models as independent samples would inflate the
    interval enough to hide the effects this study is powered for.
    """
    y_true = np.asarray(y_true)
    a = np.asarray(y_pred_a)
    b = np.asarray(y_pred_b)
    if not (len(y_true) == len(a) == len(b)):
        raise ValueError("paired bootstrap requires aligned predictions")

    gen = rng(seed)
    n = len(y_true)
    deltas = np.empty(n_boot)
    for i in range(n_boot):
        idx = gen.integers(0, n, size=n)
        deltas[i] = metric(y_true[idx], a[idx]) - metric(y_true[idx], b[idx])

    observed = float(metric(y_true, a) - metric(y_true, b))
    lo, hi = np.percentile(deltas, [(1 - level) / 2 * 100, (1 + level) / 2 * 100])
    # Two-sided bootstrap p: how often the resampled difference crosses zero.
    p = 2.0 * min((deltas <= 0).mean(), (deltas >= 0).mean())
    return PairedResult(
        delta=Interval(observed, float(lo), float(hi), level),
        p_value=float(min(1.0, p)),
        n_boot=n_boot,
    )


def holm_bonferroni(p_values: dict[str, float], alpha: float = 0.05) -> dict[str, dict]:
    """Holm step-down over the pre-declared family of comparisons (protocol §9).

    The family is fixed before results are seen; adding comparisons afterwards and
    re-running this is exactly the thing it is meant to prevent.
    """
    items = sorted(p_values.items(), key=lambda kv: kv[1])
    m = len(items)
    out: dict[str, dict] = {}
    still_rejecting = True
    for rank, (name, p) in enumerate(items):
        threshold = alpha / (m - rank)
        if still_rejecting and p <= threshold:
            reject = True
        else:
            reject = False
            still_rejecting = False
        out[name] = {
            "p": p,
            "threshold": threshold,
            "reject": reject,
            "adjusted_p": min(1.0, max(p * (m - rank), 0.0)),
        }
    return out


@dataclass
class TrendResult:
    slope: Interval
    p_value: float
    per_seed_slopes: list[float]

    def to_dict(self) -> dict:
        return {
            "slope": self.slope.to_dict(),
            "p_value": self.p_value,
            "per_seed_slopes": self.per_seed_slopes,
        }


def trend_test(
    fractions: Sequence[float],
    scores_by_seed: dict[int, Sequence[float]],
    level: float = 0.95,
) -> TrendResult:
    """Test H2: does performance decline with synthetic fraction?

    Implemented as a summary-statistics random-intercept model: fit the slope
    separately within each seed, then do a one-sample t-test on those slopes. This
    is equivalent to a random-intercept mixed model for a balanced design, and it
    avoids pulling in a heavier dependency for a four-point regression.
    """
    x = np.asarray(fractions, dtype=float)
    slopes: list[float] = []
    for seed, scores in sorted(scores_by_seed.items()):
        y = np.asarray(scores, dtype=float)
        if len(y) != len(x):
            raise ValueError(f"seed {seed}: {len(y)} scores for {len(x)} fractions")
        mask = np.isfinite(y)
        if mask.sum() < 2:
            continue
        slopes.append(float(np.polyfit(x[mask], y[mask], 1)[0]))

    if not slopes:
        nan = float("nan")
        return TrendResult(Interval(nan, nan, nan, level), nan, [])

    ci = mean_ci(slopes, level)
    if len(slopes) > 1:
        p = float(sps.ttest_1samp(slopes, 0.0).pvalue)
    else:
        p = float("nan")
    return TrendResult(slope=ci, p_value=p, per_seed_slopes=slopes)


def exchange_rate(
    multipliers: Sequence[float],
    synthetic_scores: Sequence[float],
    target_score: float,
) -> float:
    """How many synthetic images per real image are needed to reach `target_score`.

    `synthetic_scores[i]` is performance at real budget n plus `multipliers[i]` x n
    synthetic images; `target_score` is real-only performance at the next real budget
    up. Returns the interpolated multiplier, or `inf` when the synthetic curve never
    reaches the target — which is a result, not a failure, and is the outcome H3
    predicts.
    """
    k = np.asarray(multipliers, dtype=float)
    y = np.asarray(synthetic_scores, dtype=float)
    order = np.argsort(k)
    k, y = k[order], y[order]

    if np.all(y < target_score):
        return float("inf")
    hits = np.where(y >= target_score)[0]
    first = int(hits[0])
    if first == 0:
        return float(k[0])
    x0, x1 = k[first - 1], k[first]
    y0, y1 = y[first - 1], y[first]
    if y1 == y0:
        return float(x1)
    return float(x0 + (target_score - y0) * (x1 - x0) / (y1 - y0))
