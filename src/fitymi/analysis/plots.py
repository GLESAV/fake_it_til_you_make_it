"""Figures.

Three plots carry the paper: the substitution curve (H1/H2), the additive curves from
which the exchange rate is read (H3), and the per-class tail, which is where a
generator's mode-dropping becomes visible.

Deliberately plain matplotlib with no seaborn dependency and no style file, so the
figures render identically on a headless GPU host.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

# Colour-blind safe, and distinguishable in greyscale print.
GENERATOR_COLOURS = {
    "synth_closed": "#0072B2",
    "synth_open": "#D55E00",
    "synth_edit": "#009E73",
    "real_subset": "#666666",
}


def _series_colour(name: str) -> str:
    return GENERATOR_COLOURS.get(str(name), "#333333")


def substitution_curve(
    curve: pd.DataFrame,
    out_path: str | Path,
    metric_label: str = "Balanced accuracy",
    real_baseline: float | None = None,
) -> Path:
    """Performance against synthetic fraction, one line per generator/init cell."""
    fig, ax = plt.subplots(figsize=(6.0, 4.0), dpi=200)

    for (generator, init), group in curve.groupby(["generator", "init"], dropna=False):
        group = group.sort_values("synthetic_fraction")
        x = group["synthetic_fraction"].to_numpy(dtype=float) * 100
        y = group["mean"].to_numpy(dtype=float)
        lo = group["ci_lo"].to_numpy(dtype=float)
        hi = group["ci_hi"].to_numpy(dtype=float)
        colour = _series_colour(generator)
        style = "-" if init == "scratch" else "--"
        ax.plot(x, y, style, color=colour, marker="o", markersize=4,
                label=f"{generator} ({init})")
        finite = np.isfinite(lo) & np.isfinite(hi)
        if finite.any():
            ax.fill_between(x[finite], lo[finite], hi[finite], color=colour, alpha=0.15, linewidth=0)

    if real_baseline is not None:
        ax.axhline(real_baseline, color="black", linewidth=0.8, linestyle=":",
                   label="real-only baseline")

    ax.set_xlabel("Synthetic fraction of the training set (%)")
    ax.set_ylabel(metric_label)
    ax.set_title("Substitution at matched training budget")
    ax.grid(alpha=0.25, linewidth=0.5)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def additive_curves(
    runs: pd.DataFrame,
    out_path: str | Path,
    metric: str = "balanced_accuracy",
    metric_label: str = "Balanced accuracy",
) -> Path:
    """One panel per real budget: performance against synthetic multiplier k."""
    from .aggregate import expand_shared_arms

    add = expand_shared_arms(runs[runs["kind"] == "additive"] if "kind" in runs else runs)
    budgets = sorted(add["real_budget_fraction"].dropna().unique())
    if not budgets:
        raise ValueError("no additive runs to plot")

    fig, axes = plt.subplots(
        1, len(budgets), figsize=(3.0 * len(budgets), 3.2), dpi=200, sharey=True
    )
    axes = np.atleast_1d(axes)

    for ax, budget in zip(axes, budgets):
        cell = add[add["real_budget_fraction"] == budget]
        for generator, group in cell.groupby("generator", dropna=False):
            curve = group.groupby("synthetic_multiplier")[metric].agg(["mean", "std", "count"])
            curve = curve.sort_index()
            x = curve.index.to_numpy(dtype=float)
            y = curve["mean"].to_numpy(dtype=float)
            sem = (curve["std"] / np.sqrt(curve["count"].clip(lower=1))).to_numpy(dtype=float)
            colour = _series_colour(generator)
            ax.plot(x, y, marker="o", markersize=4, color=colour, label=str(generator))
            ok = np.isfinite(sem)
            if ok.any():
                ax.fill_between(x[ok], (y - sem)[ok], (y + sem)[ok], color=colour, alpha=0.15,
                                linewidth=0)
        ax.set_title(f"real budget = {budget:.0%}", fontsize=9)
        ax.set_xlabel("synthetic images per real image (k)")
        ax.grid(alpha=0.25, linewidth=0.5)

    axes[0].set_ylabel(metric_label)
    axes[-1].legend(frameon=False, fontsize=8)
    fig.suptitle("Additive scaling: what synthetic data buys at each real budget", fontsize=10)
    fig.tight_layout()

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def tail_recall(runs: pd.DataFrame, out_path: str | Path) -> Path:
    """Severe + very-severe recall against synthetic fraction.

    The overall metric can hold up while the rare grades collapse; that is exactly the
    late-collapse signature, and it is the part with clinical consequences.
    """
    from .aggregate import expand_shared_arms

    sub = expand_shared_arms(
        runs[runs["kind"] == "substitution"] if "kind" in runs else runs
    )
    fig, ax = plt.subplots(figsize=(6.0, 4.0), dpi=200)

    for generator, group in sub.groupby("generator", dropna=False):
        curve = group.groupby("synthetic_fraction")[["balanced_accuracy", "tail_recall"]].mean()
        curve = curve.sort_index()
        x = curve.index.to_numpy(dtype=float) * 100
        colour = _series_colour(generator)
        ax.plot(x, curve["balanced_accuracy"], "-", color=colour, marker="o", markersize=4,
                label=f"{generator}: overall")
        ax.plot(x, curve["tail_recall"], "--", color=colour, marker="s", markersize=4,
                alpha=0.75, label=f"{generator}: severe + very severe")

    ax.set_xlabel("Synthetic fraction of the training set (%)")
    ax.set_ylabel("Recall")
    ax.set_title("Overall performance vs. the severe tail")
    ax.grid(alpha=0.25, linewidth=0.5)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    return out_path
