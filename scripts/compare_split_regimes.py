#!/usr/bin/env python
"""What is subject leakage worth, in points?

Two arms differing in exactly one thing: how the splits were drawn. Same classifier,
same hyperparameters, same training budget, same seeds, same corpus. One splits at the
image level after perceptual and CLIP deduplication -- which is what every published
ACNE04 result does, and what our own first attempt did -- and carries 74.5% subject
leakage. The other groups by face identity first and carries none.

The difference between them is the leakage bonus. It is the number the rest of the
ACNE04 literature needs, and it costs one extra sweep to measure.

    python scripts/compare_split_regimes.py \
        --leaky runs/acne04_baseline --clean runs/acne04_baseline_subject

Reports paired differences by seed, since the seeds are matched -- pairing removes the
seed-to-seed variance that would otherwise swamp a few points of effect.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

METRICS = ("balanced_accuracy", "accuracy", "macro_f1", "qwk", "mae")


def load_runs(directory: str) -> dict[int, dict]:
    """Map seed -> validation metrics for every run record under `directory`."""
    out: dict[int, dict] = {}
    for path in sorted(Path(directory).rglob("*.json")):
        if path.name in {"substitution_curve.json", "h2_trend.json"}:
            continue
        try:
            record = json.loads(path.read_text())
        except json.JSONDecodeError:
            continue
        if "val" not in record or not isinstance(record.get("val"), dict):
            continue
        seed = record.get("config", {}).get("seed")
        if seed is None:
            match = re.search(r"_s(\d+)_", record.get("run_id", ""))
            seed = int(match.group(1)) if match else None
        if seed is None:
            continue
        out[int(seed)] = record["val"]
    return out


def bootstrap_ci(values: np.ndarray, n: int = 20000, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    draws = rng.choice(values, size=(n, len(values)), replace=True).mean(axis=1)
    return float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--leaky", default="runs/acne04_baseline",
                    help="image-level splits (carries subject leakage)")
    ap.add_argument("--clean", default="runs/acne04_baseline_subject",
                    help="subject-disjoint splits")
    args = ap.parse_args()

    leaky, clean = load_runs(args.leaky), load_runs(args.clean)
    seeds = sorted(set(leaky) & set(clean))
    if not seeds:
        raise SystemExit(
            f"no matched seeds: {sorted(leaky)} in {args.leaky}, {sorted(clean)} in {args.clean}"
        )
    print(f"matched seeds: {seeds}")
    if set(leaky) != set(clean):
        print(f"  note: unmatched seeds ignored -- "
              f"leaky-only {sorted(set(leaky) - set(clean))}, "
              f"clean-only {sorted(set(clean) - set(leaky))}")

    print(f"\n{'metric':>20} {'image-level':>14} {'subject-disjoint':>18} "
          f"{'difference':>12} {'95% CI':>20}")
    for metric in METRICS:
        a = np.array([leaky[s][metric] for s in seeds])
        b = np.array([clean[s][metric] for s in seeds])
        diff = a - b
        lo, hi = bootstrap_ci(diff) if len(seeds) > 1 else (float("nan"), float("nan"))
        print(f"{metric:>20} {a.mean():>8.4f} ±{a.std(ddof=1) if len(a) > 1 else 0:>4.3f} "
              f"{b.mean():>12.4f} ±{b.std(ddof=1) if len(b) > 1 else 0:>4.3f} "
              f"{diff.mean():>+12.4f} {f'[{lo:+.4f}, {hi:+.4f}]':>20}")

    print("\nper-seed balanced accuracy")
    print(f"{'seed':>6} {'image-level':>13} {'subject-disjoint':>18} {'difference':>12}")
    for s in seeds:
        a, b = leaky[s]["balanced_accuracy"], clean[s]["balanced_accuracy"]
        print(f"{s:>6} {a:>13.4f} {b:>18.4f} {a - b:>+12.4f}")

    print("\nper-class recall (mean across seeds)")
    classes = list(leaky[seeds[0]].get("per_class_recall", {}))
    if classes:
        print(f"{'grade':>14} {'image-level':>13} {'subject-disjoint':>18} {'difference':>12}")
        for name in classes:
            a = np.mean([leaky[s]["per_class_recall"][name] for s in seeds])
            b = np.mean([clean[s]["per_class_recall"][name] for s in seeds])
            print(f"{name:>14} {a:>13.4f} {b:>18.4f} {a - b:>+12.4f}")

    delta = np.array([leaky[s]["balanced_accuracy"] - clean[s]["balanced_accuracy"]
                      for s in seeds])
    lo, hi = bootstrap_ci(delta) if len(seeds) > 1 else (float("nan"), float("nan"))
    print(f"\nLEAKAGE BONUS on balanced accuracy: {100 * delta.mean():+.2f} points "
          f"(95% CI {100 * lo:+.2f} to {100 * hi:+.2f})")
    if len(seeds) > 1 and lo <= 0 <= hi:
        print("  The interval spans zero: on this metric and this many seeds, the leakage")
        print("  bonus is not distinguishable from noise. Report the bound, not an effect.")


if __name__ == "__main__":
    main()
