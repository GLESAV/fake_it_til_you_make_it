#!/usr/bin/env python
"""Does the synthetic pool actually cover skin tones the real dataset does not?

The coverage arm's whole premise is that a prompted generator can supply breadth no real
acne dataset has. That has been asserted here from looking at a contact sheet, which is
exactly the kind of evidence this project has repeatedly shown to be unreliable. This
measures it.

Individual typology angle (ITA) from CIE L*a*b* on the central skin region, with the most
erythematous pixels dropped so lesions do not drag the estimate darker. ITA is a proxy for
constitutive pigmentation and is not a Fitzpatrick or Monk label; it is reported as a
distribution, and the comparison between two distributions measured the same way is far
more defensible than either number alone.

    python scripts/coverage_comparison.py --pool data/synthetic/gemini_pool
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np


def _entropy(counts: list[int]) -> float:
    """Shannon entropy of the bin distribution: how evenly the tone range is filled.

    A corpus concentrated in one bin scores near zero however central that bin is; one
    spread across every bin approaches log(n_bins). This is the quantity the coverage claim
    is actually about.
    """
    total = sum(counts)
    if total == 0:
        return 0.0
    p = np.array([c / total for c in counts if c > 0])
    return float(-(p * np.log(p)).sum())


def summarise(paths: list[str], label: str) -> dict:
    from fitymi.controls.skintone import COARSE, ITA_BINS, estimate_ita

    estimates = []
    for p in paths:
        try:
            estimates.append(estimate_ita(p))
        except Exception:  # noqa: BLE001 - a single unreadable file must not stop the audit
            continue
    itas = np.array([e.ita for e in estimates])
    fine = Counter(e.bin for e in estimates)
    coarse = Counter(e.coarse_bin for e in estimates)
    order = [name for name, _, _ in ITA_BINS]

    print(f"\n== {label}: {len(estimates)} images ==")
    print(f"  ITA median {np.median(itas):6.1f}   IQR {np.percentile(itas,25):.1f} "
          f"to {np.percentile(itas,75):.1f}   range {itas.min():.1f} to {itas.max():.1f}")
    print(f"  {'bin':>14} {'n':>5} {'share':>7}")
    for name in order:
        n = fine.get(name, 0)
        bar = "#" * int(40 * n / max(len(estimates), 1))
        print(f"  {name:>14} {n:>5} {100*n/max(len(estimates),1):>6.1f}% {bar}")
    return {
        "label": label, "n": len(estimates),
        "median_ita": float(np.median(itas)),
        "fine": dict(fine), "coarse": dict(coarse),
        # Two statistics, because the obvious one is misleading here. "Share outside the
        # lightest bins" makes a corpus concentrated in the MIDDLE look broad, which is
        # exactly backwards: ACNE04 scores 85% on it while holding 2.7% dark-skinned images.
        # What the coverage claim is about is representation at the underserved end, and
        # how evenly the range is filled.
        "share_two_darkest_bins": float(
            (fine.get("brown", 0) + fine.get("dark", 0)) / max(len(estimates), 1)),
        "share_darkest_bin": float(fine.get("dark", 0) / max(len(estimates), 1)),
        "bin_entropy": float(_entropy([fine.get(n, 0) for n in order])),
        "max_bin_entropy": float(np.log(len(order))),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", default="data/splits_subject")
    ap.add_argument("--pool", action="append", default=[])
    ap.add_argument("--limit", type=int, default=400)
    args = ap.parse_args()

    rows = [json.loads(l) for l in
            (Path(args.splits) / "train.jsonl").read_text().splitlines() if l.strip()]
    rng = np.random.default_rng(0)
    idx = rng.choice(len(rows), size=min(args.limit, len(rows)), replace=False)
    real = summarise([rows[i]["path"] for i in idx], "ACNE04 (real training split)")

    out = [real]
    for pool in args.pool:
        paths = sorted(str(p) for p in Path(pool).glob("*.png"))[: args.limit]
        if paths:
            out.append(summarise(paths, pool))

    if len(out) > 1:
        print("\n== the claim under test ==")
        print(f"  {'corpus':>34} {'2 darkest bins':>15} {'darkest bin':>12} {'evenness':>10}")
        for row in [real, *out[1:]]:
            print(f"  {row['label'][:34]:>34} {100*row['share_two_darkest_bins']:>14.1f}% "
                  f"{100*row['share_darkest_bin']:>11.1f}% "
                  f"{row['bin_entropy']/row['max_bin_entropy']:>10.2f}")
        print("  ITA is a proxy for pigmentation, not a Fitzpatrick label, and generated")
        print("  images have no ground truth. The defensible statement is about the two")
        print("  distributions measured identically, not about either in absolute terms.")

    Path("results").mkdir(exist_ok=True)
    Path("results/coverage_comparison.json").write_text(json.dumps(out, indent=2))
    print("\nwrote results/coverage_comparison.json")


if __name__ == "__main__":
    main()
