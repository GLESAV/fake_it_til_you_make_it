#!/usr/bin/env python
"""Why the stage-1 gate cannot answer its question: the scorer does not survive the move.

The gate applies the real-trained grade scorer to generated images and reads Continuity and
Scope off its output. That is only a measurement of severity rendering if the scorer means
the same thing on every substrate. On the wide-domain pool it does not, and the pool's own
design makes that checkable rather than arguable.

The design rule in generate_wide_domain.py randomises substrate independently of severity,
precisely so the pool cannot carry the provenance confound this project documented in
ACNE04. Refusals are strongly substrate-dependent and could have broken that balance, so
this script tests it first. If requested severity is still balanced across substrates, then
comparing mean scorer output across substrates holds severity fixed by construction, and any
difference is a substrate effect rather than a severity effect.

    python scripts/scorer_substrate_shift.py
"""

from __future__ import annotations

import argparse
import collections
import json
import re
from pathlib import Path

import numpy as np
import torch
from scipy.stats import chi2_contingency

from fitymi.data.records import NUM_CLASSES, Corpus, Record, Source
from fitymi.data.torchds import make_loader
from fitymi.train.loop import TrainConfig, predict
from fitymi.train.models import build_model


def concordance(pair_mask: np.ndarray, y: np.ndarray, pred: np.ndarray):
    """Continuity restricted to a chosen set of pairs. Ties score a half, as in the gate."""
    dy = np.sign(y[:, None] - y[None, :])
    dp = np.sign(pred[:, None] - pred[None, :])
    m = pair_mask & (dy != 0)
    if not m.any():
        return float("nan"), 0
    agree = (dy == dp)[m].astype(float)
    ties = (dp[m] == 0).astype(float)
    return 100 * float((agree + 0.5 * ties * (1 - agree)).mean()), int(m.sum())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default="data/synthetic/wide_pool")
    ap.add_argument("--scorer", default="models/grade_scorer.pt")
    ap.add_argument("--min-n", type=int, default=10)
    ap.add_argument("--out", default="results/scorer_substrate_shift.json")
    args = ap.parse_args()

    pool = Path(args.pool)
    meta = json.loads((pool / "prompt_meta.json").read_text())
    files = [(p, m) for p in sorted(pool.glob("*.png"))
             if (m := re.match(r"g(\d)_", p.name)) and p.stem in meta]
    if not files:
        raise SystemExit(f"no scored-able images in {pool}")
    recs = [Record(path=str(p), label=int(m.group(1)), source=Source.SYNTH_OPEN)
            for p, m in files]
    names = [p.stem for p, _ in files]

    tc = TrainConfig(num_workers=0, device="cpu")
    ck = torch.load(args.scorer, map_location="cpu")
    model = build_model(tc.arch, tc.init, NUM_CLASSES)
    model.load_state_dict(ck["state_dict"])
    model.eval()
    y, pred, _ = predict(model, make_loader(Corpus(recs), 32, tc.image_size, False, 0, 0),
                         torch.device("cpu"))
    y, pred = np.asarray(y), np.asarray(pred)
    person = np.array([meta[n]["needs_person"] for n in names])
    sub = np.array([meta[n]["substrate"] for n in names])

    # ---- 1. did the design rule survive the refusals?
    print(f"{len(y)} images. First: is requested severity still balanced across the "
          f"substrate split?\n")
    tab = np.array([[int(((y == g) & (person == w)).sum()) for g in range(NUM_CLASSES)]
                    for w in (True, False)])
    chi2, p_split, dof, _ = chi2_contingency(tab)
    for row, label in zip(tab, ("on a person", "NOT on a person")):
        tot = row.sum()
        print(f"  {label:>16} n={tot:>3}  " + "  ".join(
            f"g{g}:{row[g]:>3} ({100 * row[g] / tot:4.1f}%)" for g in range(NUM_CLASSES)))
    print(f"  chi2 {chi2:.2f}, dof {dof}, p {p_split:.3f} -> requested grade is "
          f"{'independent of' if p_split >= 0.05 else 'ASSOCIATED WITH'} the split")

    big = sorted({s for s in sub if (sub == s).sum() >= args.min_n})
    tab2 = np.array([[int(((y == g) & (sub == s)).sum()) for g in range(NUM_CLASSES)]
                     for s in big])
    chi2b, p_sub, dofb, _ = chi2_contingency(tab2)
    print(f"  across the {len(big)} substrates with n>={args.min_n}: chi2 {chi2b:.2f}, "
          f"dof {dofb}, p {p_sub:.3f}")
    if p_split < 0.05 or p_sub < 0.05:
        print("\n  BALANCE BROKEN -- the comparison below is confounded and must not be "
              "read as a substrate effect.")
    else:
        print("\n  Balance holds, so severity is fixed by construction below and any "
              "difference is substrate.")

    # ---- 2. what the scorer outputs, by substrate
    print(f"\nMean scorer output per substrate, requested severity balanced:")
    print(f"{'substrate':>26} {'person':>7} {'mean':>6} {'n':>4}")
    rows = []
    for s in sorted({*sub}, key=lambda s: -pred[sub == s].mean()):
        idx = sub == s
        pers = bool(person[idx][0])
        rows.append({"substrate": s, "needs_person": pers,
                     "mean_pred": float(pred[idx].mean()), "n": int(idx.sum())})
        print(f"{s[:26]:>26} {'person' if pers else '':>7} "
              f"{pred[idx].mean():>6.2f} {int(idx.sum()):>4}")

    print(f"\nPredicted-class distribution, ignoring what was requested:")
    for label, mask in (("on a person", person), ("NOT on a person", ~person)):
        c = collections.Counter(pred[mask].tolist())
        tot = int(mask.sum())
        print(f"  {label:>16}: " + "  ".join(
            f"{k}:{100 * c[k] / tot:4.0f}%" for k in range(NUM_CLASSES)))

    print(f"\nMean output per requested grade, and the gap between the two groups:")
    print(f"{'grade':>6} {'person':>9} {'no-person':>11} {'offset':>8}")
    offsets = {}
    for g in range(NUM_CLASSES):
        a, b = pred[(y == g) & person], pred[(y == g) & ~person]
        if len(a) and len(b):
            offsets[g] = float(b.mean() - a.mean())
            print(f"{g:>6} {a.mean():>9.3f} {b.mean():>11.3f} {offsets[g]:>+8.3f}")

    # ---- 3. is the pooled Continuity drop the cross-substrate pairs?
    n = len(y)
    iu = np.triu(np.ones((n, n), bool), 1)
    same = person[:, None] == person[None, :]
    print(f"\nContinuity, split by whether a pair crosses the substrate divide:")
    conts = {}
    for label, mask in (("all pairs", iu),
                        ("within person", iu & same & person[:, None]),
                        ("within no-person", iu & same & ~person[:, None]),
                        ("within either", iu & same),
                        ("ACROSS the divide", iu & ~same)):
        c, k = concordance(mask, y, pred)
        conts[label] = {"continuity": c, "pairs": k}
        print(f"  {label:>20}  {c:>5.1f}%  ({k} pairs)")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({
        "n": int(n), "balance_p_person_split": float(p_split),
        "balance_p_substrate": float(p_sub),
        "per_substrate": rows, "offsets_by_grade": offsets, "continuity": conts,
    }, indent=1))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
