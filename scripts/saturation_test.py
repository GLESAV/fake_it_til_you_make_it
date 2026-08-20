#!/usr/bin/env python
"""Does the generator's severity saturate, and where?

The coverage arm's value proposition is that it can supply what the real data lacks. ACNE04
lacks two things: demographic breadth, and severe cases -- 129 very-severe images from 81
people. If the generator can supply the first but not the second, that is a sharp and
useful limit, because the severity tail is the clinically load-bearing part.

This measures the transfer curve: requested severity in, predicted severity out, using the
real-data classifier. A generator with full range gives a line; one that saturates gives a
curve that flattens at the top, and the flattening point is the ceiling of what it can
usefully produce.

    python scripts/saturation_test.py --pool data/synthetic/gemini_v2 --pool data/synthetic/gemini_calib
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))


def main():
    from fitymi.data.records import Corpus, Record, Source
    from fitymi.train.loop import evaluate_corpus
    from grade_fidelity import load_scorer, spearman

    ap = argparse.ArgumentParser()
    ap.add_argument("--scorer", default="models/grade_scorer.pt")
    ap.add_argument("--pool", action="append", default=[])
    ap.add_argument("--splits", default="data/splits_subject")
    args = ap.parse_args()

    model, cfg = load_scorer(args.scorer)
    cfg.num_workers = 0

    def transfer(paths, labels, name):
        if len(paths) < 8:
            print(f"  {name}: only {len(paths)} images, skipping")
            return
        corpus = Corpus(Record(path=p, label=l, source=Source.SYNTH_OPEN)
                        for p, l in zip(paths, labels))
        _, pr = evaluate_corpus(model, corpus, cfg, return_predictions=True)
        yt, yp = np.array(pr["y_true"]), np.array(pr["y_pred"])
        print(f"\n  {name}  (n={len(yt)}, Spearman {spearman(yt.astype(float), yp.astype(float)):+.3f})")
        print(f"    {'requested':>10} {'n':>4} {'mean predicted':>15} {'range':>16}")
        for g in sorted(set(yt.tolist())):
            m = yt == g
            hist = dict(sorted(Counter(yp[m].tolist()).items()))
            print(f"    {g:>10} {int(m.sum()):>4} {float(yp[m].mean()):>15.2f}   {hist}")
        # The flattening point: the first requested grade whose mean prediction fails to
        # exceed its predecessor's. That is where asking for more stops producing more.
        means = [float(yp[yt == g].mean()) for g in sorted(set(yt.tolist()))]
        grades = sorted(set(yt.tolist()))
        for i in range(1, len(means)):
            if means[i] <= means[i - 1] + 0.05:
                print(f"    -> saturates at requested grade {grades[i]}: "
                      f"mean {means[i]:.2f} vs {means[i-1]:.2f} at grade {grades[i-1]}")
                break
        else:
            print("    -> monotone across every requested grade, no saturation point")

    rows = [__import__("json").loads(l) for l in
            (Path(args.splits) / "val.jsonl").read_text().splitlines() if l.strip()]
    transfer([r["path"] for r in rows], [r["label"] for r in rows], "real held-out (ceiling)")

    for pool in args.pool:
        paths, labels = [], []
        for p in sorted(Path(pool).glob("*.png")):
            m = re.match(r"g(\d)_", p.name)
            if m:
                paths.append(str(p)); labels.append(int(m.group(1)))
        transfer(paths, labels, pool)


if __name__ == "__main__":
    main()
