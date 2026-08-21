#!/usr/bin/env python
"""Stage-1 gate: does moving acne off the face render severity any better?

The cheapest decisive measurement in the wide-domain plan, and the one that decides whether
the rest is worth paying for. It needs no training -- just the real-trained grade scorer
applied to generated images -- so it can run on a few hundred images and answer the question
the whole premise rests on.

The face-only pool scores Continuity 70.4% and Scope 36.3% (docs/07 section 12.15): severity
is monotone but four clinical grades come out as roughly one grade of real variation. The
hypothesis is that the face prior causes it -- a model that has seen millions of portraits
returns someone who still looks like a person when asked for confluent nodulocystic disease.
If that is right, substrates with no person in them should score a wider Scope.

Reported per substrate group, because the answer may be that some framings work and others
do not, and "Petri dishes render severity but silicone models do not" is a more useful
result than a single pooled number.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from fitymi.data.records import NUM_CLASSES, Corpus, Record, Source
from fitymi.data.torchds import make_loader
from fitymi.train.loop import TrainConfig, predict
from fitymi.train.models import build_model


def continuity_and_scope(y: np.ndarray, pred: np.ndarray) -> tuple[float, float, int]:
    """CompSlider's two statistics: a monotonicity rate and a dynamic range."""
    conc = tot = 0.0
    for a in range(len(y)):
        for b in range(a + 1, len(y)):
            if y[a] == y[b]:
                continue
            tot += 1
            hi, lo = (a, b) if y[a] > y[b] else (b, a)
            conc += 1.0 if pred[hi] > pred[lo] else (0.5 if pred[hi] == pred[lo] else 0.0)
    means = [pred[y == c].mean() for c in range(NUM_CLASSES) if (y == c).any()]
    scope = (max(means) - min(means)) / (NUM_CLASSES - 1) if len(means) > 1 else float("nan")
    return (100 * conc / tot if tot else float("nan")), 100 * scope, len(y)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default="data/synthetic/wide_pool")
    ap.add_argument("--scorer", default="models/grade_scorer.pt")
    ap.add_argument("--min-per-group", type=int, default=24)
    args = ap.parse_args()

    meta = json.loads((Path(args.pool) / "prompt_meta.json").read_text())
    recs, groups = [], []
    for p in sorted(Path(args.pool).glob("*.png")):
        m = re.match(r"g(\d)_s(\d+)_", p.name)
        if not m or p.stem not in meta:
            continue
        recs.append(Record(path=str(p), label=int(m.group(1)), source=Source.SYNTH_OPEN))
        groups.append(meta[p.stem])
    if len(recs) < 40:
        raise SystemExit(f"only {len(recs)} images so far; wait for more")

    tc = TrainConfig(num_workers=0, device="cpu")
    ck = torch.load(args.scorer, map_location="cpu")
    model = build_model(tc.arch, tc.init, NUM_CLASSES)
    model.load_state_dict(ck["state_dict"])
    model.eval()
    y, pred, _ = predict(model, make_loader(Corpus(recs), 32, tc.image_size, False, 0, 0),
                         torch.device("cpu"))

    print(f"{len(recs)} wide-domain images scored by the real-trained classifier\n")
    c, s, n = continuity_and_scope(y, pred)
    print(f"{'ALL wide-domain':>34}  Continuity {c:>5.1f}%  Scope {s:>5.1f}%  n={n}")
    on_person = np.array([g["needs_person"] for g in groups])
    for label, mask in (("on a person", on_person), ("NOT on a person", ~on_person)):
        if mask.sum() >= args.min_per_group:
            c, s, n = continuity_and_scope(y[mask], pred[mask])
            print(f"{label:>34}  Continuity {c:>5.1f}%  Scope {s:>5.1f}%  n={n}")

    print(f"\n{'per substrate':>34}")
    by_sub = defaultdict(list)
    for i, g in enumerate(groups):
        by_sub[g["substrate"]].append(i)
    rows = []
    for name, idx in by_sub.items():
        if len(idx) < args.min_per_group:
            continue
        idx = np.array(idx)
        c, s, n = continuity_and_scope(y[idx], pred[idx])
        rows.append((s, c, n, name))
    for s, c, n, name in sorted(rows, reverse=True):
        print(f"{name[:34]:>34}  Continuity {c:>5.1f}%  Scope {s:>5.1f}%  n={n}")
    if not rows:
        print(f"{'(no substrate has ' + str(args.min_per_group) + ' images yet)':>34}")

    print(f"\n{'face-only pool, for comparison':>34}  Continuity  70.4%  Scope  36.3%")
    print(f"{'real images, same scorer':>34}  Continuity  88.4%  Scope  93.1%")
    print("\nGATE: if no substrate group beats Scope 36.3%, the face prior is not the")
    print("cause of compression and the wide-domain premise fails. Stop and say so.")


if __name__ == "__main__":
    main()
