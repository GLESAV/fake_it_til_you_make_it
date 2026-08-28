#!/usr/bin/env python
"""Is the stage-1 gate's comparison fair? Scope is a range statistic, and ranges are biased.

The kill criterion compares the wide-domain pool's Scope against the face-only pool's
36.3%. Scope is

    (max_c mean(pred | grade c) - min_c mean(pred | grade c)) / (K - 1)

-- the range of four sample means. The maximum of noisy sample means is biased upward and
the minimum downward, and both biases grow as the number of images per grade falls. The
face-only pool has 644 images, roughly 161 per grade. The wide pool at the gate's threshold
has 240, roughly 60. So the two numbers being compared are not on the same footing, and the
wide pool's would be expected to come out higher even if the two pools were identical.

This script measures the size of that bias instead of arguing about it: it scores the
face-only pool with the same real-trained scorer, then repeatedly subsamples it down to the
wide pool's per-grade counts and reports the Scope distribution. If the subsampled face pool
reaches the wide pool's Scope on its own, the gate's apparent gain is the estimator and not
the substrates.

    python scripts/scope_smalln_bias.py --draws 2000
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import torch

from fitymi.data.records import NUM_CLASSES, Corpus, Record, Source
from fitymi.data.torchds import make_loader
from fitymi.train.loop import TrainConfig, predict
from fitymi.train.models import build_model


def scope(y: np.ndarray, pred: np.ndarray) -> float:
    means = [pred[y == c].mean() for c in range(NUM_CLASSES) if (y == c).any()]
    return 100 * (max(means) - min(means)) / (NUM_CLASSES - 1) if len(means) > 1 else np.nan


def continuity(y: np.ndarray, pred: np.ndarray) -> float:
    """Vectorised form of substrate_fidelity.continuity_and_scope's first statistic.

    Identical by construction -- ties score a half -- but O(n^2) in numpy rather than in
    Python, because the resampling below needs it thousands of times.
    """
    dy = np.sign(y[:, None] - y[None, :])
    dp = np.sign(pred[:, None] - pred[None, :])
    mask = np.triu(dy != 0, k=1)
    if not mask.any():
        return np.nan
    agree = (dy == dp)[mask].astype(float)
    ties = (dp[mask] == 0).astype(float)
    return 100 * float((agree + 0.5 * ties * (1 - agree)).mean())


def score_pool(pool: Path, scorer: str):
    recs = [Record(path=str(p), label=int(m.group(1)), source=Source.SYNTH_OPEN)
            for p in sorted(pool.glob("*.png")) if (m := re.match(r"g(\d)_", p.name))]
    if not recs:
        raise SystemExit(f"no gN_ images in {pool}")
    tc = TrainConfig(num_workers=0, device="cpu")
    ck = torch.load(scorer, map_location="cpu")
    model = build_model(tc.arch, tc.init, NUM_CLASSES)
    model.load_state_dict(ck["state_dict"])
    model.eval()
    y, pred, _ = predict(model, make_loader(Corpus(recs), 32, tc.image_size, False, 0, 0),
                         torch.device("cpu"))
    return np.asarray(y), np.asarray(pred)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--face-pool", default="data/synthetic/gemini_pool")
    ap.add_argument("--wide-pool", default="data/synthetic/wide_pool")
    ap.add_argument("--scorer", default="models/grade_scorer.pt")
    ap.add_argument("--draws", type=int, default=2000)
    ap.add_argument("--cont-draws", type=int, default=500,
                    help="Continuity is O(n^2) per draw, so it uses a smaller sample")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/scope_smalln_bias.json")
    args = ap.parse_args()

    yf, pf = score_pool(Path(args.face_pool), args.scorer)
    yw, pw = score_pool(Path(args.wide_pool), args.scorer)
    face_full, wide_full = scope(yf, pf), scope(yw, pw)
    counts = {c: int((yw == c).sum()) for c in range(NUM_CLASSES)}
    face_counts = {c: int((yf == c).sum()) for c in range(NUM_CLASSES)}

    print(f"face-only pool  n={len(yf)} {face_counts}  Scope {face_full:.1f}%  "
          f"Continuity {continuity(yf, pf):.1f}%")
    print(f"wide pool       n={len(yw)} {counts}  Scope {wide_full:.1f}%  "
          f"Continuity {continuity(yw, pw):.1f}%\n")

    rng = np.random.default_rng(args.seed)
    idx_by_class = {c: np.flatnonzero(yf == c) for c in range(NUM_CLASSES)}
    draws, cont_draws = [], []
    for i in range(args.draws):
        take = []
        for c, n in counts.items():
            pool_c = idx_by_class[c]
            if len(pool_c) == 0 or n == 0:
                continue
            take.append(rng.choice(pool_c, size=min(n, len(pool_c)), replace=False))
        sub = np.concatenate(take)
        draws.append(scope(yf[sub], pf[sub]))
        if i < args.cont_draws:
            cont_draws.append(continuity(yf[sub], pf[sub]))
    draws, cont_draws = np.array(draws), np.array(cont_draws)

    lo, hi = np.percentile(draws, [2.5, 97.5])
    print(f"face-only pool subsampled to the wide pool's per-grade counts, "
          f"{args.draws} draws:")
    print(f"  Scope  mean {draws.mean():.1f}%   95% range [{lo:.1f}, {hi:.1f}]   "
          f"vs {face_full:.1f}% at full size")
    print(f"  upward bias from small n alone: {draws.mean() - face_full:+.1f} points")
    share = float((draws >= wide_full).mean())
    print(f"  share of subsamples reaching the wide pool's {wide_full:.1f}%: "
          f"{100 * share:.1f}%")

    wide_cont, face_cont = continuity(yw, pw), continuity(yf, pf)
    clo, chi = np.percentile(cont_draws, [2.5, 97.5])
    cshare = float((cont_draws <= wide_cont).mean())
    print(f"  Continuity  mean {cont_draws.mean():.1f}%   95% range [{clo:.1f}, {chi:.1f}]"
          f"   vs {face_cont:.1f}% at full size ({len(cont_draws)} draws)")
    print(f"  share of subsamples falling to the wide pool's {wide_cont:.1f}%: "
          f"{100 * cshare:.1f}%")
    print()
    if share > 0.05:
        print("VERDICT: the wide pool's Scope is inside what the face-only pool produces by")
        print("chance at the same sample size. The gate's comparison does not survive it,")
        print("and any 'wider Scope' claim needs the face number recomputed at matched n.")
    else:
        print("VERDICT: the wide pool's Scope is above what small-n bias alone explains.")
        print("The comparison stands, though the face baseline should still be quoted at")
        print("matched n rather than at 644.")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "face_pool_n": int(len(yf)), "face_counts": face_counts,
        "face_scope_full": float(face_full),
        "wide_pool_n": int(len(yw)), "wide_counts": counts,
        "wide_scope": float(wide_full),
        "draws": args.draws,
        "subsampled_face_scope_mean": float(draws.mean()),
        "subsampled_face_scope_ci95": [float(lo), float(hi)],
        "bias_points": float(draws.mean() - face_full),
        "share_of_draws_reaching_wide": share,
        "face_continuity_full": float(face_cont),
        "wide_continuity": float(wide_cont),
        "subsampled_face_continuity_mean": float(cont_draws.mean()),
        "subsampled_face_continuity_ci95": [float(clo), float(chi)],
        "share_of_draws_falling_to_wide_continuity": cshare,
    }, indent=1))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
