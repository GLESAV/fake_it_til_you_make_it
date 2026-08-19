#!/usr/bin/env python
"""Run the gating controls against a partially generated pool.

Generation takes 18 hours. Two of its failure modes are visible in the first few hundred
images and both invalidate the arm they feed:

- **Memorisation.** If the generator reproduces its training set, the synthetic-only arm is
  a laundered copy of the real one and the substitution question is not being measured.
- **Identity collapse.** If the pool depicts a handful of people, a substitution failure is
  about identity diversity rather than image quality -- a different and more tractable
  finding, and one that is invisible to FID and to every per-image quality metric.

Neither needs the pool finished, and finding either at 18 hours instead of at 20 minutes
costs a day.

    python scripts/pool_early_check.py --pool data/synthetic/closed --splits data/splits_subject
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from fitymi.controls.calibration import calibrate
from fitymi.controls.embedders import SSCD_THRESHOLD, cached_embedder, face_embedder, sscd_embedder
from fitymi.controls.identity_diversity import identity_diversity
from fitymi.data.records import Corpus, Record, Source


def load_split(splits_dir: str, name: str) -> list[str]:
    return [json.loads(line)["path"]
            for line in (Path(splits_dir) / f"{name}.jsonl").read_text().splitlines()
            if line.strip()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default="data/synthetic/closed")
    ap.add_argument("--splits", default="data/splits_subject")
    ap.add_argument("--device", default="cpu", help="cpu by default so it does not "
                                                    "contend with generation on the GPU")
    ap.add_argument("--limit", type=int, default=0, help="cap the pool sample")
    args = ap.parse_args()

    pool = sorted(str(p) for p in Path(args.pool).glob("*.png"))
    if args.limit:
        pool = pool[: args.limit]
    if len(pool) < 20:
        raise SystemExit(f"only {len(pool)} images in {args.pool}; wait for more")
    train = load_split(args.splits, "train")
    print(f"{len(pool)} generated images against {len(train)} real training images")

    # -- memorisation, at a threshold calibrated on the real corpus, not inherited -------
    sscd = cached_embedder(sscd_embedder(device=args.device), "data/splits/sscd_real.npz")
    real_cal = calibrate(train, sscd, target_fpr=0.01, mode="per_item",
                         reference_threshold=SSCD_THRESHOLD, max_component_fraction=None)
    threshold = real_cal.threshold
    print(f"\n== memorisation ==")
    print(f"threshold calibrated on the real training split: {threshold:.4f} "
          f"(the published 0.5 would flag {100 * real_cal.fpr_at_reference:.1f}% of real "
          f"images as copies of each other)")

    pool_emb = np.asarray(sscd_embedder(device=args.device)(pool), np.float32)
    pool_emb /= np.linalg.norm(pool_emb, axis=1, keepdims=True) + 1e-12
    real_emb = np.asarray(sscd(train), np.float32)
    real_emb /= np.linalg.norm(real_emb, axis=1, keepdims=True) + 1e-12
    best = (pool_emb @ real_emb.T).max(axis=1)
    for t, label in ((threshold, "calibrated"), (SSCD_THRESHOLD, "published 0.5")):
        rate = float((best >= t).mean())
        print(f"  at {t:.3f} ({label}): {100 * rate:.2f}% of generated images are "
              f"near-copies of a training image")
    print(f"  max similarity to any training image: {best.max():.4f}")
    print(f"  Somepalli et al. report 1.88% for SD v1.4 against LAION, at their threshold "
          f"on their corpus")

    # -- identity diversity ------------------------------------------------------------
    print("\n== identity diversity ==")
    face = cached_embedder(face_embedder(device=args.device), "data/splits/face_pool.npz")
    synth = Corpus(Record(path=p, label=0, source=Source.SYNTH_CLOSED) for p in pool)
    real = Corpus(Record(path=p, label=0, source=Source.REAL) for p in train)
    result = identity_diversity(synth, face, threshold=0.60, reference=real)
    d = result.to_dict()
    print(f"  faces detected: {d['n_with_face']}/{d['n_images']}")
    print(f"  distinct identities: {d['n_identities']}  "
          f"(effective {d['effective_identities']:.1f}, largest {100*d['largest_identity_share']:.1f}%)")
    print(f"  identities per image: {d['n_identities'] / max(d['n_images'], 1):.3f}")
    if d.get("coverage_of_reference") is not None:
        print(f"  covers {100 * d['coverage_of_reference']:.1f}% of the "
              f"{d['n_reference_identities']} real training identities")
    print(f"\n  reference: the real training split holds 267 identities across 948 images "
          f"(0.282 per image)")


if __name__ == "__main__":
    main()
