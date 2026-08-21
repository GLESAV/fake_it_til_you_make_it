#!/usr/bin/env python
"""Does device stratification destroy real associations, or only confounded ones?

The ITA result rests on an association vanishing inside a capture-device stratum. The
serious objection is that stratification removes variance, and removing variance weakens
any association -- confounded or genuine. Without a negative control the result is
unreadable: "it disappeared" and "we broke the test" look identical.

So this runs the identical procedure on a feature that genuinely tracks acne severity for
reasons that have nothing to do with the camera. Inflammatory acne is erythematous, so
median a* -- the green-to-red axis of CIE Lab, over the same central skin region and with
the same erythema-decile handling as the ITA estimator -- should track severity through
biology rather than through provenance.

If a* survives the stratification that kills ITA, the stratification is doing its job. If
both die, the test is destroying variance and the ITA result means nothing.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score

from fitymi.controls.skintone import _srgb_to_lab, estimate_ita


def features(path: str) -> tuple[float, float]:
    """Return (ITA, median a*) from the same pixels, so only the channel differs."""
    with Image.open(path) as im:
        im = im.convert("RGB")
        im.thumbnail((256, 256))
        arr = np.asarray(im, dtype=np.float32) / 255.0
    h, w, _ = arr.shape
    centre = arr[h // 6: h - h // 6, w // 6: w - w // 6]
    lab = _srgb_to_lab(centre.reshape(-1, 3))
    L, a = lab[:, 0], lab[:, 1]
    lit = (L > 15) & (L < 95)
    if lit.sum() < 32:
        lit = np.ones_like(L, dtype=bool)
    keep = lit & (a <= np.percentile(a[lit], 90))
    if keep.sum() < 32:
        keep = lit
    return estimate_ita(path).ita, float(np.median(lab[keep][:, 1]))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", default="data/splits_subject")
    args = ap.parse_args()

    records = []
    for split in ("train", "val", "test"):
        f = Path(args.splits) / f"{split}.jsonl"
        if f.exists():
            records += [json.loads(line) for line in f.read_text().splitlines() if line.strip()]

    sizes, feats = {}, {}
    for r in records:
        with Image.open(r["path"]) as im:
            sizes[r["path"]] = im.size
        feats[r["path"]] = features(r["path"])
    dominant = Counter(sizes.values()).most_common(1)[0][0]

    def score(paths, col: int) -> float:
        x = np.array([feats[p][col] for p in paths]).reshape(-1, 1)
        y = np.array([r["label"] for r in records if r["path"] in set(paths)])
        if len(set(y.tolist())) < 2 or len(y) < 60:
            return float("nan")
        return float(cross_val_score(GradientBoostingClassifier(random_state=0), x, y,
                                     cv=5, scoring="balanced_accuracy").mean())

    strata = {
        "all images": [r["path"] for r in records],
        "within dominant device": [r["path"] for r in records if sizes[r["path"]] == dominant],
        "every other device": [r["path"] for r in records if sizes[r["path"]] != dominant],
    }
    print(f"{len(records)} images; severity floor 0.250. Dominant resolution "
          f"{dominant[0]}x{dominant[1]}.\n")
    print(f"{'stratum':>26} {'n':>6} {'ITA':>18} {'median a* (erythema)':>22}")
    for name, paths in strata.items():
        # Recover labels in the same order as paths.
        by_path = {r["path"]: r["label"] for r in records}
        y = np.array([by_path[p] for p in paths])
        def s(col):
            x = np.array([feats[p][col] for p in paths]).reshape(-1, 1)
            return float(cross_val_score(GradientBoostingClassifier(random_state=0), x, y,
                                         cv=5, scoring="balanced_accuracy").mean())
        ita_s, a_s = s(0), s(1)
        print(f"{name:>26} {len(paths):>6} {ita_s:>10.3f} ({ita_s-0.25:+.3f}) "
              f"{a_s:>14.3f} ({a_s-0.25:+.3f})")
    print("\nIf a* holds up where ITA collapses, the stratification is removing a confound.")
    print("If both collapse, it is removing variance and the ITA result is uninterpretable.")


if __name__ == "__main__":
    main()
