"""Stage 3: robustness checks for the SCIN replication.

1. Degenerate-ITA pathology: the repo estimator uses atan2(L-50, max(b,1e-6)), so
   images whose median b* is ~0 pin at +/-90 deg and can swing 180 deg. Quantify
   and re-run (A)/(B) with them dropped, and with high-coverage CLOSE_UP images
   only (the closest analogue to ACNE04's framed face photos).
2. Verify the linear-light vs naive-sRGB-scaling protocol point.
"""
from __future__ import annotations

import glob
import os
import sys

import numpy as np
import pandas as pd
from PIL import Image
from scipy import stats

sys.path.insert(0, "/Users/gs/fake_it_til_you_make_it/src")
sys.path.insert(0, "/Users/gs/fake_it_til_you_make_it/scripts_replication")
from analyse import cv_accuracy, majority_floor, separation, tone_bin  # noqa: E402
from compute_ita import expose, ita_from_array  # noqa: E402

ROOT = "/Users/gs/fake_it_til_you_make_it"
TAGS = {-0.5: "m05", -0.3: "m03", 0.3: "p03", 0.5: "p05"}
ORDER = ["FST1", "FST2", "FST3", "FST4", "FST5", "FST6"]


def report_A(d, name):
    base = d["ita_raw_p00"].values
    bb = tone_bin(base)
    print(f"\n-- (A) on {name}: n={len(d)} --")
    flips = []
    for ev, t in TAGS.items():
        p = d[f"ita_raw_{t}"].values
        dd = np.abs(p - base)
        f = tone_bin(p) != bb
        flips.append(f)
        print(f"  {ev:+.1f} EV: mean|dITA|={dd.mean():6.2f}  median={np.median(dd):6.2f} "
              f" p95={np.percentile(dd,95):6.2f}  max={dd.max():7.2f}  bin flip={100*f.mean():5.1f}%")
    print(f"  flips under at least one EV: {100*np.any(np.stack(flips),0).mean():.1f}%")


def report_B(d, name):
    d = d[d.fst_self.isin(ORDER)]
    y, ita = d.fst_self.values, d.ita_raw_p00.values
    acc, fl = cv_accuracy(ita, ita, y), majority_floor(y)
    med, sd, gaps, psd = separation(ita, y, ORDER)
    rho, p = stats.spearmanr(ita, d.fst_self.map({k: i for i, k in enumerate(ORDER)}).values)
    print(f"\n-- (B) self-FST on {name}: n={len(y)} --")
    print(f"  CV acc {acc:.3f} vs floor {fl:.3f} (lift {acc-fl:+.3f})  rho={rho:+.3f} (p={p:.1e})")
    print(f"  gap={np.mean(gaps):.2f}  within-label SD={psd:.2f}  RATIO={np.mean(gaps)/psd:.3f}")


def main():
    d = pd.read_csv(f"{ROOT}/data/scin_ita.csv").drop_duplicates(subset="file")
    print("=" * 72)
    print("ROBUSTNESS 1: degenerate-ITA pathology")
    print("=" * 72)
    deg = d.ita_raw_p00.abs() > 89
    print(f"images pinned at |ITA|~90 (median b* ~ 0): {deg.sum()} / {len(d)} "
          f"= {100*deg.mean():.1f}%")
    report_A(d, "ALL images")
    report_A(d[~deg], "non-degenerate (|ITA|<=89)")
    cu = (~deg) & (d.shot_type == "CLOSE_UP") & (d.cov_raw_p00 > 0.8)
    report_A(d[cu], "CLOSE_UP + coverage>0.8, non-degenerate")
    report_B(d, "ALL images")
    report_B(d[~deg], "non-degenerate")
    report_B(d[cu], "CLOSE_UP + coverage>0.8, non-degenerate")

    print("\n" + "=" * 72)
    print("ROBUSTNESS 2: linear-light exposure vs naive sRGB scaling")
    print("=" * 72)
    files = sorted(glob.glob(f"{ROOT}/data/scin_images/*.png"))[:200]
    lin_d, naive_d = {ev: [] for ev in TAGS}, {ev: [] for ev in TAGS}
    for f in files:
        with Image.open(f) as im:
            im = im.convert("RGB")
            im.thumbnail((256, 256))
            a = np.asarray(im, dtype=np.float64) / 255.0
        b = ita_from_array(a)[0]
        for ev in TAGS:
            lin_d[ev].append(abs(ita_from_array(expose(a, ev))[0] - b))
            naive_d[ev].append(abs(ita_from_array(np.clip(a * 2.0**ev, 0, 1))[0] - b))
    print(f"n={len(files)} images")
    for ev in TAGS:
        L, N = np.median(lin_d[ev]), np.median(naive_d[ev])
        print(f"  {ev:+.1f} EV: median|dITA| linear={L:6.2f}  naive-sRGB={N:6.2f}  "
              f"overstatement x{N/max(L,1e-9):.2f}")


if __name__ == "__main__":
    main()
