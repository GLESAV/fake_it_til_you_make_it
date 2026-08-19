#!/usr/bin/env python
"""Compare ACNE04 v1 labels against the independent ACNE04-v2 re-annotation.

ACNE04's severity grade is exactly the Hayashi banding of a lesion count, and the
release does not state which lesion types were counted. ACNE04-v2 (Gazeau, Nguyen et
al., MICCAI 2024, github.com/AIpourlapeau/acne04v2) is an independent expert
re-annotation of 1,204 of the same photographs under an explicitly different and more
inclusive protocol, with the original filenames preserved. Joining them measures how
much of the benchmark's label is a property of the image and how much is a property of
the annotation team.

    git clone https://github.com/AIpourlapeau/acne04v2
    python scripts/compare_acne04_versions.py --v2 acne04v2/Acne04-v2_annotations.json

See docs/05_acne04_audit.md §4 for the interpretation, including what part of the gap
is definitional and why that does not account for it.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

#: Hayashi et al., J Dermatol 35:255-260 (2008). Bands over INFLAMMATORY eruptions
#: (papules + pustules) on half the face; the paper reports that global severity did
#: not correlate with comedone counts, which is why comedones are excluded.
HAYASHI_UPPER = (5, 20, 50)


def hayashi(counts: np.ndarray) -> np.ndarray:
    out = np.full(len(counts), 3)
    for grade, upper in reversed(list(enumerate(HAYASHI_UPPER))):
        out[counts <= upper] = grade
    return out


def quadratic_weighted_kappa(a: np.ndarray, b: np.ndarray, k: int = 4) -> float:
    observed = np.zeros((k, k))
    for x, y in zip(a, b):
        observed[x, y] += 1
    weights = np.array([[(i - j) ** 2 / (k - 1) ** 2 for j in range(k)] for i in range(k)])
    expected = np.outer(np.bincount(a, minlength=k), np.bincount(b, minlength=k)) / len(a)
    expected *= observed.sum() / expected.sum()
    return 1 - (weights * observed).sum() / (weights * expected).sum()


def quantile_match(source: np.ndarray, target_grades: np.ndarray) -> np.ndarray:
    """Re-band `source` counts so the grade histogram equals `target_grades`'.

    This grants the second annotation an arbitrary monotone recalibration, which removes
    every effect of "they counted more lesion types". Residual disagreement after this
    cannot be definitional.
    """
    out = np.empty(len(source), int)
    order = np.argsort(source, kind="stable")
    pos = 0
    for grade in range(4):
        n = int((target_grades == grade).sum())
        out[order[pos : pos + n]] = grade
        pos += n
    return out


def read_v1(base: Path) -> dict[str, tuple[int, int]]:
    out: dict[str, tuple[int, int]] = {}
    for path in sorted(base.glob("NNEW_*.txt")):
        for line in path.read_text().splitlines():
            parts = line.split()
            if len(parts) >= 3:
                out[parts[0]] = (int(parts[1]), int(parts[2]))
    return out


def read_v2(path: Path) -> dict[str, int]:
    data = json.loads(path.read_text())
    names = {im["id"]: im["file_name"] for im in data["images"]}
    counts = Counter(names[a["image_id"]] for a in data["annotations"])
    return {name: counts.get(name, 0) for name in names.values()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/acne04")
    ap.add_argument("--v2", required=True, help="path to Acne04-v2_annotations.json")
    args = ap.parse_args()

    base = next((p.parent for p in Path(args.root).rglob("NNEW_trainval_0.txt")), None)
    if base is None:
        raise SystemExit(f"no ACNE04 label files under {args.root}")
    v1, v2 = read_v1(base), read_v2(Path(args.v2))

    shared = sorted(set(v1) & set(v2))
    print(f"v1 images {len(v1)}   v2 images {len(v2)}   shared {len(shared)}")
    print(f"v2 filenames absent from v1: {len(set(v2) - set(v1))}")

    c1 = np.array([v1[n][1] for n in shared])
    c2 = np.array([v2[n] for n in shared])
    g1 = np.array([v1[n][0] for n in shared])
    g2 = hayashi(c2)

    print(f"\nlesions per image  v1: mean {c1.mean():.1f} median {np.median(c1):.0f} max {c1.max()}")
    print(f"                   v2: mean {c2.mean():.1f} median {np.median(c2):.0f} max {c2.max()}")
    print(f"ratio of means {c2.mean() / c1.mean():.2f}x   v2 > v1 on {100 * (c2 > c1).mean():.1f}% of images")
    print(f"Pearson r {np.corrcoef(c1, c2)[0, 1]:.3f}   "
          f"Spearman rho {np.corrcoef(c1.argsort().argsort(), c2.argsort().argsort())[0, 1]:.3f}")

    ratio = c2 / np.maximum(c1, 1)
    print(f"\nper-image ratio: median {np.median(ratio):.2f}  "
          f"IQR {np.percentile(ratio, 25):.2f}-{np.percentile(ratio, 75):.2f}  "
          f"p10-p90 {np.percentile(ratio, 10):.2f}-{np.percentile(ratio, 90):.2f}")
    print(f"images where v2 counted FEWER than v1: {int((c2 < c1).sum())} "
          f"({100 * (c2 < c1).mean():.1f}%)")
    edges = np.quantile(c1, np.linspace(0, 1, 6))
    print("ratio by v1 count band (a constant definitional factor would be flat):")
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (c1 >= lo) & (c1 <= hi)
        if m.sum() < 10:
            continue
        print(f"  v1 {int(lo):>3}-{int(hi):<3} (n={int(m.sum()):>4}): median ratio {np.median(ratio[m]):.2f}")

    g2q = quantile_match(c2, g1)
    print(f"\ngrade agreement, v2 under ACNE04's own Hayashi banding: "
          f"{100 * (g1 == g2).mean():.1f}%   QWK {quadratic_weighted_kappa(g1, g2):.3f}")
    print(f"grade agreement, v2 after optimal monotone rescaling:    "
          f"{100 * (g1 == g2q).mean():.1f}%   QWK {quadratic_weighted_kappa(g1, g2q):.3f}")
    print(f"  off by one grade: {100 * (np.abs(g1 - g2q) == 1).mean():.1f}%   "
          f"by two or more: {100 * (np.abs(g1 - g2q) >= 2).mean():.1f}%")
    print(f"\nshare graded severe or worse: v1 {100 * (g1 >= 2).mean():.1f}%   "
          f"v2 {100 * (g2 >= 2).mean():.1f}%")


if __name__ == "__main__":
    main()
