#!/usr/bin/env python
"""Is ACNE04's severity label confounded with image provenance?

Looking at random images by grade, the severe ones carry watermarks, Chinese
acne-treatment branding and "Before 治疗前" banners with pixelated eyes, while the mild
ones look like ordinary clinic photographs. If the severe class was sourced largely from
marketing material and the mild class from a clinic camera, a classifier can predict
severity from capture artefacts without looking at skin.

Image resolution fingerprints the capture device, so the confound is measurable with no
modelling at all: no embeddings, no classifier, just PIL reading headers.

    python scripts/audit_acne04_provenance.py
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

NAMES = ("mild", "moderate", "severe", "very severe")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/acne04")
    args = ap.parse_args()

    base = next(p.parent for p in Path(args.root).rglob("NNEW_trainval_0.txt"))
    labels: dict[str, int] = {}
    for f in sorted(base.glob("NNEW_*.txt")):
        for line in f.read_text().splitlines():
            parts = line.split()
            if len(parts) >= 3:
                labels[parts[0]] = int(parts[1])

    sizes: dict[str, tuple[int, int]] = {}
    for name in labels:
        with Image.open(base / "JPEGImages" / name) as im:
            sizes[name] = im.size
    dominant, n_dominant = Counter(sizes.values()).most_common(1)[0]

    print(f"{len(labels)} images. Dominant resolution {dominant[0]}x{dominant[1]} "
          f"covers {n_dominant} ({100 * n_dominant / len(labels):.1f}%) -- one capture device.")
    print(f"\n{'grade':>12} {'n':>5} {'dominant device':>18} {'elsewhere':>14}")
    for g in range(4):
        names = [n for n, v in labels.items() if v == g]
        on = sum(1 for n in names if sizes[n] == dominant)
        print(f"{NAMES[g]:>12} {len(names):>5} {100 * on / len(names):>16.1f}% "
              f"{100 * (len(names) - on) / len(names):>13.1f}%")

    # A rule that ignores the image entirely.
    severe = [n for n, v in labels.items() if v >= 2]
    mild = [n for n, v in labels.items() if v <= 1]
    tpr = sum(1 for n in severe if sizes[n] != dominant) / len(severe)
    fpr = sum(1 for n in mild if sizes[n] != dominant) / len(mild)
    print(f"\nA rule that reads only the file header -- 'not the dominant resolution "
          f"therefore severe' --\nfires on {100 * tpr:.1f}% of severe images and "
          f"{100 * fpr:.1f}% of mild ones.")
    print(f"That is a sensitivity of {tpr:.2f} at a specificity of {1 - fpr:.2f}, from "
          f"metadata alone,\nfor the distinction the benchmark exists to make.")

    print(f"\nGrade distribution within each source group:")
    for group, label in ((True, "dominant device"), (False, "everything else")):
        names = [n for n in labels if (sizes[n] == dominant) == group]
        hist = Counter(labels[n] for n in names)
        share = {NAMES[g]: f"{100 * hist.get(g, 0) / len(names):.1f}%" for g in range(4)}
        print(f"  {label:>16} (n={len(names):>4}): {share}")

    print("\nWhat this does NOT establish: that any published result exploited it. What it")
    print("does establish is that the shortcut is available, that it is strongest for the")
    print("two clinically important grades, and that nothing in the release warns of it.")


if __name__ == "__main__":
    main()
