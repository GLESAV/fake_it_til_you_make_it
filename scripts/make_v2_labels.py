#!/usr/bin/env python
"""Emit ACNE04-format labels derived from the independent ACNE04-v2 annotations.

Protocol §8.9. The severity label in ACNE04 is exactly the Hayashi banding of a lesion
count, and the two expert teams' counts differ by a factor of 3.2 because they counted
different lesion types (docs/05_acne04_audit.md §4). Applying the Hayashi bands directly
to the v2 counts is clinically wrong -- those bands are defined over inflammatory
lesions and the v2 counts include comedones -- and would also change the class balance
beyond recognition, which would change the per-class training budget and confound the
comparison.

So we quantile-match instead: re-band the v2 counts so the v2 grade histogram equals the
v1 histogram on the shared images by construction. That grants the second team an
arbitrary monotone recalibration, holds class balance and budget fixed, and leaves only
the part of the disagreement that is about *which images are more severe than which*.
That residual is what the robustness control is testing against.

    python scripts/make_v2_labels.py --v2 acne04v2/Acne04-v2_annotations.json \
        --out data/acne04/Classification/NNEW_v2_all.txt

Then point a config at it:

    data:
      label_files: ["NNEW_v2_all.txt"]
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/acne04")
    ap.add_argument("--v2", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    base = next((p.parent for p in Path(args.root).rglob("NNEW_trainval_0.txt")), None)
    if base is None:
        raise SystemExit(f"no ACNE04 label files under {args.root}")

    v1: dict[str, tuple[int, int]] = {}
    for path in sorted(base.glob("NNEW_trainval_*.txt")) + sorted(base.glob("NNEW_test_*.txt")):
        for line in path.read_text().splitlines():
            parts = line.split()
            if len(parts) >= 3:
                v1[parts[0]] = (int(parts[1]), int(parts[2]))

    data = json.loads(Path(args.v2).read_text())
    names = {im["id"]: im["file_name"] for im in data["images"]}
    counts = Counter(names[a["image_id"]] for a in data["annotations"])
    v2 = {name: counts.get(name, 0) for name in names.values()}

    shared = sorted(set(v1) & set(v2))
    if not shared:
        raise SystemExit("no filenames in common between v1 and v2")
    c2 = np.array([v2[n] for n in shared])
    g1 = np.array([v1[n][0] for n in shared])

    # Quantile match: identical grade histogram, ordering taken from the v2 counts.
    order = np.argsort(c2, kind="stable")
    g2 = np.empty(len(shared), int)
    pos = 0
    for grade in range(4):
        n = int((g1 == grade).sum())
        g2[order[pos : pos + n]] = grade
        pos += n

    lines = [f"{name}\t{int(grade)}\t{int(v2[name])}" for name, grade in zip(shared, g2)]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n")

    agree = float((g1 == g2).mean())
    print(f"wrote {len(lines)} labels to {out}")
    print(f"grade histogram matched to v1 by construction: {dict(Counter(g2.tolist()))}")
    print(f"agreement with the v1 grade on these images: {100 * agree:.1f}%")
    print(f"images in v1 but not covered by v2 (excluded): {len(set(v1) - set(v2))}")


if __name__ == "__main__":
    main()
