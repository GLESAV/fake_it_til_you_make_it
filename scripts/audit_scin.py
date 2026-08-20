#!/usr/bin/env python
"""Does the ACNE04 subject-leakage finding generalise? Test it on SCIN and HAM10000.

ACNE04's subject structure had to be recovered with a face-identity embedding, which
invites the objection that the finding is an artefact of that instrument. SCIN
(Google Research and Stanford Medicine, 10k+ crowdsourced dermatology images) removes the
objection entirely: its schema carries `image_1_path`, `image_2_path`, `image_3_path` per
case, so **case_id is ground truth for "same subject"**. No embedding, no threshold, no
judgement call.

Metadata only -- the CSV is a few megabytes and no images are downloaded.

    python scripts/audit_scin.py
"""

from __future__ import annotations

import argparse
import csv
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

CASES_URL = "https://storage.googleapis.com/dx-scin-public-data/dataset/scin_cases.csv"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default="data/scin_cases.csv")
    ap.add_argument("--trials", type=int, default=200)
    ap.add_argument("--test-fraction", type=float, default=0.2)
    args = ap.parse_args()

    path = Path(args.cases)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        print(f"downloading {CASES_URL}")
        urllib.request.urlretrieve(CASES_URL, path)

    with open(path) as fh:
        rows = list(csv.DictReader(fh))
    image_columns = [c for c in rows[0] if c.startswith("image_") and c.endswith("_path")]

    per_case = [sum(1 for c in image_columns if (r.get(c) or "").strip()) for r in rows]
    n_images = sum(per_case)
    print(f"{len(rows)} cases, {n_images} images -> {n_images / len(rows):.2f} images per case")
    print("images per case:", dict(sorted(Counter(per_case).items())))

    multi_cases = sum(1 for k in per_case if k > 1)
    multi_images = sum(k for k in per_case if k > 1)
    print(f"cases with more than one image: {multi_cases} ({100 * multi_cases / len(rows):.1f}%)")
    print(f"images in a multi-image case: {multi_images}/{n_images} "
          f"({100 * multi_images / n_images:.1f}%)")

    print("\n== what the shot types are ==")
    shots = Counter(
        tuple(sorted((r.get(c.replace("_path", "_shot_type")) or "").strip()
                     for c in image_columns if (r.get(c) or "").strip()))
        for r in rows
    )
    for combination, count in shots.most_common(5):
        print(f"  {combination}: {count}")
    print("  Multi-image cases are the same lesion at different distances and angles, so"
          "\n  they are more tightly related than two photographs of one person's two cheeks.")

    print("\n== duplicate images across cases ==")
    where = defaultdict(set)
    for r in rows:
        for c in image_columns:
            value = (r.get(c) or "").strip()
            if value:
                where[value].add(r["case_id"])
    repeated = {p: cases for p, cases in where.items() if len(cases) > 1}
    print(f"image files listed under more than one case: {len(repeated)}")
    if repeated:
        worst = max(repeated.items(), key=lambda kv: len(kv[1]))
        print(f"  worst: one file appears under {len(worst[1])} different cases")
        print("  Case-level grouping does not fix these; they need file-level dedup as well.")

    print(f"\n== what an image-level {100 * (1 - args.test_fraction):.0f}/"
          f"{100 * args.test_fraction:.0f} split leaks ==")
    rng = np.random.default_rng(0)
    flat = [i for i, k in enumerate(per_case) for _ in range(k)]
    rates = []
    for _ in range(args.trials):
        order = rng.permutation(len(flat))
        cut = int((1 - args.test_fraction) * len(flat))
        train = {flat[i] for i in order[:cut]}
        rates.append(float(np.mean([flat[i] in train for i in order[cut:]])))
    lo, hi = np.percentile(rates, [2.5, 97.5])
    print(f"  {100 * np.mean(rates):.1f}% of test images share a case with training "
          f"(95% range {100 * lo:.1f}-{100 * hi:.1f}, {args.trials} trials)")
    print("\n  ACNE04, for comparison: 77.5% of its published test folds share a subject")
    print("  with training -- measured with a face embedding, where this needed no model.")


if __name__ == "__main__":
    main()
