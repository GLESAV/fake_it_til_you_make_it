#!/usr/bin/env python
"""HAM10000 split leakage from published metadata, broken down by diagnosis.

`lesion_id` is ground truth for "same lesion", so this needs no model and no images.

HAM10000's duplication is already documented -- Abhishek et al. (Nature Scientific Data
2025) found near-duplicate lesions spanning the train/validation boundary and released
corrected partitions for the DermaMNIST derivative. This script does not rediscover that.
It measures the size and the shape of the consequence, and the shape is the point:
duplication is not independent of diagnosis, so a single leakage rate understates the
effect on the class the benchmark is reported for.

    python scripts/audit_ham10000.py
"""

from __future__ import annotations

import argparse
import csv
import urllib.request
from collections import Counter
from pathlib import Path

import numpy as np

METADATA_URL = "https://dataverse.harvard.edu/api/access/datafile/3172582"
NAMES = {
    "mel": "melanoma", "nv": "nevus", "bcc": "basal cell carcinoma",
    "akiec": "actinic keratosis", "bkl": "benign keratosis",
    "df": "dermatofibroma", "vasc": "vascular",
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--metadata", default="data/ham10000_metadata.tsv")
    ap.add_argument("--trials", type=int, default=200)
    ap.add_argument("--test-fraction", type=float, default=0.2)
    args = ap.parse_args()

    path = Path(args.metadata)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        print(f"downloading {METADATA_URL}")
        urllib.request.urlretrieve(METADATA_URL, path)

    with open(path) as fh:
        rows = [{k: v.strip('"') for k, v in r.items()}
                for r in csv.DictReader(fh, delimiter="\t")]
    lesions = [r["lesion_id"] for r in rows]
    diagnoses = [r["dx"] for r in rows]
    per_lesion = Counter(lesions)

    print(f"{len(rows)} images across {len(per_lesion)} lesions "
          f"= {len(rows) / len(per_lesion):.2f} images per lesion")
    multi = sum(v for v in per_lesion.values() if v > 1)
    print(f"images belonging to a multi-image lesion: {multi}/{len(rows)} "
          f"({100 * multi / len(rows):.1f}%)")

    rng = np.random.default_rng(0)
    overall, by_dx = [], {d: [] for d in set(diagnoses)}
    for _ in range(args.trials):
        order = rng.permutation(len(lesions))
        cut = int((1 - args.test_fraction) * len(lesions))
        train = {lesions[i] for i in order[:cut]}
        test = order[cut:]
        leaked = np.array([lesions[i] in train for i in test])
        overall.append(leaked.mean())
        test_dx = np.array([diagnoses[i] for i in test])
        for d in by_dx:
            mask = test_dx == d
            if mask.sum():
                by_dx[d].append(leaked[mask].mean())

    counts = Counter(diagnoses)
    print(f"\n{'diagnosis':>22} {'images':>8} {'img/lesion':>11} {'test leaked':>12}")
    for d in sorted(by_dx, key=lambda d: -float(np.mean(by_dx[d]))):
        n_lesions = len({lesions[i] for i in range(len(lesions)) if diagnoses[i] == d})
        print(f"{NAMES.get(d, d):>22} {counts[d]:>8} {counts[d] / n_lesions:>11.2f} "
              f"{100 * np.mean(by_dx[d]):>11.1f}%")
    print(f"{'OVERALL':>22} {len(rows):>8} {len(rows) / len(per_lesion):>11.2f} "
          f"{100 * np.mean(overall):>11.1f}%")
    print("\nMelanoma sensitivity is the headline metric on this benchmark, and melanoma")
    print("is the most duplicated class -- reasonable clinical practice, awkward for a")
    print("split. Duplication is almost never independent of class, so a single leakage")
    print("rate understates its effect on the class anyone reports.")


if __name__ == "__main__":
    main()
