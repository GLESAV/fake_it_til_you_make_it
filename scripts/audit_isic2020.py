#!/usr/bin/env python
"""ISIC 2020 leakage, at both grouping levels, from published metadata.

The point of this one is the comparison between the two levels. ISIC 2020 carries both
`lesion_id` and `patient_id`, and it looks clean at one and catastrophic at the other:
1.01 images per lesion and 2.0% leakage, against 16.11 images per patient and 99.9%. An
audit that reached for `lesion_id` -- the column HAM10000 taught the field to use -- would
clear this dataset.

It also falsifies a prediction. Where duplication reflects clinical judgement we expected
the leak to concentrate on the worrying class, as it does in HAM10000. Here it reverses:
benign leaks at 99.8% and malignant at 39.3%, because screening photographs many nevi per
patient while a malignant finding is usually one lesion.

    python scripts/audit_isic2020.py
"""

from __future__ import annotations

import argparse
import csv
import urllib.request
from pathlib import Path

import numpy as np

URL = ("https://isic-challenge-data.s3.amazonaws.com/2020/"
       "ISIC_2020_Training_GroundTruth_v2.csv")


def leakage(groups: list[str], mask: list[bool] | None = None,
            trials: int = 100, test_fraction: float = 0.2) -> float | None:
    idx = [i for i, g in enumerate(groups) if g and (mask is None or mask[i])]
    if len(idx) < 100:
        return None
    g = [groups[i] for i in idx]
    rng = np.random.default_rng(0)
    out = []
    for _ in range(trials):
        order = rng.permutation(len(g))
        cut = int((1 - test_fraction) * len(g))
        train = {g[i] for i in order[:cut]}
        out.append(float(np.mean([g[i] in train for i in order[cut:]])))
    return float(np.mean(out))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--metadata", default="data/isic2020_metadata.csv")
    args = ap.parse_args()

    path = Path(args.metadata)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        print(f"downloading {URL}")
        urllib.request.urlretrieve(URL, path)
    with open(path) as fh:
        rows = list(csv.DictReader(fh))

    lesions = [r["lesion_id"] for r in rows]
    patients = [r["patient_id"] for r in rows]
    malignancy = [r["benign_malignant"] for r in rows]
    print(f"{len(rows)} images")

    print("\n== the same dataset at two grouping levels ==")
    for name, groups in (("lesion", lesions), ("patient", patients)):
        unique = len(set(g for g in groups if g))
        rate = leakage(groups)
        print(f"  by {name:>7}: {unique:>6} groups, {len(rows) / unique:>6.2f} images each, "
              f"leakage {100 * rate:>5.1f}%")
    print("  An audit that grouped by lesion_id here would clear this dataset.")

    print("\n== by malignancy, at patient level ==")
    for label in ("malignant", "benign"):
        mask = [m == label for m in malignancy]
        n_images = sum(mask)
        n_patients = len({patients[i] for i in range(len(rows)) if mask[i] and patients[i]})
        rate = leakage(patients, mask)
        print(f"  {label:>9}: {n_images:>6} images, {n_patients:>5} patients "
              f"({n_images / max(n_patients, 1):>5.2f} each), leakage {100 * rate:>5.1f}%")
    print("  The direction is the opposite of HAM10000's: screening photographs many nevi")
    print("  per patient, while a malignant finding is usually a single lesion.")


if __name__ == "__main__":
    main()
