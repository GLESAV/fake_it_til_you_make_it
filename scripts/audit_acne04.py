#!/usr/bin/env python
"""Data-quality audit of the ACNE04 archive. See docs/05_acne04_audit.md.

Requires nothing but the extracted archive: no model, no GPU, no network. Every table
in the audit document is produced here, so the claims are checkable by anyone with the
dataset.

    python scripts/audit_acne04.py [--root data/acne04] [--json out.json]
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from math import sqrt
from pathlib import Path

CLASSIFICATION = "Classification"
IMAGE_DIR = "JPEGImages"
#: Hayashi lesion-count bands, verified against every row of the label files.
HAYASHI = ((5, 0), (20, 1), (50, 2))


def hayashi(count: int) -> int:
    for upper, grade in HAYASHI:
        if count <= upper:
            return grade
    return 3


def wilson(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    p = successes / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return centre - half, centre + half


def find_root(root: Path) -> Path:
    for candidate in (root / CLASSIFICATION, root):
        if (candidate / IMAGE_DIR).is_dir():
            return candidate
    matches = [p.parent for p in root.rglob(IMAGE_DIR) if p.is_dir()]
    if not matches:
        raise SystemExit(f"no {IMAGE_DIR}/ directory under {root}")
    return matches[0]


def read_labels(base: Path) -> dict[str, tuple[int, int]]:
    """Every label row across all folds: filename -> (grade, lesion_count)."""
    out: dict[str, tuple[int, int]] = {}
    for path in sorted(base.glob("NNEW_*.txt")):
        for line in path.read_text().splitlines():
            parts = line.split()
            if len(parts) >= 3:
                out[parts[0]] = (int(parts[1]), int(parts[2]))
    return out


def read_fold(base: Path, name: str) -> dict[str, tuple[int, int]]:
    out: dict[str, tuple[int, int]] = {}
    for line in (base / name).read_text().splitlines():
        parts = line.split()
        if len(parts) >= 3:
            out[parts[0]] = (int(parts[1]), int(parts[2]))
    return out


def duplicate_groups(images: Path) -> list[list[str]]:
    by_hash: dict[str, list[str]] = defaultdict(list)
    for p in sorted(images.iterdir()):
        if p.suffix.lower() in {".jpg", ".jpeg", ".png"}:
            by_hash[hashlib.md5(p.read_bytes()).hexdigest()].append(p.name)
    return [v for v in by_hash.values() if len(v) > 1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/acne04")
    ap.add_argument("--json", default=None, help="write the findings as JSON")
    args = ap.parse_args()

    base = find_root(Path(args.root))
    images = base / IMAGE_DIR
    labels = read_labels(base)
    groups = duplicate_groups(images)
    n_files = sum(1 for p in images.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})

    print(f"archive: {base}")
    print(f"images on disk: {n_files}   labelled records: {len(labels)}")

    print("\n== 1. exact duplicates ==")
    dup_files = sum(len(g) for g in groups)
    print(f"byte-identical groups: {len(groups)}  files involved: {dup_files} "
          f"({100 * dup_files / n_files:.1f}%)")

    print("\n== 2. the grade is a deterministic function of the count ==")
    mismatched = [(n, v) for n, v in labels.items() if hayashi(v[1]) != v[0]]
    print(f"records where grade != hayashi(count): {len(mismatched)} of {len(labels)}")

    print("\n== 3. annotation self-consistency on duplicate pairs ==")
    pairs = [(labels[g[0]], labels[g[1]]) for g in groups
             if len(g) == 2 and g[0] in labels and g[1] in labels]
    grade_agree = sum(a[0] == b[0] for a, b in pairs)
    count_agree = sum(a[1] == b[1] for a, b in pairs)
    lo, hi = wilson(grade_agree, len(pairs))
    diffs = sorted(abs(a[1] - b[1]) for a, b in pairs)
    print(f"pairs: {len(pairs)}")
    print(f"grade agreement:  {grade_agree}/{len(pairs)} = {100 * grade_agree / len(pairs):.1f}% "
          f"(95% Wilson CI {100 * lo:.1f}-{100 * hi:.1f}%)")
    print(f"exact count agreement: {count_agree}/{len(pairs)} = "
          f"{100 * count_agree / len(pairs):.1f}%")
    print(f"|delta count|: median {diffs[len(diffs) // 2]}  "
          f"mean {sum(diffs) / len(diffs):.1f}  max {diffs[-1]}")
    for a, b in pairs:
        if a[0] != b[0]:
            print(f"    grade conflict: {a} vs {b}")

    print("\n== 4. leakage in the published 5-fold splits ==")
    fold_rows = []
    for k in range(5):
        tr = read_fold(base, f"NNEW_trainval_{k}.txt")
        te = read_fold(base, f"NNEW_test_{k}.txt")
        free = forced = 0
        for g in groups:
            in_tr = [n for n in g if n in tr]
            in_te = [n for n in g if n in te]
            if not in_tr or not in_te:
                continue
            for t in in_te:
                if any(tr[s][0] == te[t][0] for s in in_tr):
                    free += 1
                else:
                    forced += 1
        fold_rows.append({"fold": k, "n_test": len(te), "free": free, "forced": forced})
        print(f"fold {k}: test n={len(te)}  leaked {free + forced} "
              f"({100 * (free + forced) / len(te):.1f}%)  "
              f"free-correct {free} ({100 * free / len(te):.1f}%)  "
              f"forced-error {forced} ({100 * forced / len(te):.1f}%)")
    mean_leak = sum((r["free"] + r["forced"]) / r["n_test"] for r in fold_rows) / 5
    mean_free = sum(r["free"] / r["n_test"] for r in fold_rows) / 5
    mean_forced = sum(r["forced"] / r["n_test"] for r in fold_rows) / 5
    print(f"mean across folds: leaked {100 * mean_leak:.2f}%  "
          f"free-correct {100 * mean_free:.2f}%  forced-error {100 * mean_forced:.2f}%")

    print("\n== 5. filename prefix vs label ==")
    bad = [n for n, v in labels.items()
           if n.startswith("levle") and n[5:6].isdigit() and int(n[5]) != v[0]]
    print(f"files whose levleN prefix disagrees with the label: {len(bad)} of {len(labels)}")

    if args.json:
        Path(args.json).write_text(json.dumps({
            "n_files": n_files,
            "n_labelled": len(labels),
            "duplicate_groups": groups,
            "grade_agreement": {"agree": grade_agree, "n": len(pairs),
                                "ci_lo": lo, "ci_hi": hi},
            "count_agreement": {"agree": count_agree, "n": len(pairs)},
            "fold_leakage": fold_rows,
            "prefix_mismatches": bad,
        }, indent=2))
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
