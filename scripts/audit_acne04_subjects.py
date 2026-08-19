#!/usr/bin/env python
"""Face-identity audit of ACNE04: how many people, and how much subject leakage.

Deduplication asks "is this the same picture?". On a face dataset the question that
governs train/test leakage is "is this the same person?". This script answers the
second one and measures the consequence for ACNE04's own published folds and for any
split directory you point it at.

    python scripts/audit_acne04_subjects.py --root data/acne04 \
        --splits data/splits --splits data/splits_subject

Needs `insightface` and `onnxruntime`. Embeddings are cached, so the second run is fast.
See docs/05_acne04_audit.md section 5.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from fitymi.controls.embedders import cached_embedder, face_embedder
from fitymi.data.acne04 import load_acne04
from fitymi.data.dedup import UnionFind

#: Reported alongside every leakage figure so the false-positive budget is visible.
THRESHOLDS = (0.85, 0.80, 0.75, 0.70, 0.65, 0.60, 0.50)


def read_fold(base: Path, name: str) -> set[str]:
    return {
        line.split()[0]
        for line in (base / name).read_text().splitlines()
        if len(line.split()) >= 2
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/acne04")
    ap.add_argument("--cache", default="data/splits/face_embeddings.npz")
    ap.add_argument("--splits", action="append", default=[],
                    help="split directory to check; repeatable")
    args = ap.parse_args()

    records = list(load_acne04(args.root))
    paths = [r.path for r in records]
    names = [Path(p).name for p in paths]
    where = {p: i for i, p in enumerate(paths)}

    emb = cached_embedder(face_embedder(), args.cache)(paths)
    emb = np.asarray(emb, dtype=np.float32)
    detected = np.abs(emb).sum(axis=1) > 0
    print(f"faces detected in {int(detected.sum())}/{len(paths)} images "
          f"({100 * detected.mean():.1f}%) -- images are padded before detection, "
          f"without which the rate is about 9%")

    keep = np.where(detected)[0]
    cos = emb[keep] @ emb[keep].T
    upper = np.triu_indices(len(keep), k=1)
    chance = {t: float((cos[upper] >= t).mean()) for t in THRESHOLDS}

    print("\n== identity structure ==")
    print(f"{'cosine':>7} {'subjects':>9} {'in multi':>9} {'largest':>8}")
    for t in THRESHOLDS:
        uf = UnionFind(len(keep))
        ii, jj = np.where(np.triu(cos >= t, k=1))
        for a, b in zip(ii.tolist(), jj.tolist()):
            uf.union(a, b)
        sizes = Counter(uf.find(i) for i in range(len(keep)))
        multi = sum(v for v in sizes.values() if v > 1)
        print(f"{t:>7.2f} {len(sizes):>9} {multi:>9} {max(sizes.values()):>8}")

    base = next((p.parent for p in Path(args.root).rglob("NNEW_trainval_0.txt")), None)
    if base is not None:
        print("\n== leakage in ACNE04's own published folds ==")
        print("direct links only, no transitive closure; 'chance' is the share of ALL "
              "image pairs above threshold")
        print(f"{'cosine':>7} {'chance':>8} " + " ".join(f"fold{k}" for k in range(5)) + f" {'mean':>7}")
        for t in THRESHOLDS:
            rates = []
            for k in range(5):
                train = read_fold(base, f"NNEW_trainval_{k}.txt")
                test = read_fold(base, f"NNEW_test_{k}.txt")
                ti = [where[p] for p, n in zip(paths, names) if n in train and detected[where[p]]]
                ei = [where[p] for p, n in zip(paths, names) if n in test and detected[where[p]]]
                keep_pos = {int(k_): i for i, k_ in enumerate(keep)}
                rows = [keep_pos[i] for i in ei]
                cols = [keep_pos[i] for i in ti]
                rates.append(float((cos[np.ix_(rows, cols)].max(1) >= t).mean()))
            print(f"{t:>7.2f} {100 * chance[t]:>7.3f}% " +
                  " ".join(f"{100 * r:5.1f}" for r in rates) +
                  f" {100 * np.mean(rates):>6.1f}%")

    capture_condition_control(paths, emb, detected)

    for split_dir in args.splits:
        d = Path(split_dir)
        if not (d / "train.jsonl").exists():
            print(f"\n{split_dir}: no train.jsonl, skipping")
            continue
        def load(name: str) -> list[int]:
            out = []
            for line in (d / f"{name}.jsonl").read_text().splitlines():
                if line.strip():
                    p = json.loads(line)["path"]
                    if p in where and detected[where[p]]:
                        out.append(where[p])
            return out
        keep_pos = {int(k_): i for i, k_ in enumerate(keep)}
        rows = [keep_pos[i] for i in load("test")]
        cols = [keep_pos[i] for i in load("train")]
        best = cos[np.ix_(rows, cols)].max(1)
        print(f"\n== {split_dir}: test n={len(rows)} ==")
        for t in THRESHOLDS:
            print(f"  cosine >= {t:.2f}: {100 * (best >= t).mean():5.1f}% of test images "
                  f"share a face with training")


def capture_condition_control(paths, emb, detected, thresholds=THRESHOLDS) -> None:
    """Separate identity from shared capture conditions.

    The obvious objection to an embedding-based leakage claim is that the model may be
    scoring the photo shoot rather than the person. Restricting to the dominant
    resolution group -- one camera, one setup -- decides it: if capture conditions drove
    the similarity, the distribution inside that group would be shifted high and
    unimodal rather than keeping a thin identity tail.
    """
    from collections import Counter

    from PIL import Image

    keep = np.where(detected)[0]
    sizes = []
    for i in keep:
        with Image.open(paths[i]) as im:
            sizes.append(f"{im.size[0]}x{im.size[1]}")
    sizes = np.array(sizes)
    dominant, n = Counter(sizes).most_common(1)[0]
    sel = np.where(sizes == dominant)[0]

    emb_keep = emb[keep]
    cos = emb_keep[sel] @ emb_keep[sel].T
    upper = np.triu_indices(len(sel), k=1)
    d = cos[upper]

    print(f"\n== capture-condition control: {dominant}, {n} images, one camera ==")
    print("  median cosine {:.3f}   p90 {:.3f}   p99 {:.3f}".format(
        float(np.median(d)), float(np.percentile(d, 90)), float(np.percentile(d, 99))))
    for t in (0.85, 0.60):
        print(f"  share of pairs >= {t:.2f}: {100 * float((d >= t).mean()):.3f}%")
    print("  A high, unimodal distribution here would mean the links are capture "
          "conditions.\n  A low one with a thin tail means they are identity.")


if __name__ == "__main__":
    main()
