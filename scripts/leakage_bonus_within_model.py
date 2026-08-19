#!/usr/bin/env python
"""What subject leakage is worth, measured inside a single model.

Comparing an image-level-split arm against a subject-disjoint arm is confounded: the two
arms have different test sets, so the difference mixes the leakage bonus with whatever
difficulty difference exists between those sets. This script removes that confound by
staying inside one model and one test set, partitioning the test images by whether the
person in them also appears in that model's training split.

    python scripts/leakage_bonus_within_model.py \
        --runs runs/acne04_baseline_preds --splits data/splits

Requires run records carrying `val_predictions` and the cached face embeddings.

The remaining caveat, stated rather than hidden: the two partitions are different
images, so an intrinsic difficulty difference between them is not excluded. Two things
bound it. The partition is by identity, which is unrelated to severity for the 73% of
subjects whose images all carry one grade; and the per-class breakdown below shows
whether the gap is uniform or concentrated where leakage should help most.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from fitymi.controls.embedders import cached_embedder, face_embedder
from fitymi.eval.metrics import balanced_accuracy
from fitymi.data.records import NUM_CLASSES, SEVERITY_NAMES

THRESHOLDS = (0.85, 0.75, 0.60)


def bootstrap_ci(values: np.ndarray, n: int = 20000, seed: int = 0) -> tuple[float, float]:
    if len(values) < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    draws = rng.choice(values, size=(n, len(values)), replace=True).mean(axis=1)
    return float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs/acne04_baseline_preds")
    ap.add_argument("--splits", default="data/splits")
    ap.add_argument("--cache", default="data/splits/face_embeddings.npz")
    args = ap.parse_args()

    records = []
    for path in sorted(Path(args.runs).rglob("run_*.json")):
        record = json.loads(path.read_text())
        if record.get("val_predictions"):
            records.append(record)
    if not records:
        raise SystemExit(
            f"no records with val_predictions under {args.runs}. Runs written before "
            "predictions were persisted cannot be analysed without retraining."
        )
    print(f"{len(records)} runs with per-image predictions")

    def split_paths(name: str) -> list[str]:
        return [json.loads(l)["path"]
                for l in (Path(args.splits) / f"{name}.jsonl").read_text().splitlines()
                if l.strip()]

    train_paths, val_paths = split_paths("train"), split_paths("val")
    all_paths = train_paths + val_paths
    emb = np.asarray(cached_embedder(face_embedder(), args.cache)(all_paths), np.float32)
    detected = np.abs(emb).sum(1) > 0
    n_train = len(train_paths)
    tr_emb = emb[:n_train][detected[:n_train]]
    val_names = [Path(p).name for p in val_paths]
    val_emb = emb[n_train:]
    val_det = detected[n_train:]

    best = np.full(len(val_paths), -1.0)
    best[val_det] = (val_emb[val_det] @ tr_emb.T).max(1)

    for threshold in THRESHOLDS:
        leaked_by_name = {n: bool(b >= threshold) for n, b in zip(val_names, best)}
        n_leaked = sum(leaked_by_name.values())
        print(f"\n===== identity threshold {threshold:.2f}: "
              f"{n_leaked}/{len(val_names)} validation images share a subject with training "
              f"({100 * n_leaked / len(val_names):.1f}%) =====")
        if n_leaked < 10 or len(val_names) - n_leaked < 10:
            print("  too few images on one side of the partition to report")
            continue

        rows = []
        for record in records:
            p = record["val_predictions"]
            names = p["images"]
            y_true = np.array(p["y_true"])
            y_pred = np.array(p["y_pred"])
            mask = np.array([leaked_by_name.get(n, False) for n in names])
            if mask.sum() == 0 or (~mask).sum() == 0:
                continue
            rows.append({
                "seed": record["config"]["seed"],
                "acc_leaked": float((y_true[mask] == y_pred[mask]).mean()),
                "acc_clean": float((y_true[~mask] == y_pred[~mask]).mean()),
                "bacc_leaked": balanced_accuracy(y_true[mask], y_pred[mask], NUM_CLASSES),
                "bacc_clean": balanced_accuracy(y_true[~mask], y_pred[~mask], NUM_CLASSES),
            })

        print(f"{'seed':>6} {'acc leaked':>12} {'acc clean':>11} {'diff':>9}"
              f" {'bacc leaked':>13} {'bacc clean':>12} {'diff':>9}")
        for r in rows:
            print(f"{r['seed']:>6} {r['acc_leaked']:>12.4f} {r['acc_clean']:>11.4f} "
                  f"{r['acc_leaked'] - r['acc_clean']:>+9.4f} "
                  f"{r['bacc_leaked']:>13.4f} {r['bacc_clean']:>12.4f} "
                  f"{r['bacc_leaked'] - r['bacc_clean']:>+9.4f}")

        for key, label in (("acc", "plain accuracy"), ("bacc", "balanced accuracy")):
            d = np.array([r[f"{key}_leaked"] - r[f"{key}_clean"] for r in rows])
            lo, hi = bootstrap_ci(d)
            print(f"  {label:>18}: leaked minus clean = {100 * d.mean():+.2f} points "
                  f"(95% CI {100 * lo:+.2f} to {100 * hi:+.2f})")
            if not np.isnan(lo) and lo <= 0 <= hi:
                print(f"  {'':>18}  interval spans zero -- report the bound, not an effect")

        # Per-class, because a leakage bonus should concentrate where classes are small.
        print(f"  {'grade':>14} {'n leaked':>9} {'n clean':>8} {'recall leaked':>14} {'recall clean':>13}")
        for c in range(NUM_CLASSES):
            rl, rc, nl, nc = [], [], 0, 0
            for record in records:
                p = record["val_predictions"]
                names, y_true, y_pred = p["images"], np.array(p["y_true"]), np.array(p["y_pred"])
                mask = np.array([leaked_by_name.get(n, False) for n in names])
                a, b = (y_true == c) & mask, (y_true == c) & ~mask
                nl, nc = int(a.sum()), int(b.sum())
                if a.sum():
                    rl.append(float((y_pred[a] == c).mean()))
                if b.sum():
                    rc.append(float((y_pred[b] == c).mean()))
            name = SEVERITY_NAMES[c] if c < len(SEVERITY_NAMES) else str(c)
            print(f"  {name:>14} {nl:>9} {nc:>8} "
                  f"{np.mean(rl) if rl else float('nan'):>14.4f} "
                  f"{np.mean(rc) if rc else float('nan'):>13.4f}")


if __name__ == "__main__":
    main()
