#!/usr/bin/env python
"""Does a generated image carry the severity it was asked for?

The instrument is a classifier trained on the real subject-disjoint training split, and
the question is its agreement with the requested grade. That is protocol section 4.5's
grade-conditional fidelity measure, promoted to primary after the lesion counter failed
validation (Spearman 0.365 against ACNE04's own counts; see controls/lesion_count.py).

**The confound, and the control for it.** A classifier trained on ACNE04 -- half-face crops
at roughly 70 degrees against dark backgrounds -- scoring a studio-style portrait may fail
for domain reasons that have nothing to do with severity. So the same classifier is run on
three sets:

1. **real held-out images** -- the ceiling, and a check the scorer works at all;
2. **closed-set SD images**, fine-tuned on ACNE04 and therefore in its domain;
3. **the Gemini coverage pool**, out of domain by construction.

If the scorer reads severity on (1) and (2) but collapses on (3), that is domain shift and
the measurement says nothing about Gemini's severity control. If it reads severity on all
three, the comparison is meaningful. Reporting (3) alone would not distinguish these.

    python scripts/grade_fidelity.py --pool /tmp/coverage_probe --pattern 'g{grade}_*'
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import numpy as np


def rank(x: np.ndarray) -> np.ndarray:
    return np.argsort(np.argsort(x)).astype(float)


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 3 or np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(np.corrcoef(rank(a), rank(b))[0, 1])


def load_scorer(path: str):
    import torch

    from fitymi.data.records import NUM_CLASSES
    from fitymi.train.loop import TrainConfig
    from fitymi.train.models import build_model

    blob = torch.load(path, map_location="cpu", weights_only=False)
    config = TrainConfig(**blob["config"])
    model = build_model(config.arch, "scratch", NUM_CLASSES)
    model.load_state_dict(blob["state_dict"])
    print(f"scorer: {config.arch}/{config.init}, val balanced accuracy "
          f"{blob['best_val']:.4f} at epoch {blob['best_epoch']}")
    return model.eval(), config


def score(model, config, paths: list[str], labels: list[int]) -> dict:
    from fitymi.data.records import Corpus, Record, Source
    from fitymi.train.loop import evaluate_corpus

    corpus = Corpus(Record(path=p, label=int(l), source=Source.SYNTH_OPEN)
                    for p, l in zip(paths, labels))
    result, preds = evaluate_corpus(model, corpus, config, return_predictions=True)
    y_true = np.array(preds["y_true"]); y_pred = np.array(preds["y_pred"])
    return {
        "n": len(paths),
        "exact_agreement": float((y_true == y_pred).mean()),
        "within_one_grade": float((np.abs(y_true - y_pred) <= 1).mean()),
        "spearman_requested_vs_predicted": spearman(y_true.astype(float), y_pred.astype(float)),
        "mean_predicted_by_requested": {
            int(g): round(float(y_pred[y_true == g].mean()), 2)
            for g in sorted(set(y_true.tolist())) if (y_true == g).any()
        },
        "prediction_histogram": dict(Counter(y_pred.tolist())),
        "balanced_accuracy": result.balanced_accuracy,
    }


def from_pattern(directory: str, pattern: str) -> tuple[list[str], list[int]]:
    """Recover the requested grade from filenames like `g2_t4_r0.png`."""
    regex = re.compile(pattern.replace("{grade}", r"(?P<grade>\d)").replace("*", ".*"))
    paths, labels = [], []
    for p in sorted(Path(directory).glob("*.png")):
        m = regex.match(p.name)
        if m:
            paths.append(str(p))
            labels.append(int(m.group("grade")))
    return paths, labels


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scorer", default="models/grade_scorer.pt")
    ap.add_argument("--splits", default="data/splits_subject")
    ap.add_argument("--pool", action="append", default=[],
                    help="directory of generated images; repeatable")
    ap.add_argument("--pattern", default="g{grade}_*")
    args = ap.parse_args()

    model, config = load_scorer(args.scorer)

    print("\n== 1. real held-out images (the ceiling) ==")
    rows = [json.loads(l) for l in
            (Path(args.splits) / "val.jsonl").read_text().splitlines() if l.strip()]
    out = score(model, config, [r["path"] for r in rows], [r["label"] for r in rows])
    print(json.dumps(out, indent=2))

    for pool in args.pool:
        paths, labels = from_pattern(pool, args.pattern)
        if not paths:
            print(f"\n== {pool}: no filenames matched {args.pattern} ==")
            continue
        print(f"\n== {pool} ({len(paths)} images) ==")
        print(json.dumps(score(model, config, paths, labels), indent=2))


if __name__ == "__main__":
    main()
