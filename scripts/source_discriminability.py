#!/usr/bin/env python
"""Is capture source still visually detectable after normalisation?

The provenance ablation found that removing resolution, aspect ratio and quantisation does
not reduce accuracy -- so the classifier was not using those. But watermarks, branding and
eye-pixelation survive re-encoding, and severity correlates with source either way. So the
question becomes narrower: after normalisation, can a model still tell which source an
image came from?

If it can, the shortcut channel is open even though the ablation did not detect it being
used, and any future model on this dataset may find it. If it cannot, the confound is
closed and the audit's concern is discharged.

Trained and evaluated on the same subject-disjoint splits, so the answer is not itself
contaminated by subject leakage.

    python scripts/source_discriminability.py
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import asdict
from pathlib import Path


def main() -> None:
    import numpy as np
    from PIL import Image

    from fitymi.config import ExperimentConfig
    from fitymi.data.records import Corpus, Record, Source
    from fitymi.train.loop import TrainConfig, evaluate_corpus, train_model

    ap = argparse.ArgumentParser()
    ap.add_argument("--original", default="data/splits_subject")
    ap.add_argument("--normalised", default="data/splits_normalised")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    # Source label comes from the ORIGINAL file's resolution; the images fed to the model
    # are the NORMALISED ones, so resolution itself cannot be read off them.
    dominant = None
    sizes: dict[str, tuple[int, int]] = {}
    for split in ("train", "val"):
        for line in (Path(args.original) / f"{split}.jsonl").read_text().splitlines():
            if line.strip():
                p = Path(json.loads(line)["path"])
                with Image.open(p) as im:
                    sizes[p.stem] = im.size
    dominant = Counter(sizes.values()).most_common(1)[0][0]

    def load(split: str) -> Corpus:
        rows = [json.loads(l) for l in
                (Path(args.normalised) / f"{split}.jsonl").read_text().splitlines() if l.strip()]
        return Corpus(
            Record(path=r["path"], label=int(sizes[Path(r["path"]).stem] != dominant),
                   source=Source.REAL, group=r.get("group"))
            for r in rows if Path(r["path"]).stem in sizes
        )

    train_corpus, val_corpus = load("train"), load("val")
    print(f"predicting CAPTURE SOURCE from normalised images")
    print(f"  train {len(train_corpus)}  val {len(val_corpus)}")
    print(f"  class balance (0 = dominant device): {train_corpus.class_counts()}")

    cfg = ExperimentConfig.load("configs/acne04_closed.yaml")
    tc = TrainConfig(**{**asdict(cfg.train), "seed": args.seed, "num_workers": 4,
                        "epochs": 25, "patience": 8})
    model, _ = train_model(train_corpus, val_corpus, tc)
    result, preds = evaluate_corpus(model, val_corpus, tc, return_predictions=True)

    y_true = np.array(preds["y_true"]); y_pred = np.array(preds["y_pred"])
    majority = max(float((y_true == 0).mean()), float((y_true == 1).mean()))
    print(f"\n  accuracy {float((y_true == y_pred).mean()):.4f} against a majority-class "
          f"baseline of {majority:.4f}")
    print(f"  balanced accuracy {result.balanced_accuracy:.4f}")
    print("\n  Well above chance means source survives normalisation and the shortcut")
    print("  channel is open, whatever the ablation showed about it being used.")
    print("  Near chance means the confound is closed by re-encoding.")


if __name__ == "__main__":
    main()
