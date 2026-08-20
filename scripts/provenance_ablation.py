#!/usr/bin/env python
"""How much of ACNE04 accuracy survives normalising away capture provenance?

The audit established that severity correlates with which device took the picture -- 87.1%
of mild images and 16.3% of very-severe ones share one resolution -- and that a file-header
rule reaches sensitivity 0.57 at specificity 0.85. What it did not establish is whether a
classifier actually uses that. This measures it.

Every image is re-encoded to a common square resolution and a common JPEG quality, which
removes resolution, aspect ratio and quantisation as signals while leaving the skin intact.
Training then repeats with identical splits, hyperparameters and seed. The drop is an
estimate of what provenance was contributing.

**It is an upper bound, not a point estimate.** Re-encoding also destroys real information:
a 690x920 web image genuinely carries less detail than a 3112x3456 clinical photograph, and
squashing both to 512 square costs the second one something. So a drop confounds "lost the
shortcut" with "lost real resolution". A drop of roughly zero would be the clean result,
because it would rule the shortcut out; a large drop bounds it from above.

Watermarks and eye-pixelation survive re-encoding, so any remaining shortcut is theirs.

    python scripts/provenance_ablation.py --prepare      # re-encode + rewrite splits
    python scripts/provenance_ablation.py --train        # train on the normalised copy
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

NORMALISED = Path("data/acne04_normalised")
SPLITS_OUT = Path("data/splits_normalised")


def prepare(splits_in: Path, size: int, quality: int) -> None:
    from PIL import Image

    NORMALISED.mkdir(parents=True, exist_ok=True)
    SPLITS_OUT.mkdir(parents=True, exist_ok=True)
    seen = 0
    for split in ("train", "val", "test"):
        source = splits_in / f"{split}.jsonl"
        if not source.exists():
            continue
        rows = [json.loads(l) for l in source.read_text().splitlines() if l.strip()]
        out_rows = []
        for row in rows:
            src = Path(row["path"])
            dst = NORMALISED / (src.stem + ".jpg")
            if not dst.exists():
                with Image.open(src) as im:
                    im = im.convert("RGB")
                    w, h = im.size
                    side = min(w, h)
                    im = im.crop(((w - side) // 2, (h - side) // 2,
                                  (w - side) // 2 + side, (h - side) // 2 + side))
                    im = im.resize((size, size), Image.LANCZOS)
                    im.save(dst, "JPEG", quality=quality, subsampling=1)
                seen += 1
            out_rows.append({**row, "path": str(dst)})
        (SPLITS_OUT / f"{split}.jsonl").write_text(
            "\n".join(json.dumps(r) for r in out_rows) + "\n")
    print(f"re-encoded {seen} images to {size}x{size} JPEG q{quality} in {NORMALISED}")
    print(f"rewrote splits into {SPLITS_OUT}")
    print("resolution, aspect ratio and quantisation are now identical across every image;")
    print("watermarks and eye-pixelation are not, so any shortcut left is theirs")


def train(seed: int) -> None:
    from fitymi.config import ExperimentConfig
    from fitymi.data.records import Corpus, Record
    from fitymi.train.loop import TrainConfig, evaluate_corpus, train_model

    cfg = ExperimentConfig.load("configs/acne04_closed.yaml")

    def load(split: str) -> Corpus:
        return Corpus(
            Record(**{k: v for k, v in json.loads(l).items() if k != "meta"})
            for l in (SPLITS_OUT / f"{split}.jsonl").read_text().splitlines() if l.strip()
        )

    tc = TrainConfig(**{**asdict(cfg.train), "seed": seed, "num_workers": 4})
    model, _ = train_model(load("train"), load("val"), tc)
    result = evaluate_corpus(model, load("val"), tc)
    print("\nNORMALISED", json.dumps({k: round(v, 4) for k, v in result.to_dict().items()
                                      if isinstance(v, float)}))
    print("compare against the same config on original images: balanced accuracy 0.7296")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", default="data/splits_subject")
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--quality", type=int, default=90)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--prepare", action="store_true")
    ap.add_argument("--train", action="store_true")
    args = ap.parse_args()
    if args.prepare:
        prepare(Path(args.splits), args.size, args.quality)
    if args.train:
        train(args.seed)


if __name__ == "__main__":
    main()
