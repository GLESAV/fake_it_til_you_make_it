#!/usr/bin/env python
"""Train on synthetic images only; validate on real. The question the project was for.

Three arms, all validated on the same real subject-disjoint validation split:

1. **synthetic-only** -- the coverage pool, labels as requested from the generator;
2. **real-only** -- the same classifier on the real training split, its natural prior;
3. **real-only, class-rebalanced** -- the real split resampled to the synthetic pool's
   balanced class distribution.

Three further arms ask the question a practitioner would actually ask. Nobody ships a
classifier trained on generated images alone; the realistic use is to *add* them to the
real data you already have. So:

4. **mixed** -- the full real training split plus the whole synthetic pool;
5. **mixed_tail** -- the full real split plus synthetic images for the scarce classes only,
   which is the targeted version of the premise: ACNE04 has 126 severe and 86 very-severe
   images, and the pitch for generated data is that it fills exactly that hole;
6. **pretrain** -- synthetic first, then fine-tune on real. The standard recipe, and the
   one that in the literature often works when synthetic-only does not, because the
   generated images only have to teach features rather than carry the decision boundary.

Arms 4-6 are compared against arm 2, not against each other: the question is whether
adding generated data to an existing real corpus buys anything at all.

The third arm exists because the synthetic pool differs from the real one in *two* ways: the
images are generated, and the label distribution is balanced rather than ACNE04's
35/43/12/9. That confound is deliberate -- rebalancing is part of the intervention, since
tail scarcity is the deficiency being fixed -- but a reader has to be able to see how much
of any difference is the rebalancing alone, and arm 3 shows it.

    python scripts/train_on_synthetic.py --pool data/synthetic/gemini_pool
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from dataclasses import asdict
from pathlib import Path


def main() -> None:
    import numpy as np

    from fitymi.config import ExperimentConfig
    from fitymi.data.records import NUM_CLASSES, SEVERITY_NAMES, Corpus, Record, Source
    from fitymi.train.loop import TrainConfig, evaluate_corpus, train_model
    from fitymi.utils.seeding import rng

    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default="data/synthetic/gemini_pool")
    ap.add_argument("--splits", default="data/splits_subject")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--arms", nargs="+",
                    default=["synthetic", "real", "real_balanced"],
                    help="synthetic real real_balanced mixed mixed_tail pretrain")
    ap.add_argument("--tail-classes", type=int, nargs="+", default=[2, 3],
                    help="which classes mixed_tail tops up; the scarce end by default")
    ap.add_argument("--balance", action="store_true",
                    help="cap every class at the smallest, so a partially generated pool "
                         "is still a balanced one and learning-curve points are comparable")
    ap.add_argument("--cap", type=int, default=0,
                    help="further cap images per class, for learning-curve points")
    args = ap.parse_args()

    cfg = ExperimentConfig.load("configs/acne04_closed.yaml")

    def real(split: str) -> Corpus:
        return Corpus(
            Record(**{k: v for k, v in json.loads(l).items() if k != "meta"})
            for l in (Path(args.splits) / f"{split}.jsonl").read_text().splitlines() if l.strip()
        )

    def synthetic(seed: int = 1234) -> Corpus:
        """`seed` selects *which* generated images the balanced subsample takes.

        It has to vary with the training seed, or the error bars answer the wrong
        question. Holding the subsample fixed measures only how much the result moves when
        the weights are initialised differently, and reports that as the uncertainty --
        while the thing a reader actually wants bounded is how much it moves when you get a
        different draw of generated images, which is the larger source and the one under
        the practitioner's control.
        """
        records = []
        for p in sorted(Path(args.pool).glob("*.png")):
            m = re.match(r"g(\d)_", p.name)
            if m:
                records.append(Record(path=str(p), label=int(m.group(1)),
                                      source=Source.SYNTH_OPEN))
        corpus = Corpus(records)
        if not (args.balance or args.cap):
            return corpus
        by_class = corpus.by_class()
        per = min(len(v) for v in by_class.values()) if by_class else 0
        if args.cap:
            per = min(per, args.cap)
        gen = rng(seed)
        out = []
        for c in sorted(by_class):
            items = list(by_class[c])
            idx = gen.permutation(len(items))[:per]
            out.extend(items[i] for i in idx)
        return Corpus(out)

    train_real, val_real = real("train"), real("val")
    pool = synthetic(args.seeds[0])
    print(f"real train {len(train_real)} {train_real.class_counts()}")
    print(f"synthetic   {len(pool)} {pool.class_counts()}")
    print(f"real val    {len(val_real)} {val_real.class_counts()}  <- every arm scored here")

    def rebalanced(corpus: Corpus, target: dict[int, int], seed: int) -> Corpus:
        """Resample `corpus` to `target`'s class shape, with replacement where short."""
        gen = rng(seed)
        by_class = corpus.by_class()
        out = []
        for c in range(NUM_CLASSES):
            want = target.get(c, 0)
            have = list(by_class.get(c, []))
            if not have or want == 0:
                continue
            idx = gen.choice(len(have), size=want, replace=want > len(have))
            out.extend(have[i] for i in idx)
        return Corpus(out)

    results: dict[str, list[dict]] = {}
    for seed in args.seeds:
        pool = synthetic(seed)
        arms = {}
        if "synthetic" in args.arms:
            arms["synthetic"] = pool
        if "real" in args.arms:
            arms["real"] = train_real
        if "real_balanced" in args.arms:
            arms["real_balanced"] = rebalanced(train_real, pool.class_counts(), seed)
        if "mixed" in args.arms:
            arms["mixed"] = Corpus(list(train_real) + list(pool))
        if "mixed_tail" in args.arms:
            tail = [r for r in pool if r.label in set(args.tail_classes)]
            arms["mixed_tail"] = Corpus(list(train_real) + tail)
        if "mixed_tail_control" in args.arms:
            # The control that decides whether mixed_tail means anything. Topping up the
            # scarce classes changes TWO things at once: it adds generated images, and it
            # flattens the class distribution. Duplicating real tail images instead adds
            # the same count to the same classes with zero new information, so whatever
            # mixed_tail gains over THIS is attributable to the generated content rather
            # than to the rebalancing. Without it, a gain from mixed_tail is unreadable.
            gen = rng(seed + 9000)
            by_class = train_real.by_class()
            extra = []
            for c in sorted(set(args.tail_classes)):
                want = sum(1 for r in pool if r.label == c)
                have = list(by_class.get(c, []))
                if have and want:
                    idx = gen.choice(len(have), size=want, replace=want > len(have))
                    extra.extend(have[i] for i in idx)
            arms["mixed_tail_control"] = Corpus(list(train_real) + extra)
        # "pretrain" is two-stage and handled below, since it needs the synthetic model.

        for name, corpus in arms.items():
            tc = TrainConfig(**{**asdict(cfg.train), "seed": seed, "num_workers": 4})
            print(f"\n--- {name}, seed {seed}: {len(corpus)} images "
                  f"{corpus.class_counts()} ---", flush=True)
            model, _ = train_model(corpus, val_real, tc)
            result, preds = evaluate_corpus(model, val_real, tc, return_predictions=True)
            row = result.to_dict()
            row["seed"] = seed
            row["n_train"] = len(corpus)
            results.setdefault(name, []).append(row)
            yt, yp = np.array(preds["y_true"]), np.array(preds["y_pred"])
            print(f"  balanced accuracy {result.balanced_accuracy:.4f}  "
                  f"accuracy {result.accuracy:.4f}  QWK {result.qwk:.3f}")
            print(f"  predictions {dict(sorted(Counter(yp.tolist()).items()))}")
            if name == "synthetic":
                synthetic_model = model

        if "pretrain" in args.arms:
            # Stage one is the synthetic arm's own model, reused rather than retrained, so
            # the two arms share a starting point exactly and any difference is stage two.
            tc = TrainConfig(**{**asdict(cfg.train), "seed": seed, "num_workers": 4})
            stage_one = arms.get("synthetic") and synthetic_model
            if stage_one is None:
                print(f"\n--- pretrain stage 1 (synthetic), seed {seed} ---", flush=True)
                stage_one, _ = train_model(pool, val_real, tc)
            print(f"\n--- pretrain stage 2 (real), seed {seed}: "
                  f"{len(train_real)} images ---", flush=True)
            model, _ = train_model(train_real, val_real, tc, init_from=stage_one)
            result, preds = evaluate_corpus(model, val_real, tc, return_predictions=True)
            row = result.to_dict()
            row["seed"] = seed
            row["n_train"] = len(pool) + len(train_real)
            results.setdefault("pretrain", []).append(row)
            print(f"  balanced accuracy {result.balanced_accuracy:.4f}  "
                  f"accuracy {result.accuracy:.4f}  QWK {result.qwk:.3f}")

    print(f"\n{'arm':>16} {'n train':>8} {'bal acc':>9} {'accuracy':>9} {'QWK':>7}")
    for name, rows in results.items():
        b = np.array([r["balanced_accuracy"] for r in rows])
        a = np.array([r["accuracy"] for r in rows])
        q = np.array([r["qwk"] for r in rows])
        sd = f" ±{b.std(ddof=1):.3f}" if len(b) > 1 else ""
        print(f"{name:>16} {rows[0]['n_train']:>8} {b.mean():>9.4f}{sd} "
              f"{a.mean():>9.4f} {q.mean():>7.3f}")

    print(f"\n{'arm':>16} " + " ".join(f"{n:>12}" for n in SEVERITY_NAMES))
    for name, rows in results.items():
        recalls = [r["per_class_recall"] for r in rows]
        means = [np.mean([d[n] for d in recalls]) for n in SEVERITY_NAMES]
        print(f"{name:>16} " + " ".join(f"{m:>12.3f}" for m in means))

    Path("results").mkdir(exist_ok=True)
    Path("results/synthetic_vs_real.json").write_text(json.dumps(results, indent=2))
    print("\nwrote results/synthetic_vs_real.json")


if __name__ == "__main__":
    main()
