#!/usr/bin/env python
"""The classical-ML arm: does the compression finding survive a classifier that cannot
learn its own features?

Every classifier result in this project so far is a ResNet-50 -- a model that learns its
own representation. That leaves an ambiguity in the project's own question. "Can generated
images train a traditional classifier?" has been answered for supervised *deep* classifiers;
it has not been answered for the classical sense of the phrase, where a fixed feature
extractor feeds an SVM, a random forest or a gradient-boosted ensemble.

The reason this is worth an arm rather than a footnote is the mechanism (§12.9). The
generator compresses four severity grades into 41% of the real severity range. A deep
network can partly route around a compressed input distribution by learning features that
magnify whatever variation remains. A classical head over frozen features cannot: it sees
the compression directly. So the prediction is that the substitution deficit is *larger*
here, not smaller -- and if it comes back smaller, the compression story is wrong in a way
that matters.

Two feature regimes, reported separately and never pooled:

  handcrafted  Colour histograms, colour moments, LBP texture, edge statistics. No learned
               component whatsoever. This is the only regime that is honestly "classical":
               nothing in it has ever seen a photograph other than the ones passed in.

  embedding    The cached 512-d embeddings in data/splits_subject/. Faster and stronger,
               but those weights consumed a large corpus of real photographs, so a
               "100% synthetic" arm built on them is not 100% synthetic. This is the same
               objection the study already makes to ImageNet-pretrained backbones
               (README design commitment 4), and it is honoured the same way: separate
               tables, no pooling.

Arms are constructed by the same code path as scripts/train_on_synthetic.py, so the only
thing that varies between the two studies is the classifier. Budgets stay matched; the
sealed test split is not touched; everything is scored on the real subject-disjoint
validation split.

The cheap thing this buys: classical heads fit in milliseconds, so the seed counts that are
prohibitive for the deep tail arm (22 seeds for 80% power, §13.1) are free here.

    python scripts/classical_baseline.py --features handcrafted --seeds 30
    python scripts/classical_baseline.py --features embedding --seeds 30
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))


# ----------------------------------------------------------------- features
def handcrafted_features(path: str, size: int = 128):
    """~230 dims of colour and texture. No learned component.

    Chosen to be the features a 2010-era paper would have used on this task: colour
    distribution (acne is erythematous, so the red channel carries signal), colour moments
    in a perceptual space, local binary patterns at two scales for lesion texture, and
    gradient statistics for lesion edges.
    """
    import numpy as np
    from PIL import Image
    from skimage.color import rgb2hsv, rgb2lab
    from skimage.feature import local_binary_pattern
    from skimage.filters import sobel

    img = Image.open(path).convert("RGB").resize((size, size), Image.BILINEAR)
    rgb = np.asarray(img, dtype=np.float64) / 255.0
    hsv, lab = rgb2hsv(rgb), rgb2lab(rgb)
    grey = rgb.mean(axis=2)

    feats = []
    for ch in range(3):  # RGB and HSV histograms, 32 bins each
        feats.append(np.histogram(rgb[:, :, ch], bins=32, range=(0, 1), density=True)[0])
        feats.append(np.histogram(hsv[:, :, ch], bins=32, range=(0, 1), density=True)[0])
    for ch in range(3):  # Lab colour moments
        v = lab[:, :, ch].ravel()
        m, s = v.mean(), v.std() + 1e-8
        feats.append(np.array([m, s, (((v - m) / s) ** 3).mean()]))
    for P, R in ((8, 1), (16, 2)):  # LBP texture at two scales
        lbp = local_binary_pattern(grey, P, R, method="uniform")
        feats.append(np.histogram(lbp, bins=P + 2, range=(0, P + 2), density=True)[0])
    edges = sobel(grey)  # lesion edge statistics
    feats.append(np.array([edges.mean(), edges.std(), (edges > 0.1).mean(),
                           np.percentile(edges, 90), np.percentile(edges, 99)]))
    return np.concatenate(feats).astype(np.float32)


def build_feature_table(paths, cache: Path, regime: str):
    """Extract (or load) features for every path. Cached by regime; extraction dominates."""
    import numpy as np

    if regime == "embedding":
        d = np.load(ROOT / "data/splits_subject/embeddings.npz", allow_pickle=True)
        table = {str(k): v for k, v in zip(d["keys"], d["vecs"])}
        missing = [p for p in paths if Path(p).name not in table and str(p) not in table]
        if missing:
            raise SystemExit(
                f"{len(missing)} images have no cached embedding (e.g. {missing[0]}).\n"
                "The cache covers the real corpus only, so --features embedding cannot "
                "score synthetic arms. Use --features handcrafted, or extend the cache.")
        return np.stack([table.get(str(p), table.get(Path(p).name)) for p in paths])

    cached = {}
    if cache.exists():
        d = np.load(cache, allow_pickle=True)
        cached = {str(k): v for k, v in zip(d["keys"], d["vecs"])}
    todo = [p for p in paths if str(p) not in cached]
    if todo:
        t0 = time.time()
        for i, p in enumerate(todo, 1):
            cached[str(p)] = handcrafted_features(p)
            if i % 200 == 0 or i == len(todo):
                print(f"  features {i}/{len(todo)}  {time.time() - t0:.0f}s", flush=True)
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache, keys=np.array(list(cached)),
                            vecs=np.stack(list(cached.values())))
    return np.stack([cached[str(p)] for p in paths])


# ------------------------------------------------------------------- models
def make_models(seed: int):
    from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC

    # class_weight="balanced" throughout: the arms differ in class distribution by design,
    # and an unweighted head would report the rebalancing rather than the images.
    return {
        "logreg": make_pipeline(StandardScaler(), LogisticRegression(
            max_iter=2000, class_weight="balanced", random_state=seed)),
        "linsvm": make_pipeline(StandardScaler(), SVC(
            kernel="linear", class_weight="balanced", random_state=seed)),
        "rbfsvm": make_pipeline(StandardScaler(), SVC(
            kernel="rbf", class_weight="balanced", random_state=seed)),
        "rf": RandomForestClassifier(
            n_estimators=400, class_weight="balanced", n_jobs=-1, random_state=seed),
        "hgb": HistGradientBoostingClassifier(
            max_iter=300, class_weight="balanced", random_state=seed),
    }


# --------------------------------------------------------------------- main
def main() -> None:
    import numpy as np
    from scipy import stats
    from sklearn.metrics import balanced_accuracy_score

    from fitymi.data.records import NUM_CLASSES, Corpus, Record, Source
    from fitymi.utils.seeding import rng

    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default="data/synthetic/gemini_pool")
    ap.add_argument("--splits", default="data/splits_subject")
    ap.add_argument("--features", choices=["handcrafted", "embedding"],
                    default="handcrafted")
    ap.add_argument("--seeds", type=int, default=30)
    ap.add_argument("--arms", nargs="+",
                    default=["real", "synthetic", "real_balanced",
                             "mixed_tail", "mixed_tail_control"])
    ap.add_argument("--tail-classes", type=int, nargs="+", default=[2, 3])
    ap.add_argument("--balance", action="store_true", default=True)
    ap.add_argument("--out", default="results/classical_arm.json")
    args = ap.parse_args()

    def real(split: str) -> Corpus:
        return Corpus(
            Record(**{k: v for k, v in json.loads(l).items() if k != "meta"})
            for l in (Path(args.splits) / f"{split}.jsonl").read_text().splitlines()
            if l.strip())

    # Snapshot the pool once, for the reason given in train_on_synthetic.py.
    POOL_FILES = sorted(Path(args.pool).glob("*.png"))

    def synthetic(seed: int) -> Corpus:
        records = [Record(path=str(p), label=int(m.group(1)), source=Source.SYNTH_OPEN)
                   for p in POOL_FILES if (m := re.match(r"g(\d)_", p.name))]
        corpus = Corpus(records)
        if not args.balance:
            return corpus
        by_class = corpus.by_class()
        per = min(len(v) for v in by_class.values()) if by_class else 0
        gen, out = rng(seed), []
        for c in sorted(by_class):
            items = list(by_class[c])
            out.extend(items[i] for i in gen.permutation(len(items))[:per])
        return Corpus(out)

    train_real, val_real = real("train"), real("val")
    print(f"real train {len(train_real)} {train_real.class_counts()}")
    print(f"real val   {len(val_real)} {val_real.class_counts()}  <- every arm scored here")
    print(f"pool       {len(POOL_FILES)} files in {args.pool}")

    cache = ROOT / f"data/features_{args.features}.npz"
    all_paths = sorted({r.path for r in list(train_real) + list(val_real)} |
                       {str(p) for p in POOL_FILES})
    print(f"\nfeatures: {args.features} ({len(all_paths)} images, cache {cache.name})")
    X_all = build_feature_table(all_paths, cache, args.features)
    index = {p: i for i, p in enumerate(all_paths)}
    print(f"feature dim {X_all.shape[1]}")

    def xy(corpus):
        rows = list(corpus)
        return (X_all[[index[r.path] for r in rows]],
                np.array([r.label for r in rows]))

    Xv, yv = xy(val_real)

    def rebalanced(corpus, target, seed):
        gen, by_class, out = rng(seed), corpus.by_class(), []
        for c in range(NUM_CLASSES):
            want, have = target.get(c, 0), list(by_class.get(c, []))
            if have and want:
                idx = gen.choice(len(have), size=want, replace=want > len(have))
                out.extend(have[i] for i in idx)
        return Corpus(out)

    scores: dict[str, dict[str, list[float]]] = {}
    for seed in range(args.seeds):
        pool = synthetic(seed)
        arms = {}
        if "real" in args.arms:
            arms["real"] = train_real
        if "synthetic" in args.arms:
            arms["synthetic"] = pool
        if "real_balanced" in args.arms:
            arms["real_balanced"] = rebalanced(train_real, pool.class_counts(), seed)
        if "mixed_tail" in args.arms:
            tail = [r for r in pool if r.label in set(args.tail_classes)]
            arms["mixed_tail"] = Corpus(list(train_real) + tail)
        if "mixed_tail_control" in args.arms:
            # Same count, same classes, zero new information -- see train_on_synthetic.py.
            gen, by_class, extra = rng(seed + 9000), train_real.by_class(), []
            for c in sorted(set(args.tail_classes)):
                want, have = sum(1 for r in pool if r.label == c), list(by_class.get(c, []))
                if have and want:
                    idx = gen.choice(len(have), size=want, replace=want > len(have))
                    extra.extend(have[i] for i in idx)
            arms["mixed_tail_control"] = Corpus(list(train_real) + extra)

        for arm, corpus in arms.items():
            Xt, yt = xy(corpus)
            for name, model in make_models(seed).items():
                model.fit(Xt, yt)
                ba = balanced_accuracy_score(yv, model.predict(Xv))
                scores.setdefault(arm, {}).setdefault(name, []).append(float(ba))
        print(f"seed {seed:>2}  " + "  ".join(
            f"{a}:{np.mean([scores[a][m][-1] for m in scores[a]]):.3f}" for a in arms),
            flush=True)

    # ---------------------------------------------------------------- report
    models = list(next(iter(scores.values())))
    print(f"\nbalanced accuracy on the real validation split, {args.seeds} seeds "
          f"({args.features} features)\n")
    head = f"{'arm':<22}" + "".join(f"{m:>16}" for m in models)
    print(head + "\n" + "-" * len(head))
    for arm in scores:
        row = f"{arm:<22}"
        for m in models:
            v = np.array(scores[arm][m])
            row += f"{v.mean():>10.4f}±{v.std():.3f}"
        print(row)

    summary = {"features": args.features, "seeds": args.seeds, "pool": args.pool,
               "n_pool_files": len(POOL_FILES), "feature_dim": int(X_all.shape[1]),
               "scores": scores, "contrasts": {}}

    def contrast(a: str, b: str, label: str) -> None:
        if a not in scores or b not in scores:
            return
        print(f"\n{label}")
        for m in models:
            d = np.array(scores[a][m]) - np.array(scores[b][m])
            t, p = stats.ttest_rel(scores[a][m], scores[b][m])
            lo, hi = stats.t.interval(0.95, len(d) - 1, d.mean(),
                                      stats.sem(d) if d.std() else 1e-12)
            print(f"  {m:<8} {d.mean():+.4f}  sd {d.std(ddof=1):.4f}  "
                  f"t={t:+.2f}  p={p:.3f}  95% CI [{lo:+.4f}, {hi:+.4f}]  "
                  f"{int((d > 0).sum())}/{len(d)} positive")
            summary["contrasts"].setdefault(label, {})[m] = {
                "mean": float(d.mean()), "sd": float(d.std(ddof=1)),
                "t": float(t), "p": float(p), "ci95": [float(lo), float(hi)],
                "positive_seeds": int((d > 0).sum()), "n": int(len(d))}

    contrast("synthetic", "real", "Substitution: synthetic-only minus real")
    contrast("mixed_tail", "mixed_tail_control",
             "Paired content effect: generated tail minus duplicated real tail")
    contrast("real_balanced", "real", "Rebalancing alone: balanced real minus real")

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=1))
    print(f"\nwrote {args.out}")
    print("\nThis arm answers the classical question only. It does not transfer to the "
          "deep result and must not be pooled with it (practices R2).")


if __name__ == "__main__":
    main()
