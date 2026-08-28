#!/usr/bin/env python
"""Protocol section 8.6, applied to the one arm in this project that beats its real baseline.

The classical arm (docs/09) reports a linear SVM on which the fully-synthetic arm scores
0.468 against the real arm's 0.432 -- +3.6 points, on 28 of 30 seeds. Four of the five
heads show the expected substitution deficit and this one inverts. Protocol section 8.6
exists precisely for that case, and README design commitment 4 forbids reporting it as a
win until the checks below have been run.

The mechanism the checks are looking for: a generator that narrows within-grade variation
produces cleaner prototypes of each class, and prototypes are easier to learn from. The
synthetic arm can then win *without the generator having contributed any information the
real data lacked*. This project hit exactly that while building the simulator
(src/fitymi/data/toy.py), and it is a live candidate explanation for the acne prior's
97.6%. A weak linear head on a compressed input distribution is the situation in which the
effect should be strongest, which is why the inversion appears here and nowhere else.

## Predictions, written before the run

If the linear SVM's inversion is a prototype effect:

  P1. Within-grade scatter of the synthetic pool is LOWER than the real training split's,
      in the same feature space the head sees, and the pool's between-class separation
      relative to that scatter is HIGHER. A generator that had contributed genuine
      variation would not look like this.

  P2. The synthetic-trained head's advantage CONCENTRATES on unambiguous validation cases
      and REVERSES on borderline ones. A prototype-trained model is tuned to a corpus's
      notion of a typical case; that is worth most where the answer was already obvious.

If instead P1 comes back with synthetic scatter at or above real, or P2 comes back flat
across strata, the inversion is not explained by the prototype mechanism and needs a
different account. Either way this script does not decide whether to report the result --
it supplies the evidence section 8.6 requires before that question is opened.

## What is deliberately not here

Section 8.6's third check is "does the win survive on the external validation set (3.4)?"
Section 3.4 names AcneSCU, the deduplicated Fitzpatrick17k acne subset, and the SCIN acne
subset as candidates. None has been acquired -- data/scin_images holds the SCIN sample used
for the ITA replication, which carries no Hayashi severity grade -- so that check cannot be
run and is reported as NOT RUN rather than quietly dropped. The two checks below are
therefore necessary but not sufficient for section 8.6.

Difficulty in check 2 is defined from the REAL TRAINING split only: it must not be a
property of either arm's training data, or the stratification would be circular. The
sealed test split is not touched; everything is scored on the same real subject-disjoint
validation split the classical arm uses.

## Honest sequencing note, per practice R3

Check 2 stratifies WITHIN class, and that was not the first version. The first version
ranked all 218 validation images by margin and cut terciles, which produced strata with
class counts of [15, 42, 15, 1], [23, 43, 7, 0] and [34, 17, 4, 17]. Balanced accuracy
computed over a stratum missing a class entirely, or holding one example of it, is not
comparable to balanced accuracy over another stratum with seventeen -- sklearn warns about
exactly this -- and margin turns out to correlate with grade, so a global stratification is
partly a stratification by class. Since the two arms differ precisely in how their class
distributions are handled, that confound is fatal to the comparison.

The within-class version was therefore adopted after seeing the first version's numbers.
The reason was the class confound and not the direction of the result, and the global
stratification is still reported below as a secondary, labelled as confounded, so a reader
can see both.

    python scripts/prototype_effect.py --seeds 30
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

#: The 234 handcrafted dimensions, in the order classical_baseline.handcrafted_features
#: concatenates them. Split out because "the synthetic pool is more scattered" is only
#: interesting if you can say scattered in WHAT -- colour and lighting are nuisance
#: variation on this task, texture and edges are where lesions live.
FEATURE_BLOCKS = {
    "colour_hist": (0, 192),    # RGB and HSV histograms, 32 bins x 6
    "lab_moments": (192, 201),  # Lab mean/sd/skew x 3
    "lbp_texture": (201, 229),  # uniform LBP at (8,1) and (16,2)
    "edges": (229, 234),        # Sobel statistics
}


def load_classical():
    """Reuse the classical arm's feature extraction and heads verbatim.

    Importing rather than reimplementing is the point: if the two scripts computed features
    differently, a difference between them would be uninterpretable.
    """
    spec = importlib.util.spec_from_file_location(
        "classical_baseline", ROOT / "scripts" / "classical_baseline.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def scatter_stats(X, y, classes):
    """Within-class scatter per class, and the between/within ratio over the whole set.

    Within-class scatter is the mean squared distance to the class centroid, which is the
    trace of the class covariance -- total variance summed over dimensions. Reported per
    class because the compression finding is grade-dependent.
    """
    import numpy as np

    within, counts = {}, {}
    centroids = {}
    for c in classes:
        Xc = X[y == c]
        counts[c] = int(len(Xc))
        if len(Xc) < 2:
            within[c] = float("nan")
            continue
        centroids[c] = Xc.mean(axis=0)
        within[c] = float(((Xc - centroids[c]) ** 2).sum(axis=1).mean())

    present = [c for c in classes if c in centroids]
    grand = np.stack([centroids[c] for c in present]).mean(axis=0)
    n_total = sum(counts[c] for c in present)
    between = float(sum(counts[c] * ((centroids[c] - grand) ** 2).sum()
                        for c in present) / max(n_total, 1))
    within_pooled = float(sum(counts[c] * within[c] for c in present) / max(n_total, 1))
    return {
        "within_per_class": within,
        "counts": counts,
        "within_pooled": within_pooled,
        "between": between,
        # Fisher-style separability. Higher means cleaner prototypes.
        "between_over_within": between / within_pooled if within_pooled else float("nan"),
    }


def difficulty(X_train, y_train, X_val, y_val, classes):
    """Margin of each validation image, from real-train class centroids only.

    margin = distance to the nearest WRONG class centroid - distance to the OWN centroid.
    Large positive means unambiguous; near zero or negative means borderline, i.e. the
    image sits as close to another grade as to its own. This is "inter-grade proximity" in
    section 8.6's phrasing, computed in the feature space the head actually sees.
    """
    import numpy as np

    centroids = {c: X_train[y_train == c].mean(axis=0) for c in classes
                 if (y_train == c).sum() > 0}
    margins = np.empty(len(X_val))
    for i, (x, y) in enumerate(zip(X_val, y_val)):
        d = {c: float(np.sqrt(((x - m) ** 2).sum())) for c, m in centroids.items()}
        own = d.get(y, float("nan"))
        other = min(v for c, v in d.items() if c != y)
        margins[i] = other - own
    return margins


def strata_within_class(margins, y, classes, n_strata: int):
    """Tercile (or n-tile) indices that hold the class composition of the whole split.

    Ranking within class and then cutting is what makes the strata comparable: each one
    receives roughly the same share of every grade, so balanced accuracy is computed over
    the same class set everywhere and the difficulty contrast is not a class contrast.
    """
    import numpy as np

    buckets = [[] for _ in range(n_strata)]
    for c in classes:
        idx = np.flatnonzero(y == c)
        if len(idx) == 0:
            continue
        ranked = idx[np.argsort(margins[idx])]
        for k, part in enumerate(np.array_split(ranked, n_strata)):
            buckets[k].extend(part.tolist())
    return [np.array(sorted(b)) for b in buckets]


def main() -> None:
    import numpy as np
    from scipy import stats
    from sklearn.metrics import balanced_accuracy_score
    from sklearn.preprocessing import StandardScaler

    from fitymi.data.records import Corpus, Record, Source
    from fitymi.utils.seeding import rng

    cb = load_classical()

    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default="data/synthetic/gemini_pool")
    ap.add_argument("--splits", default="data/splits_subject")
    ap.add_argument("--head", default="linsvm",
                    help="the head under test; linsvm is the one that inverts")
    ap.add_argument("--seeds", type=int, default=30)
    ap.add_argument("--strata", type=int, default=3)
    ap.add_argument("--out", default="results/prototype_effect.json")
    args = ap.parse_args()

    def real(split: str) -> Corpus:
        return Corpus(
            Record(**{k: v for k, v in json.loads(l).items() if k != "meta"})
            for l in (Path(args.splits) / f"{split}.jsonl").read_text().splitlines()
            if l.strip())

    train_real, val_real = real("train"), real("val")
    POOL_FILES = sorted(Path(args.pool).glob("*.png"))

    def synthetic(seed: int) -> Corpus:
        """Class-balanced draw from the frozen pool -- identical to the classical arm."""
        records = [Record(path=str(p), label=int(m.group(1)), source=Source.SYNTH_OPEN)
                   for p in POOL_FILES if (m := re.match(r"g(\d)_", p.name))]
        by_class = Corpus(records).by_class()
        per = min(len(v) for v in by_class.values()) if by_class else 0
        gen, out = rng(seed), []
        for c in sorted(by_class):
            items = list(by_class[c])
            out.extend(items[i] for i in gen.permutation(len(items))[:per])
        return Corpus(out)

    cache = ROOT / "data/features_handcrafted.npz"
    all_paths = sorted({r.path for r in list(train_real) + list(val_real)} |
                       {str(p) for p in POOL_FILES})
    X_all = cb.build_feature_table(all_paths, cache, "handcrafted")
    index = {p: i for i, p in enumerate(all_paths)}

    def xy(corpus):
        rows = list(corpus)
        return (X_all[[index[r.path] for r in rows]],
                np.array([r.label for r in rows]))

    Xtr, ytr = xy(train_real)
    Xv, yv = xy(val_real)
    classes = sorted(set(ytr.tolist()) | set(yv.tolist()))

    # One scaler, fitted on the real training split, applied to everything. The head fits
    # its own scaler per arm; here a SHARED frame is required or the two arms' scatters
    # would be measured in different units and could not be compared at all.
    scaler = StandardScaler().fit(Xtr)
    Ztr, Zv = scaler.transform(Xtr), scaler.transform(Xv)

    print(f"real train {len(train_real)} {train_real.class_counts()}")
    print(f"real val   {len(val_real)} {val_real.class_counts()}")
    print(f"pool       {len(POOL_FILES)} files, feature dim {X_all.shape[1]}\n")

    # ---------------------------------------------------- check 1: within-class scatter
    print("CHECK 1 -- within-grade scatter, real train vs synthetic pool "
          "(shared real-train scaling)")
    real_stats = scatter_stats(Ztr, ytr, classes)
    per_seed_synth = []
    for seed in range(args.seeds):
        Xs, ys = xy(synthetic(seed))
        per_seed_synth.append(scatter_stats(scaler.transform(Xs), ys, classes))

    def mean_over_seeds(key, cls=None):
        vals = [(s[key][cls] if cls is not None else s[key]) for s in per_seed_synth]
        return float(np.mean(vals)), float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0

    check1 = {"real": real_stats, "synthetic_mean_over_seeds": {}}
    print(f"  {'grade':>6} {'real':>10} {'synth':>10} {'ratio':>8}   n_real n_synth")
    for c in classes:
        s_mean, _ = mean_over_seeds("within_per_class", c)
        r = real_stats["within_per_class"][c]
        n_s = int(np.mean([s["counts"][c] for s in per_seed_synth]))
        print(f"  {c:>6} {r:>10.1f} {s_mean:>10.1f} {s_mean / r:>8.2f}   "
              f"{real_stats['counts'][c]:>6} {n_s:>7}")
        check1["synthetic_mean_over_seeds"][str(c)] = {
            "within": s_mean, "ratio_to_real": s_mean / r, "n": n_s}

    sw_mean, sw_sd = mean_over_seeds("within_pooled")
    sb_mean, _ = mean_over_seeds("between")
    sr_mean, sr_sd = mean_over_seeds("between_over_within")
    check1["pooled"] = {
        "real_within": real_stats["within_pooled"],
        "synth_within_mean": sw_mean, "synth_within_sd": sw_sd,
        "real_between_over_within": real_stats["between_over_within"],
        "synth_between_over_within_mean": sr_mean,
        "synth_between_over_within_sd": sr_sd,
    }
    print(f"\n  pooled within-class scatter   real {real_stats['within_pooled']:.1f}   "
          f"synth {sw_mean:.1f} +/- {sw_sd:.1f}   "
          f"ratio {sw_mean / real_stats['within_pooled']:.2f}")
    print(f"  between / within (separability) real "
          f"{real_stats['between_over_within']:.3f}   synth {sr_mean:.3f} +/- {sr_sd:.3f}")
    p1 = (sw_mean < real_stats["within_pooled"]
          and sr_mean > real_stats["between_over_within"])
    print(f"  P1 (synth scatter lower AND separability higher): "
          f"{'HOLDS' if p1 else 'DOES NOT HOLD'}")
    check1["p1_holds"] = bool(p1)

    # Where the scatter lives. "The synthetic pool is more scattered" is only a finding if
    # it can be said in what: colour and lighting are nuisance variation on a severity
    # task, texture and edges are where lesions are.
    print(f"\n  by feature block ({'ratio' } = synth within-class scatter / real):")
    print(f"  {'block':>12} {'dims':>5} {'real':>9} {'synth':>9} {'ratio':>7}")
    check1["by_block"] = {}
    for name, (a, b) in FEATURE_BLOCKS.items():
        rb = scatter_stats(Ztr[:, a:b], ytr, classes)["within_pooled"]
        sb = float(np.mean([
            scatter_stats(scaler.transform(xy(synthetic(seed))[0])[:, a:b],
                          xy(synthetic(seed))[1], classes)["within_pooled"]
            for seed in range(min(args.seeds, 10))]))
        print(f"  {name:>12} {b - a:>5} {rb:>9.1f} {sb:>9.1f} {sb / rb:>7.2f}")
        check1["by_block"][name] = {"dims": b - a, "real": rb, "synth": sb,
                                    "ratio": sb / rb}

    # -------------------------------------- check 2: difficulty-stratified val accuracy
    print(f"\nCHECK 2 -- validation stratified by inter-grade margin, {args.strata} strata")
    margins = difficulty(Ztr, ytr, Zv, yv, classes)
    labels = (["borderline", "middle", "unambiguous"] if args.strata == 3
              else [f"stratum {i}" for i in range(args.strata)])

    bounds = strata_within_class(margins, yv, classes, args.strata)
    global_bounds = np.array_split(np.argsort(margins), args.strata)

    print("  primary, ranked WITHIN class so every stratum keeps the split's composition:")
    for name, idx in zip(labels, bounds):
        print(f"  {name:>12}: n={len(idx):3d}  margin "
              f"[{margins[idx].min():.2f}, {margins[idx].max():.2f}]  "
              f"classes {np.bincount(yv[idx], minlength=len(classes)).tolist()}")
    print("  secondary, ranked globally -- CONFOUNDED, class composition differs by "
          "stratum:")
    for name, idx in zip(labels, global_bounds):
        print(f"  {name:>12}: n={len(idx):3d}  "
              f"classes {np.bincount(yv[idx], minlength=len(classes)).tolist()}")

    per_stratum: dict[str, dict[str, list[float]]] = {
        name: {"real": [], "synthetic": []} for name in labels}
    per_global: dict[str, dict[str, list[float]]] = {
        name: {"real": [], "synthetic": []} for name in labels}
    overall: dict[str, list[float]] = {"real": [], "synthetic": []}

    for seed in range(args.seeds):
        pool = synthetic(seed)
        for arm, corpus in (("real", train_real), ("synthetic", pool)):
            X, y = xy(corpus)
            head = cb.make_models(seed)[args.head]
            head.fit(X, y)
            pred = head.predict(Xv)
            overall[arm].append(balanced_accuracy_score(yv, pred))
            for name, idx in zip(labels, bounds):
                per_stratum[name][arm].append(
                    balanced_accuracy_score(yv[idx], pred[idx]))
            for name, idx in zip(labels, global_bounds):
                per_global[name][arm].append(
                    balanced_accuracy_score(yv[idx], pred[idx]))
        if (seed + 1) % 10 == 0:
            print(f"    {seed + 1}/{args.seeds} seeds", flush=True)

    def paired(a, b):
        d = np.array(b) - np.array(a)
        t, p = stats.ttest_rel(b, a)
        return {"mean": float(d.mean()), "sd": float(d.std(ddof=1)),
                "t": float(t), "p": float(p),
                "positive_seeds": int((d > 0).sum()), "n": len(d)}

    check2 = {"overall": paired(overall["real"], overall["synthetic"]), "strata": {}}
    print(f"\n  {'stratum':>12} {'real':>8} {'synth':>8} {'synth-real':>11} {'p':>8}")
    print(f"  {'OVERALL':>12} {np.mean(overall['real']):>8.3f} "
          f"{np.mean(overall['synthetic']):>8.3f} "
          f"{check2['overall']['mean']:>+11.3f} {check2['overall']['p']:>8.4f}")
    gaps = []
    for name in labels:
        st = paired(per_stratum[name]["real"], per_stratum[name]["synthetic"])
        check2["strata"][name] = {
            "real_mean": float(np.mean(per_stratum[name]["real"])),
            "synthetic_mean": float(np.mean(per_stratum[name]["synthetic"])),
            **st}
        gaps.append(st["mean"])
        print(f"  {name:>12} {np.mean(per_stratum[name]['real']):>8.3f} "
              f"{np.mean(per_stratum[name]['synthetic']):>8.3f} "
              f"{st['mean']:>+11.3f} {st['p']:>8.4f}")

    # P2 asks for a monotone gradient: the advantage grows from borderline to unambiguous.
    p2 = all(gaps[i] < gaps[i + 1] for i in range(len(gaps) - 1))
    print(f"\n  P2 (advantage rises from borderline to unambiguous): "
          f"{'HOLDS' if p2 else 'DOES NOT HOLD'}  gaps {[round(g, 4) for g in gaps]}")
    check2["p2_holds"] = bool(p2)

    check2["global_stratification_confounded"] = {}
    print(f"\n  secondary (globally ranked, class composition differs -- not comparable "
          f"across strata):")
    for name in labels:
        st = paired(per_global[name]["real"], per_global[name]["synthetic"])
        check2["global_stratification_confounded"][name] = {
            "real_mean": float(np.mean(per_global[name]["real"])),
            "synthetic_mean": float(np.mean(per_global[name]["synthetic"])), **st}
        print(f"  {name:>12} {np.mean(per_global[name]['real']):>8.3f} "
              f"{np.mean(per_global[name]['synthetic']):>8.3f} "
              f"{st['mean']:>+11.3f} {st['p']:>8.4f}")

    # ------------------------------------------- exploratory: where does the win come from
    # Post-hoc and labelled as such. Both registered predictions failed, so the inversion
    # needs some account, and check 1 supplies a candidate: the excess scatter is almost
    # entirely in the 192 colour-histogram dimensions. ACNE04 is one cohort under one
    # capture protocol, so a linear head fitted on it can lean on colour that does not
    # generalise; a pool prompted across Fitzpatrick I-VI, lighting and backdrops cannot.
    # If that is the mechanism, removing the colour histograms should shrink the inversion.
    # This is a hypothesis with a first test, not a finding -- this project's record on
    # causal claims is nought for three (audit 15.1).
    print("\nEXPLORATORY (post-hoc) -- drop the colour-histogram block and refit")
    a, b = FEATURE_BLOCKS["colour_hist"]
    keep = np.r_[0:a, b:X_all.shape[1]]
    abl = {"real": [], "synthetic": []}
    for seed in range(args.seeds):
        for arm, corpus in (("real", train_real), ("synthetic", synthetic(seed))):
            X, y = xy(corpus)
            head = cb.make_models(seed)[args.head]
            head.fit(X[:, keep], y)
            abl[arm].append(balanced_accuracy_score(yv, head.predict(Xv[:, keep])))
    ablated = paired(abl["real"], abl["synthetic"])
    full = check2["overall"]
    print(f"  {'':>12} {'real':>8} {'synth':>8} {'synth-real':>11} {'p':>8}")
    print(f"  {'all 234 dims':>12} {np.mean(overall['real']):>8.3f} "
          f"{np.mean(overall['synthetic']):>8.3f} {full['mean']:>+11.3f} "
          f"{full['p']:>8.4f}")
    print(f"  {'no colour':>12} {np.mean(abl['real']):>8.3f} "
          f"{np.mean(abl['synthetic']):>8.3f} {ablated['mean']:>+11.3f} "
          f"{ablated['p']:>8.4f}   ({len(keep)} dims)")
    flipped = (full["mean"] > 0) != (ablated["mean"] > 0)
    print(f"  sign {'FLIPS' if flipped else 'holds'}: "
          f"{full['mean']:+.3f} -> {ablated['mean']:+.3f}. The real arm gains "
          f"{np.mean(abl['real']) - np.mean(overall['real']):+.3f} from losing 192 of 234 "
          f"dimensions;\n  the synthetic arm loses "
          f"{np.mean(abl['synthetic']) - np.mean(overall['synthetic']):+.3f}.")

    # What this ablation is and is not. Removing 192 named columns needs no verification --
    # they are gone by construction, and this is NOT the audit 15.2 situation, where a
    # channel was declared removed and turned out to be recoverable. The useful question is
    # a different one: does dropping colour close the gap between the two corpora? If it
    # did, the sign flip would just be the domain gap narrowing rather than anything about
    # what the head learned.
    from sklearn.model_selection import cross_val_score
    src_y = np.r_[np.zeros(len(Xtr)), np.ones(len(xy(synthetic(0))[0]))]
    src_X = np.vstack([Xtr, xy(synthetic(0))[0]])
    rec_full = float(cross_val_score(cb.make_models(0)[args.head], src_X, src_y,
                                     cv=3, scoring="balanced_accuracy").mean())
    rec_kept = float(cross_val_score(cb.make_models(0)[args.head], src_X[:, keep], src_y,
                                     cv=3, scoring="balanced_accuracy").mean())
    print(f"  real-vs-synthetic discriminability: {rec_full:.3f} on all 234 dims, "
          f"{rec_kept:.3f} on the kept {len(keep)}")
    print("  It does not. The two corpora stay all but perfectly separable without colour,"
          "\n  so the sign flip is not the domain gap closing -- the pool is out of domain"
          "\n  in texture and edges too, and colour is not what marks it as synthetic.")
    exploratory = {"full": full, "colour_removed": ablated, "kept_dims": int(len(keep)),
                   "retained_fraction": float(ablated["mean"] / full["mean"]),
                   "source_discriminability_all_dims": rec_full,
                   "source_discriminability_kept_dims": rec_kept}

    print("\nCHECK 3 -- external validation set: NOT RUN. Protocol 3.4 names AcneSCU, the "
          "deduplicated\n  Fitzpatrick17k acne subset and the SCIN acne subset; none has "
          "been acquired with Hayashi\n  grades, so the transfer question stays open and "
          "section 8.6 is not fully discharged.")

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "head": args.head, "seeds": args.seeds, "features": "handcrafted",
        "pool": args.pool, "pool_files": len(POOL_FILES),
        "check1_within_class_scatter": check1,
        "check2_difficulty_stratification": check2,
        "check3_external_validation": "NOT RUN -- no external set with Hayashi grades",
        "exploratory_colour_ablation": exploratory,
        "verdict_note": ("P1 and P2 are the two runnable signatures of the prototype "
                         "mechanism in protocol 8.6. Both holding is evidence the linear "
                         "SVM inversion is a prototype effect rather than a contribution "
                         "from the generator; 8.6 is still not discharged without check 3."),
    }, indent=1))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
