"""Stage 2: replication analysis (A)-(D) on SCIN. Reads data/scin_ita.csv."""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, "/Users/gs/fake_it_til_you_make_it/src")
from fitymi.controls.skintone import ITA_BINS  # noqa: E402

ROOT = "/Users/gs/fake_it_til_you_make_it"
EV_TAGS = {-0.5: "m05", -0.3: "m03", 0.0: "p00", 0.3: "p03", 0.5: "p05"}
PERTURB = [-0.5, -0.3, 0.3, 0.5]


def tone_bin(ita):
    ita = np.asarray(ita)
    out = np.full(ita.shape, "dark", dtype=object)
    for label, lo, hi in ITA_BINS:
        out[(ita >= lo) & (ita < hi)] = label
    return out


def cv_accuracy(x_fit, x_pred, y, seed=0, n_splits=5):
    """5-fold CV. Fits on x_fit[train], predicts x_pred[test]. Both out-of-sample."""
    y = np.asarray(y)
    x_fit = np.asarray(x_fit, dtype=float).reshape(-1, 1)
    x_pred = np.asarray(x_pred, dtype=float).reshape(-1, 1)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    correct = 0
    for tr, te in skf.split(x_fit, y):
        clf = LogisticRegression(max_iter=2000, C=1.0)
        clf.fit(x_fit[tr], y[tr])
        correct += int((clf.predict(x_pred[te]) == y[te]).sum())
    return correct / len(y)


def majority_floor(y):
    v = pd.Series(y).value_counts()
    return v.iloc[0] / v.sum()


def separation(ita, y, order):
    med = {k: float(np.median(ita[y == k])) for k in order if (y == k).sum() > 1}
    sd = {k: float(np.std(ita[y == k], ddof=1)) for k in order if (y == k).sum() > 1}
    ks = [k for k in order if k in med]
    gaps = [abs(med[ks[i + 1]] - med[ks[i]]) for i in range(len(ks) - 1)]
    pooled_sd = float(np.mean(list(sd.values())))
    return med, sd, gaps, pooled_sd


def main():
    df = pd.read_csv(f"{ROOT}/data/scin_ita.csv").drop_duplicates(subset="file")
    print(f"SCIN sample: {len(df)} unique images\n")
    print("self-reported FST counts:")
    print(df.fst_self.value_counts().sort_index().to_string(), "\n")

    # ---------------- (A) EXPOSURE SENSITIVITY ----------------
    print("=" * 72)
    print("(A) EXPOSURE SENSITIVITY  [linear-light exposure change]")
    print("=" * 72)
    for pipe in ("raw", "norm"):
        base = df[f"ita_{pipe}_p00"].values
        base_bin = tone_bin(base)
        print(f"\n-- pipeline: {pipe} --")
        print(f"{'EV':>6} {'mean dITA':>10} {'mean|dITA|':>11} {'max|dITA|':>10} "
              f"{'%bin change':>12}")
        all_abs = []
        all_flip = []
        for ev in PERTURB:
            p = df[f"ita_{pipe}_{EV_TAGS[ev]}"].values
            d = p - base
            flip = (tone_bin(p) != base_bin)
            all_abs.append(np.abs(d))
            all_flip.append(flip)
            print(f"{ev:+6.1f} {d.mean():10.2f} {np.abs(d).mean():11.2f} "
                  f"{np.abs(d).max():10.2f} {100*flip.mean():11.1f}%")
        any_flip = np.any(np.stack(all_flip), axis=0)
        pooled = np.concatenate(all_abs)
        print(f"{'POOLED':>6} {'':>10} {pooled.mean():11.2f} {pooled.max():10.2f} "
              f"{100*np.mean(np.stack(all_flip)):11.1f}%")
        print(f"  images whose bin flips under AT LEAST ONE of +/-0.3,+/-0.5 EV: "
              f"{100*any_flip.mean():.1f}%")
        print(f"  ITA range across the 5 exposures, mean: "
              f"{np.mean([df[[f'ita_{pipe}_{t}' for t in EV_TAGS.values()]].max(axis=1) - df[[f'ita_{pipe}_{t}' for t in EV_TAGS.values()]].min(axis=1)]):.2f} deg")
        print(f"  between-image SD of clean ITA (the signal): {base.std(ddof=1):.2f} deg")

    # ---------------- (B) GROUND-TRUTH RECOVERY ----------------
    print("\n" + "=" * 72)
    print("(B) GROUND-TRUTH RECOVERY")
    print("=" * 72)
    label_specs = [
        ("fst_self (self-reported, NOT pixel-derived)", "fst_self",
         ["FST1", "FST2", "FST3", "FST4", "FST5", "FST6"]),
        ("fst_derm (dermatologist, rated FROM pixels)", "fst_derm",
         ["FST1", "FST2", "FST3", "FST4", "FST5", "FST6"]),
        ("monk_us (Monk scale, rated FROM pixels)", "monk_us", None),
    ]
    results_b = {}
    for name, col, order in label_specs:
        sub = df[df[col].notna()].copy()
        if col == "monk_us":
            sub[col] = sub[col].astype(int).astype(str)
            order = sorted(sub[col].unique(), key=int)
        sub = sub[sub[col].isin(order)]
        y = sub[col].values
        keep = pd.Series(y).map(pd.Series(y).value_counts()) >= 5
        sub, y = sub[keep.values], y[keep.values]
        ita = sub["ita_raw_p00"].values
        acc = cv_accuracy(ita, ita, y)
        floor = majority_floor(y)
        med, sd, gaps, psd = separation(ita, y, order)
        ordinal = pd.Series(y).map({k: i for i, k in enumerate(order)}).values
        rho, pv = stats.spearmanr(ita, ordinal)
        print(f"\n-- {name} --   n={len(y)}, {len(set(y))} classes")
        print(f"  5-fold CV accuracy (logreg on ITA alone): {acc:.3f}")
        print(f"  majority-class floor:                     {floor:.3f}")
        print(f"  lift over floor:                          {acc-floor:+.3f}")
        print(f"  Spearman rho(ITA, label ordinal):         {rho:+.3f} (p={pv:.2e})")
        print("  per-label median ITA / within-label SD:")
        for k in order:
            if k in med:
                print(f"    {k:>6}: n={int((y==k).sum()):4d}  median={med[k]:7.2f}  SD={sd[k]:6.2f}")
        print(f"  mean |median gap| between adjacent labels: {np.mean(gaps):.2f} deg")
        print(f"  mean within-label SD:                      {psd:.2f} deg")
        print(f"  RATIO gap/SD:                              {np.mean(gaps)/psd:.3f}")
        results_b[col] = dict(acc=acc, floor=floor, y=y, sub=sub, order=order,
                              gap=np.mean(gaps), sd=psd, rho=rho)

    # ---------------- (C) NORMALISATION ----------------
    print("\n" + "=" * 72)
    print("(C) NORMALISATION (gray-world WB + p90 luminance -> 0.7, linear light)")
    print("=" * 72)
    print("Protocol: every number below is OUT-OF-SAMPLE. The ITA->label map is fit")
    print("on CLEAN images from the 4 training folds; the held-out fold is scored")
    print("either clean or perturbed. Perturbed images NEVER appear in any fit.")
    for col in ("fst_self", "fst_derm", "monk_us"):
        r = results_b[col]
        sub, y, order = r["sub"], r["y"], r["order"]
        print(f"\n-- label: {col} (n={len(y)}, floor={r['floor']:.3f}) --")
        print(f"{'test condition':>22} | {'raw ITA':>9} | {'normalised':>10}")
        for cond, tag in [("clean (0 EV)", "p00"), ("-0.5 EV", "m05"),
                          ("-0.3 EV", "m03"), ("+0.3 EV", "p03"), ("+0.5 EV", "p05")]:
            row = []
            for pipe in ("raw", "norm"):
                fit_x = sub[f"ita_{pipe}_p00"].values      # CLEAN, train folds
                pred_x = sub[f"ita_{pipe}_{tag}"].values   # perturbed, test fold
                row.append(cv_accuracy(fit_x, pred_x, y))
            print(f"{cond:>22} | {row[0]:9.3f} | {row[1]:10.3f}")
        # separation stats under normalisation
        med, sd, gaps, psd = separation(sub["ita_norm_p00"].values, y, order)
        rho, _ = stats.spearmanr(sub["ita_norm_p00"].values,
                                 pd.Series(y).map({k: i for i, k in enumerate(order)}).values)
        print(f"  normalised: gap={np.mean(gaps):.2f} SD={psd:.2f} "
              f"RATIO={np.mean(gaps)/psd:.3f}  rho={rho:+.3f}  "
              f"(raw ratio was {r['gap']/r['sd']:.3f}, rho {r['rho']:+.3f})")

    # ---------------- (D) DEVICE / CAPTURE CONFOUND ----------------
    print("\n" + "=" * 72)
    print("(D) DEVICE / CAPTURE-CONDITION CONFOUND")
    print("=" * 72)
    print("SCIN images carry no EXIF and a single `source`. Proxies used: shot_type")
    print("(CLOSE_UP / AT_AN_ANGLE / AT_DISTANCE, a real capture-condition field) and")
    print("image geometry (megapixels, aspect ratio) as a device/orientation proxy.\n")
    d = df[df.shot_type.notna()].copy()
    ita = d["ita_raw_p00"].values
    for proxy, name in [(d.shot_type.values, "shot_type"),
                        (pd.qcut(d.megapixels, 3, labels=["lo", "mid", "hi"],
                                 duplicates="drop").astype(str).values, "megapixel tertile"),
                        ((d.aspect > 1).map({True: "landscape", False: "portrait"}).values,
                         "orientation")]:
        acc = cv_accuracy(ita, ita, proxy)
        fl = majority_floor(proxy)
        f, p = stats.f_oneway(*[ita[proxy == k] for k in np.unique(proxy)])
        print(f"ITA -> {name:20s}: CV acc {acc:.3f} vs floor {fl:.3f} "
              f"(lift {acc-fl:+.3f})   ANOVA F={f:.2f} p={p:.2e}")
        for k in np.unique(proxy):
            v = ita[proxy == k]
            print(f"     {str(k):>14}: n={len(v):4d} median ITA {np.median(v):7.2f}")
    print("\nDoes the FST<->ITA association survive stratification by capture condition?")
    sub = d[d.fst_self.notna()]
    order = ["FST1", "FST2", "FST3", "FST4", "FST5", "FST6"]
    omap = {k: i for i, k in enumerate(order)}
    rho_all, _ = stats.spearmanr(sub.ita_raw_p00.values, sub.fst_self.map(omap).values)
    print(f"  pooled Spearman rho(ITA, self-FST) = {rho_all:+.3f}  (n={len(sub)})")
    for st, g in sub.groupby("shot_type"):
        if len(g) < 30:
            continue
        rho, p = stats.spearmanr(g.ita_raw_p00.values, g.fst_self.map(omap).values)
        acc = cv_accuracy(g.ita_raw_p00.values, g.ita_raw_p00.values, g.fst_self.values)
        print(f"  {st:>12}: n={len(g):4d} rho={rho:+.3f} (p={p:.1e})  "
              f"CV acc={acc:.3f} vs floor {majority_floor(g.fst_self.values):.3f}")


if __name__ == "__main__":
    main()
