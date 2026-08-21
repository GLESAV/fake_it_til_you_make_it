"""Stage 4: rank-correlation readout under exposure perturbation.

Accuracy sits near the majority floor for every label, which makes it a blunt
instrument. Spearman rho between ITA and the ordinal skin-tone label is far more
sensitive, so track how much of it survives an exposure change.
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd
from scipy import stats

ROOT = "/Users/gs/fake_it_til_you_make_it"
TAGS = [("clean", "p00"), ("-0.5 EV", "m05"), ("-0.3 EV", "m03"),
        ("+0.3 EV", "p03"), ("+0.5 EV", "p05")]
FST = ["FST1", "FST2", "FST3", "FST4", "FST5", "FST6"]


def main():
    d = pd.read_csv(f"{ROOT}/data/scin_ita.csv").drop_duplicates(subset="file")
    specs = [
        ("fst_self", d[d.fst_self.isin(FST)], lambda s: s.fst_self.map({k: i for i, k in enumerate(FST)})),
        ("fst_derm", d[d.fst_derm.isin(FST)], lambda s: s.fst_derm.map({k: i for i, k in enumerate(FST)})),
        ("monk_us", d[d.monk_us.notna()], lambda s: s.monk_us.astype(float)),
    ]
    for name, sub, f in specs:
        y = f(sub).values
        print(f"\n-- {name} (n={len(sub)}) --")
        print(f"{'condition':>10} | {'rho raw':>8} | {'rho norm':>9}")
        for label, tag in TAGS:
            r1 = stats.spearmanr(sub[f"ita_raw_{tag}"].values, y).statistic
            r2 = stats.spearmanr(sub[f"ita_norm_{tag}"].values, y).statistic
            print(f"{label:>10} | {r1:+8.3f} | {r2:+9.3f}")
        # how much of the between-image ITA variance is exposure-explainable?
        cols = [f"ita_raw_{t}" for _, t in TAGS]
        within = np.mean(np.var(sub[cols].values, axis=1, ddof=1))
        between = np.var(sub["ita_raw_p00"].values, ddof=1)
        print(f"  variance from a +/-0.5 EV exposure sweep / between-image variance "
              f"= {within/between:.3f}")
        ncols = [f"ita_norm_{t}" for _, t in TAGS]
        nw = np.mean(np.var(sub[ncols].values, axis=1, ddof=1))
        nb = np.var(sub["ita_norm_p00"].values, ddof=1)
        print(f"  same, normalised pipeline                       = {nw/nb:.4f}")


if __name__ == "__main__":
    main()
