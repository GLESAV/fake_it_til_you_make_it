#!/usr/bin/env python
"""Use the generated pool as a calibration standard for ITA.

Every real dermatology corpus has the same problem when you try to validate a skin-tone
instrument: there is no trustworthy ground truth to validate it against. Self-reported
Fitzpatrick is noisy, dermatologist-assigned Fitzpatrick disagrees between raters, and both
are read off the same photographs whose exposure is the thing in question.

The generated pool has a property no real corpus has: the tone was **specified before the
image existed**. That does not make it a gold standard for ITA's *accuracy* -- the
generator is the only guarantor that a requested type VI was rendered as a type VI, which
is circular. But it is a sound standard for ITA's *robustness*, because perturbing the
image does not perturb the label. If ITA recovers the requested type at rate A on clean
images and A' after an exposure shift no human would notice, then A - A' is attributable to
the instrument regardless of how biased A is.

That is the useful thing generated data can do here, and it is not the thing the project
set out to test.
"""
from __future__ import annotations

import json
import re
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_predict

from fitymi.controls.skintone import COARSE, ITA_BINS, estimate_ita

FITZ = ["I", "II", "III", "IV", "V", "VI"]
#: Fitzpatrick I-VI collapsed to the three groups fairness audits actually report.
FITZ_COARSE = {0: "light", 1: "light", 2: "intermediate", 3: "intermediate", 4: "dark", 5: "dark"}


def to_linear(a: np.ndarray) -> np.ndarray:
    a = a / 255.0
    return np.where(a > 0.04045, ((a + 0.055) / 1.055) ** 2.4, a / 12.92)


def to_srgb(linear: np.ndarray) -> np.ndarray:
    linear = np.clip(linear, 0, 1)
    s = np.where(linear > 0.0031308, 1.055 * linear ** (1 / 2.4) - 0.055, 12.92 * linear)
    return np.clip(s * 255, 0, 255).astype(np.uint8)


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default="data/synthetic/gemini_pool")
    ap.add_argument("--ev", type=float, nargs="+", default=[-0.5, -0.3, 0.3, 0.5])
    args = ap.parse_args()

    manifest = Path(args.pool) / "manifest.jsonl"
    items = []
    for line in manifest.read_text().splitlines():
        r = json.loads(line)
        if not r.get("path") or not Path(r["path"]).exists():
            continue
        m = re.search(r"Fitzpatrick (I{1,3}V?|IV|VI?)\b", r["prompt"])
        if m:
            items.append((FITZ.index(m.group(1)), r["path"]))
    if not items:
        raise SystemExit("no Fitzpatrick-tagged images found in the manifest")

    y = np.array([f for f, _ in items])
    tmp = Path(tempfile.mkdtemp())

    def ita_under(ev: float) -> np.ndarray:
        out = []
        for _, path in items:
            with Image.open(path) as im:
                arr = np.asarray(im.convert("RGB").resize((256, 256), Image.LANCZOS),
                                 dtype=np.float32)
            if ev == 0.0:
                f = path
            else:
                f = tmp / "x.png"
                Image.fromarray(to_srgb(to_linear(arr) * 2 ** ev)).save(f)
            out.append(estimate_ita(str(f)).ita)
        return np.array(out)

    clean = ita_under(0.0)

    def accuracy(ita: np.ndarray, coarse: bool) -> float:
        """Fit the ITA->Fitzpatrick map on CLEAN images, apply to perturbed ones.

        Refitting per condition would measure whether ITA still *ranks* tone, which it
        does. The question a practitioner has is different: they calibrate a mapping once
        and then apply it to photographs of unknown exposure, so the map must be frozen.
        """
        target = np.array([FITZ_COARSE[v] for v in y]) if coarse else y
        model = GradientBoostingClassifier(random_state=0)
        if ita is clean:
            # The clean row has to be cross-validated. Fitting and scoring the map on the
            # same images makes it in-sample and inflates the baseline, which inflates the
            # drop -- in the direction that flatters the finding. The perturbed rows are
            # already out-of-sample by construction, since the map never saw them.
            pred = cross_val_predict(model, clean.reshape(-1, 1), target, cv=5)
            return float((pred == target).mean())
        model.fit(clean.reshape(-1, 1), target)
        return float((model.predict(ita.reshape(-1, 1)) == target).mean())

    print(f"{len(items)} generated images with a requested Fitzpatrick type")
    print("Mapping ITA -> Fitzpatrick is fitted once on unperturbed images, then frozen.\n")
    print(f"{'condition':>18} {'6-type acc':>11} {'3-group acc':>12} {'median ITA':>11}")
    base6 = accuracy(clean, False)
    base3 = accuracy(clean, True)
    print(f"{'clean':>18} {base6:>11.3f} {base3:>12.3f} {np.median(clean):>11.1f}")
    for ev in args.ev:
        ita = ita_under(ev)
        a6, a3 = accuracy(ita, False), accuracy(ita, True)
        print(f"{f'{ev:+.1f} EV':>18} {a6:>11.3f} {a3:>12.3f} {np.median(ita):>11.1f}")

    chance6 = float(np.bincount(y).max() / len(y))
    coarse_y = np.array([FITZ_COARSE[v] for v in y])
    chance3 = float(max(np.sum(coarse_y == g) for g in set(coarse_y)) / len(y))
    print(f"\nmajority-class floor: {chance6:.3f} (6 types), {chance3:.3f} (3 groups)")


if __name__ == "__main__":
    main()
