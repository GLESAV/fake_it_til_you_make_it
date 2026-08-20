#!/usr/bin/env python
"""Can exposure be normalised out of ITA after the fact?

The confound result says ITA moves with exposure. The obvious response is to normalise the
image first, and a reviewer is entitled to ask whether that fixes it. This tries the three
corrections available to someone holding only the photograph, and measures each the same
way: fit the ITA->Fitzpatrick map on clean images, freeze it, apply to perturbed ones.

The prior is that none of them can work, and the reason is worth stating before the
numbers. Exposure and skin lightness both act on L*, and a single uncalibrated photograph
contains nothing that distinguishes "a lighter-skinned person" from "the same person, one
stop brighter". Any normalisation that removes the second necessarily removes the first,
because they are the same pixels. Separating them needs information the image does not
carry: a grey card in frame, or the camera's exposure metadata.

If that is right, the constructive conclusion is not a better normalisation but a change in
what ITA is used for -- group medians rather than per-image bins -- plus a calibration
target for anyone who needs per-image tone.
"""
from __future__ import annotations

import argparse
import json
import re
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_predict

from fitymi.controls.skintone import estimate_ita

FITZ = ["I", "II", "III", "IV", "V", "VI"]
COARSE3 = {0: "light", 1: "light", 2: "mid", 3: "mid", 4: "dark", 5: "dark"}


def to_linear(a):
    a = a / 255.0
    return np.where(a > 0.04045, ((a + 0.055) / 1.055) ** 2.4, a / 12.92)


def to_srgb(lin):
    lin = np.clip(lin, 0, 1)
    s = np.where(lin > 0.0031308, 1.055 * lin ** (1 / 2.4) - 0.055, 12.92 * lin)
    return np.clip(s * 255, 0, 255).astype(np.uint8)


def norm_none(lin):
    return lin


def norm_grayworld(lin):
    """Scale each channel to a common mean. Standard illuminant correction."""
    m = lin.reshape(-1, 3).mean(0)
    return lin * (m.mean() / np.maximum(m, 1e-6))


def norm_luminance(lin):
    """Fix the 90th percentile of luminance. Removes exposure -- and lightness with it."""
    y = lin @ np.array([0.2126, 0.7152, 0.0722])
    p = np.percentile(y, 90)
    return lin * (0.7 / max(p, 1e-6))


def norm_both(lin):
    return norm_luminance(norm_grayworld(lin))


NORMS = {"none": norm_none, "gray-world": norm_grayworld,
         "luminance p90": norm_luminance, "both": norm_both}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default="data/synthetic/gemini_pool")
    ap.add_argument("--ev", type=float, nargs="+", default=[-0.3, 0.3])
    args = ap.parse_args()

    items = []
    for line in (Path(args.pool) / "manifest.jsonl").read_text().splitlines():
        r = json.loads(line)
        if not r.get("path") or not Path(r["path"]).exists():
            continue
        m = re.search(r"Fitzpatrick (I{1,3}V?|IV|VI?)\b", r["prompt"])
        if m:
            items.append((FITZ.index(m.group(1)), r["path"]))
    y = np.array([f for f, _ in items])
    y3 = np.array([COARSE3[v] for v in y])
    tmp = Path(tempfile.mkdtemp())

    cache: dict[str, np.ndarray] = {}
    def ita(norm_name: str, ev: float) -> np.ndarray:
        key = f"{norm_name}|{ev}"
        if key in cache:
            return cache[key]
        fn = NORMS[norm_name]
        out = []
        for _, path in items:
            with Image.open(path) as im:
                arr = np.asarray(im.convert("RGB").resize((256, 256), Image.LANCZOS),
                                 dtype=np.float32)
            lin = to_linear(arr) * (2 ** ev)
            f = tmp / "x.png"
            Image.fromarray(to_srgb(fn(lin))).save(f)
            out.append(estimate_ita(str(f)).ita)
        cache[key] = np.array(out)
        return cache[key]

    print(f"{len(items)} generated images with a known requested Fitzpatrick type")
    print("Map fitted on clean images under each normalisation, then frozen.\n")
    # Both columns must be out-of-sample or they are not comparable. Scoring the clean
    # column by cross-validation while scoring the perturbed columns with a map fitted on
    # ALL the clean data lets the perturbed columns beat the clean one -- the map has
    # already seen those very images, only brighter. The first run of this script did
    # exactly that and reported perturbed accuracy of 0.826 against clean 0.462, a
    # "negative drop", which is not a finding about normalisation but about the protocol.
    # Here every fold fits on the clean images of the training folds and predicts the
    # perturbed images of the held-out fold, so the map never sees the image it scores.
    from sklearn.model_selection import StratifiedKFold

    def folded(name: str, ev: float) -> float:
        clean = ita(name, 0.0).reshape(-1, 1)
        shifted = ita(name, ev).reshape(-1, 1)
        pred = np.empty(len(y3), dtype=object)
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(clean, y3):
            m = GradientBoostingClassifier(random_state=0).fit(clean[tr], y3[tr])
            pred[te] = m.predict(shifted[te])
        return float((pred == y3).mean())

    print(f"{'normalisation':>16} {'clean':>8} " +
          " ".join(f"{f'{e:+.1f}EV':>8}" for e in args.ev) + f"{'  worst drop':>13}")
    for name in NORMS:
        base = folded(name, 0.0)
        scores = [folded(name, e) for e in args.ev]
        print(f"{name:>16} {base:>8.3f} " + " ".join(f"{s:>8.3f}" for s in scores) +
              f"{base - min(scores):>13.3f}")
    floor = float(max(np.sum(y3 == g) for g in set(y3)) / len(y3))
    print(f"\nmajority-class floor: {floor:.3f}  (three groups: light / mid / dark)")


if __name__ == "__main__":
    main()
