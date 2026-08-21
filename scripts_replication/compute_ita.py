"""Stage 1: compute ITA for every SCIN image under exposure perturbations and
under gray-world + luminance normalisation. Writes data/scin_ita.csv.

Reuses the repo's ITA maths (fitymi.controls.skintone._srgb_to_lab and the same
median-L / median-b / a*-decile-drop pipeline as estimate_ita) but operates on
arrays so perturbed variants can be evaluated without round-tripping to disk.
"""
from __future__ import annotations

import glob
import os
import sys

import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, "/Users/gs/fake_it_til_you_make_it/src")
from fitymi.controls.skintone import ITA_BINS, _srgb_to_lab, estimate_ita  # noqa: E402

MAX_SIDE = 256
EVS = (-0.5, -0.3, 0.0, 0.3, 0.5)


def srgb_to_linear(x: np.ndarray) -> np.ndarray:
    return np.where(x > 0.04045, ((x + 0.055) / 1.055) ** 2.4, x / 12.92)


def linear_to_srgb(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0.0, 1.0)
    return np.where(x > 0.0031308, 1.055 * x ** (1 / 2.4) - 0.055, 12.92 * x)


def expose(rgb: np.ndarray, ev: float) -> np.ndarray:
    """Exposure change in LINEAR light: linearise, scale by 2**EV, re-encode."""
    if ev == 0.0:
        return rgb
    return linear_to_srgb(srgb_to_linear(rgb) * (2.0**ev))


def normalise(rgb: np.ndarray) -> np.ndarray:
    """Gray-world white balance + 90th-pct luminance pinned to 0.7, in linear light."""
    lin = srgb_to_linear(rgb)
    means = lin.reshape(-1, 3).mean(axis=0)
    means = np.maximum(means, 1e-6)
    lin = lin * (means.mean() / means)
    lum = lin @ np.array([0.2126729, 0.7151522, 0.0721750])
    p90 = float(np.percentile(lum, 90))
    if p90 > 1e-6:
        lin = lin * (0.7 / p90)
    return linear_to_srgb(lin)


def ita_from_array(rgb: np.ndarray) -> tuple[float, float]:
    """Same estimator as estimate_ita, on an already-loaded [0,1] RGB array."""
    h, w, _ = rgb.shape
    centre = rgb[h // 6 : h - h // 6, w // 6 : w - w // 6]
    lab = _srgb_to_lab(centre.reshape(-1, 3).astype(np.float64))
    L, a = lab[:, 0], lab[:, 1]
    lit = (L > 15) & (L < 95)
    if lit.sum() < 32:
        lit = np.ones_like(L, dtype=bool)
    a_cut = np.percentile(a[lit], 90)
    keep = lit & (a <= a_cut)
    if keep.sum() < 32:
        keep = lit
    lab_keep = lab[keep]
    L_m = float(np.median(lab_keep[:, 0]))
    b_m = float(np.median(lab_keep[:, 2]))
    ita = float(np.degrees(np.arctan2(L_m - 50.0, max(b_m, 1e-6))))
    return ita, float(keep.mean())


def tone_bin(ita: float) -> str:
    for label, lo, hi in ITA_BINS:
        if lo <= ita < hi:
            return label
    return "dark"


def main() -> None:
    root = "/Users/gs/fake_it_til_you_make_it"
    meta = pd.read_csv(f"{root}/data/scin_sample.csv")
    have = {os.path.basename(p) for p in glob.glob(f"{root}/data/scin_images/*.png")}
    rows = []
    for i, r in meta.iterrows():
        fn = os.path.basename(str(r.image_1_path))
        if fn not in have:
            continue
        path = f"{root}/data/scin_images/{fn}"
        with Image.open(path) as im:
            im = im.convert("RGB")
            full_w, full_h = im.size
            im.thumbnail((MAX_SIDE, MAX_SIDE))
            arr = np.asarray(im, dtype=np.float64) / 255.0
        rec = {
            "case_id": r.case_id,
            "file": fn,
            "fst_self": r.fitzpatrick_skin_type,
            "fst_derm": r.dermatologist_fitzpatrick_skin_type_label_1,
            "monk_us": r.monk_skin_tone_label_us,
            "monk_india": r.monk_skin_tone_label_india,
            "shot_type": r.image_1_shot_type,
            "width": full_w,
            "height": full_h,
            "megapixels": full_w * full_h / 1e6,
            "aspect": full_w / full_h,
        }
        norm_base = normalise(arr)
        for ev in EVS:
            tag = f"{ev:+.1f}".replace("+", "p").replace("-", "m").replace(".", "")
            raw_ita, cov = ita_from_array(expose(arr, ev))
            rec[f"ita_raw_{tag}"] = raw_ita
            rec[f"cov_raw_{tag}"] = cov
            # normalisation is applied AFTER the perturbation, as a camera-side fix
            # would be: perturb the capture, then normalise what you received.
            nrm_ita, ncov = ita_from_array(normalise(expose(arr, ev)))
            rec[f"ita_norm_{tag}"] = nrm_ita
            rec[f"cov_norm_{tag}"] = ncov
        rec["ita_norm_base_check"] = ita_from_array(norm_base)[0]
        rows.append(rec)
        if len(rows) % 100 == 0:
            print(f"{len(rows)} images", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(f"{root}/data/scin_ita.csv", index=False)
    # sanity: our array-path ITA must match the repo's file-path estimate_ita
    chk = df.sample(min(20, len(df)), random_state=0)
    diffs = []
    for _, r in chk.iterrows():
        e = estimate_ita(f"{root}/data/scin_images/{r.file}")
        diffs.append(abs(e.ita - r["ita_raw_p00"]))
    print(f"n={len(df)}  max|array-vs-repo ITA| = {max(diffs):.4f} deg")


if __name__ == "__main__":
    main()
