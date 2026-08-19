"""Protocol §8.6 — skin-tone stratification via individual typology angle.

ITA is computed from CIE L*a*b* as atan2(L* - 50, b*) in degrees, and binned into the
conventional groups. It is a *proxy*: it measures the pixels, not the person, and it
is confounded by lighting, white balance and the erythema that acne itself produces.
We use it because ACNE04 carries no skin-tone annotation, and we report it as a proxy
everywhere it appears.

This is also the analysis where synthetic data has its strongest prior case: an
open-set generator can be prompted across Fitzpatrick I-VI, whereas the real corpus
cannot be re-collected.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

#: Conventional ITA boundaries, from lightest to darkest.
ITA_BINS = (
    ("very_light", 55.0, 90.0),
    ("light", 41.0, 55.0),
    ("intermediate", 28.0, 41.0),
    ("tan", 10.0, 28.0),
    ("brown", -30.0, 10.0),
    ("dark", -90.0, -30.0),
)

#: Coarse grouping used for reporting; six bins are too thin at n~290.
COARSE = {
    "very_light": "light",
    "light": "light",
    "intermediate": "intermediate",
    "tan": "intermediate",
    "brown": "dark",
    "dark": "dark",
}


def _srgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """rgb in [0,1], shape (..., 3) -> CIE L*a*b* under D65."""
    mask = rgb > 0.04045
    lin = np.where(mask, ((rgb + 0.055) / 1.055) ** 2.4, rgb / 12.92)
    m = np.array(
        [
            [0.4124564, 0.3575761, 0.1804375],
            [0.2126729, 0.7151522, 0.0721750],
            [0.0193339, 0.1191920, 0.9503041],
        ]
    )
    xyz = lin @ m.T
    white = np.array([0.95047, 1.0, 1.08883])
    xyz = xyz / white
    eps, kappa = 216 / 24389, 24389 / 27
    f = np.where(xyz > eps, np.cbrt(xyz), (kappa * xyz + 16) / 116)
    L = 116 * f[..., 1] - 16
    a = 500 * (f[..., 0] - f[..., 1])
    b = 200 * (f[..., 1] - f[..., 2])
    return np.stack([L, a, b], axis=-1)


@dataclass
class SkinToneEstimate:
    ita: float
    bin: str
    coarse_bin: str
    coverage: float


def estimate_ita(path: str, max_side: int = 256) -> SkinToneEstimate:
    """Estimate ITA from the central skin region, excluding erythematous pixels.

    Lesions are red and would drag ITA downward, so a* is used to drop the most
    erythematous decile before averaging. `coverage` reports what fraction of pixels
    survived, so an estimate from a badly-lit or heavily-inflamed image can be
    discounted.
    """
    from PIL import Image

    with Image.open(path) as im:
        im = im.convert("RGB")
        im.thumbnail((max_side, max_side))
        arr = np.asarray(im, dtype=np.float32) / 255.0

    h, w, _ = arr.shape
    centre = arr[h // 6 : h - h // 6, w // 6 : w - w // 6]
    lab = _srgb_to_lab(centre.reshape(-1, 3))

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

    name = "dark"
    for label, lo, hi in ITA_BINS:
        if lo <= ita < hi:
            name = label
            break
    return SkinToneEstimate(
        ita=ita, bin=name, coarse_bin=COARSE[name], coverage=float(keep.mean())
    )


def stratify(paths, labels, preds) -> dict[str, dict]:
    """Group predictions by coarse ITA bin and return per-bin index lists."""
    out: dict[str, dict] = {}
    for i, p in enumerate(paths):
        est = estimate_ita(p)
        bucket = out.setdefault(est.coarse_bin, {"idx": [], "ita": []})
        bucket["idx"].append(i)
        bucket["ita"].append(est.ita)
    labels = np.asarray(labels)
    preds = np.asarray(preds)
    for name, bucket in out.items():
        idx = np.asarray(bucket["idx"])
        bucket["n"] = int(len(idx))
        bucket["mean_ita"] = float(np.mean(bucket["ita"]))
        bucket["y_true"] = labels[idx].tolist()
        bucket["y_pred"] = preds[idx].tolist()
        del bucket["ita"]
    return out
