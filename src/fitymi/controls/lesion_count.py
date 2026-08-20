"""A crude, objective inflammatory-lesion counter, validated against ACNE04's own counts.

Whether a generated image carries the severity it was asked for is the question that
decides the coverage arm, and eyeballing a contact sheet is not a measurement. This is a
deliberately simple detector -- elevated redness, connected components, size-filtered --
whose only virtue is that it is objective, repeatable, and **checkable against ground
truth**: ACNE04 publishes a lesion count for every image, so the counter's agreement with
a dermatologist can be measured before it is pointed at anything synthetic.

It is not a lesion detector in any clinical sense and must never be described as one. It
is an instrument with a known and reportable error, used to answer one narrow question:
does a requested severity produce a correspondingly different amount of visible
inflammatory signal? If it cannot reproduce ACNE04's counts to a useful correlation, it is
not fit even for that, and the honest move is to say so rather than to use it anyway.

## Result of that validation: it is not fit for primary use

Measured on 200 real ACNE04 images against their published counts:

- **Untuned: Spearman 0.103**, and a median count of 155 against a true median of 20 --
  a 7.8x overcount with no monotone trend across the four Hayashi bands (139 / 155 / 169 /
  133 from mild to very severe). It was measuring skin texture, not inflammation.
- **After sweeping 24 settings: best Spearman 0.365**, and that number is optimistic
  because the sweep was scored on the same images it was tuned on.

A usable ordinal instrument needs roughly 0.5. Part of the ceiling is not the counter's
fault -- ACNE04's own repeat annotations of *identical* images agree on the exact count
only 31.6% of the time, so the ground truth is itself noisy -- but 0.365 is well short
regardless.

**So this is not used to calibrate the coverage arm.** It is kept because a negative result
about an instrument is worth as much as a positive one and costs a future reader the same
afternoon it cost here, and because it may still serve as a weak secondary signal *with its
0.365 correlation stated beside every number it produces*. The primary calibration
instrument is the real-trained classifier's agreement with the requested grade, which is
what protocol section 4.5 already committed to.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Sequence

import numpy as np

log = logging.getLogger(__name__)


@dataclass
class CountConfig:
    #: Working resolution. Lesions are small; downscaling too far erases them.
    size: int = 512
    #: How far above the local skin baseline a pixel's redness must sit, in units of the
    #: image's own redness standard deviation. Higher is stricter.
    redness_sigma: float = 1.6
    #: Blob area bounds as a fraction of image area. The lower bound rejects sensor noise,
    #: the upper rejects lips, nostrils and shadow.
    min_area_frac: float = 2e-5
    max_area_frac: float = 4e-3
    #: Background smoothing radius, as a fraction of the image. The baseline has to follow
    #: illumination and skin tone, or dark skin reads as uniformly "not red".
    background_frac: float = 0.08


def _to_lab_a(image: np.ndarray) -> np.ndarray:
    """The a* channel of CIELAB: green-to-red. Chosen because it separates inflammation
    from pigmentation far better than the red channel of RGB, which mostly tracks how
    light the skin is."""
    rgb = image.astype(np.float32) / 255.0
    mask = rgb > 0.04045
    linear = np.where(mask, ((rgb + 0.055) / 1.055) ** 2.4, rgb / 12.92)
    m = np.array([[0.4124, 0.3576, 0.1805],
                  [0.2126, 0.7152, 0.0722],
                  [0.0193, 0.1192, 0.9505]], dtype=np.float32)
    xyz = linear @ m.T
    white = np.array([0.95047, 1.0, 1.08883], dtype=np.float32)
    t = xyz / white
    f = np.where(t > 0.008856, np.cbrt(t), 7.787 * t + 16 / 116)
    return 500.0 * (f[..., 0] - f[..., 1])


def count_lesions(paths: Sequence[str], config: CountConfig | None = None) -> np.ndarray:
    """Count candidate inflammatory lesions in each image. Returns one integer per path."""
    from PIL import Image
    from scipy import ndimage

    config = config or CountConfig()
    out = np.zeros(len(paths), dtype=int)

    for i, path in enumerate(paths):
        with Image.open(path) as handle:
            image = np.asarray(
                handle.convert("RGB").resize((config.size, config.size), Image.LANCZOS)
            )
        a = _to_lab_a(image)

        # Subtract a smoothed version of itself, so what remains is redness *relative to
        # the surrounding skin* rather than absolute redness. Without this, a fair face
        # scores high everywhere and a dark one scores low everywhere, and the counter
        # measures skin tone instead of inflammation.
        sigma = config.background_frac * config.size
        baseline = ndimage.gaussian_filter(a, sigma)
        residual = a - baseline

        threshold = config.redness_sigma * float(residual.std())
        blobs, n = ndimage.label(residual > threshold)
        if n == 0:
            continue
        areas = np.bincount(blobs.ravel())[1:]
        pixels = config.size * config.size
        keep = (areas >= config.min_area_frac * pixels) & (areas <= config.max_area_frac * pixels)
        out[i] = int(keep.sum())

    return out


def validate_against(paths: Sequence[str], truth: Sequence[int],
                     config: CountConfig | None = None) -> dict:
    """Measure the counter against known counts before trusting it on anything else.

    Reports Spearman rank correlation as the headline, because the question the counter is
    used for is ordinal -- does more requested severity produce more signal -- and rank
    correlation does not require the counter to be unbiased, only monotone.
    """
    predicted = count_lesions(paths, config)
    truth = np.asarray(truth, dtype=float)
    if len(truth) < 3:
        raise ValueError("need at least three images to validate a counter")

    def rank(x: np.ndarray) -> np.ndarray:
        order = np.argsort(np.argsort(x))
        return order.astype(float)

    spearman = float(np.corrcoef(rank(predicted.astype(float)), rank(truth))[0, 1])
    pearson = float(np.corrcoef(predicted.astype(float), truth)[0, 1])
    return {
        "n": len(truth),
        "spearman": spearman,
        "pearson": pearson,
        "predicted_median": float(np.median(predicted)),
        "truth_median": float(np.median(truth)),
        "predicted_over_truth": float(np.median(predicted) / max(np.median(truth), 1e-9)),
    }
