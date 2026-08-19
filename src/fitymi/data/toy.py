"""A procedural stand-in for ACNE04, used to verify the pipeline end to end.

This is not a model of skin and makes no claim to be. It exists because the real
experiment needs a GPU and a dataset behind an academic-use licence, and we still
want the splitting, mixing, training, evaluation and statistics code to be exercised
by an actual run rather than by assertion.

The important part is that the *synthetic* toy images are drawn from a deliberately
mis-specified version of the real generative process, with the mismatch controlled by
a single `gap` parameter. It reproduces, in miniature, the failure modes the
literature attributes to diffusion-generated training data:

* **concept error** — the generator cannot render the label it was asked for, so the
  lesion count drifts toward the corpus mean and the severity grades blur into each
  other. This is the primary mechanism, and the one Fan et al. (CVPR 2024) blame for
  the supervised-classifier gap: an image labelled "very severe" that depicts a
  moderate case is label noise, not merely a domain shift.
* **mode collapse** — lesion-size and skin-tone diversity narrow;
* **generator fingerprint** — a faint periodic texture no real image carries.

A first version of this simulator implemented only the last two, and synthetic-trained
models came out *better* than real-trained ones: narrowing the within-class
distributions without breaking the class-to-image mapping just produces cleaner
prototypes of each grade. Useful lesson, and the reason concept error is modelled
explicitly here — a gap that does not corrupt the label-image relationship is not the
gap this study is about.

With `gap=0` the two processes are identical and every mixing arm should perform the
same; that is the null the smoke test checks against. With `gap>0` the mixing curve
should slope down. If our analysis code cannot recover those two behaviours here, it
cannot be trusted on real data either.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from ..utils.seeding import rng
from .records import NUM_CLASSES, Corpus, Record, Source

#: Lesion-count boundaries defining the four ordinal severity grades. Chosen to be
#: separable but overlapping, so the task is neither trivial nor hopeless.
SEVERITY_BINS = ((1, 5), (6, 14), (15, 28), (29, 50))

IMAGE_SIZE = 64


@dataclass
class ToyConfig:
    size: int = IMAGE_SIZE
    #: 0 = synthetic process identical to real; 1 = maximally mis-specified.
    gap: float = 0.6
    fingerprint_strength: float = 0.04
    noise: float = 0.03


def _skin_tone(gen: np.random.Generator, gap: float) -> np.ndarray:
    """Real skin tones span a wide range; the mis-specified process narrows it."""
    base = np.array([222.0, 178.0, 150.0])
    spread = 55.0 * (1.0 - 0.8 * gap)
    centre_shift = np.array([6.0, -3.0, -6.0]) * gap
    tone = base + centre_shift + gen.normal(0, spread * 0.35, size=3)
    return np.clip(tone, 60, 250)


#: Prior over severity grades. Roughly the shape of a real acne corpus: a thin tail.
CLASS_PRIOR = (0.40, 0.33, 0.19, 0.08)

#: Corpus-mean lesion count, used as the attractor for concept error.
_GLOBAL_MEAN = sum(
    p * (lo + hi) / 2.0 for p, (lo, hi) in zip(CLASS_PRIOR, SEVERITY_BINS)
)


def _lesion_count(gen: np.random.Generator, label: int, gap: float) -> int:
    """Lesion count for one image. Count is what defines the severity grade.

    For real images the count is drawn uniformly within the grade's bin, so the label
    always matches the image. For synthetic images the class-conditional mean is
    pulled toward the corpus mean in proportion to `gap`, and the result is *not*
    clipped back into the bin: at gap=1 every grade renders the same picture and the
    label carries no information about the image. That is concept error, and it is
    the thing that makes synthetic training data harmful rather than merely different.
    """
    lo, hi = SEVERITY_BINS[label]
    if gap <= 0:
        return int(gen.integers(lo, hi + 1))

    mid = (lo + hi) / 2.0
    width = (hi - lo) / 2.0
    # Concept error: the rendered severity drifts toward the corpus mean.
    mid = (1.0 - gap) * mid + gap * _GLOBAL_MEAN
    # Mode collapse: less within-grade variety on top of that.
    spread = max(width * (1.0 - 0.5 * gap), 0.75)
    n = gen.normal(mid, spread)
    # Clipped only to the physically possible range, never back into the label's bin.
    return int(np.clip(round(n), 0, SEVERITY_BINS[-1][1] + 5))


def render(
    label: int,
    seed: int,
    config: ToyConfig,
    synthetic: bool,
) -> Image.Image:
    gen = rng(seed)
    gap = config.gap if synthetic else 0.0
    s = config.size

    yy, xx = np.mgrid[0:s, 0:s].astype(np.float32)

    img = np.zeros((s, s, 3), dtype=np.float32)
    img[:] = np.array([28.0, 30.0, 36.0])

    # Face: an ellipse with a soft edge and a lighting gradient.
    cx, cy = s / 2 + gen.normal(0, s * 0.02), s / 2 + gen.normal(0, s * 0.02)
    rx, ry = s * 0.40, s * 0.46
    d = ((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2
    face = np.clip(1.6 - d * 1.6, 0.0, 1.0)[..., None]

    tone = _skin_tone(gen, gap)
    light_dir = gen.uniform(0, 2 * np.pi)
    light_amp = 0.18 * (1.0 - 0.6 * gap)
    light = 1.0 + light_amp * (
        np.cos(light_dir) * (xx - cx) / s + np.sin(light_dir) * (yy - cy) / s
    )
    img = img * (1 - face) + face * tone[None, None, :] * light[..., None]

    # Lesions: reddish blobs. Count drives the label; size/saturation are where the
    # mis-specified process gives itself away.
    n_lesions = _lesion_count(gen, label, gap)
    size_scale = 1.0 - 0.25 * gap
    size_spread = 1.0 - 0.7 * gap
    redness = np.array([48.0, -26.0, -22.0]) * (1.0 + 0.35 * gap)

    for _ in range(n_lesions):
        while True:
            lx = gen.uniform(cx - rx * 0.85, cx + rx * 0.85)
            ly = gen.uniform(cy - ry * 0.85, cy + ry * 0.85)
            if ((lx - cx) / rx) ** 2 + ((ly - cy) / ry) ** 2 < 0.75:
                break
        radius = max(
            0.6,
            (s * 0.022) * size_scale * np.exp(gen.normal(0, 0.45 * size_spread)),
        )
        blob = np.exp(-(((xx - lx) ** 2 + (yy - ly) ** 2) / (2 * radius**2)))
        img += blob[..., None] * redness[None, None, :] * face

    if synthetic and config.fingerprint_strength > 0:
        # A faint periodic texture: the toy analogue of an upsampling artefact.
        fp = np.sin(xx * 1.7) * np.sin(yy * 1.7)
        img += (config.fingerprint_strength * gap * 255.0) * fp[..., None]

    img += gen.normal(0, config.noise * 255.0, size=img.shape)
    return Image.fromarray(np.clip(img, 0, 255).astype(np.uint8))


def _class_sizes(n: int, imbalance: tuple[float, ...] = CLASS_PRIOR) -> list[int]:
    """Roughly the shape of a real acne severity corpus: the severe tail is thin."""
    from .mixing import largest_remainder

    return largest_remainder(n, np.array(imbalance)).tolist()


def make_toy_corpus(
    root: str | Path,
    n: int,
    source: Source = Source.REAL,
    config: ToyConfig | None = None,
    seed: int = 0,
    prefix: str = "img",
) -> Corpus:
    """Render `n` images to `root` and return the corpus describing them."""
    config = config or ToyConfig()
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    synthetic = source.is_synthetic

    records: list[Record] = []
    counter = 0
    for label, count in enumerate(_class_sizes(n)):
        for _ in range(count):
            img_seed = seed * 1_000_003 + counter
            img = render(label, img_seed, config, synthetic)
            path = root / f"{prefix}_{label}_{counter:06d}.png"
            img.save(path)
            records.append(
                Record(
                    path=str(path),
                    label=label,
                    source=source,
                    meta={"toy_seed": img_seed, "gap": config.gap if synthetic else 0.0},
                )
            )
            counter += 1
    return Corpus(records)


def make_toy_study(
    root: str | Path,
    n_real: int = 600,
    n_synth: int = 1200,
    gap: float = 0.6,
    seed: int = 0,
) -> dict[str, Corpus]:
    """Real corpus plus closed-set and open-set synthetic pools.

    The open-set pool carries a larger gap, mirroring the expectation that a
    generator which never saw our data is further from its distribution than one
    fine-tuned on it.
    """
    root = Path(root)
    real = make_toy_corpus(
        root / "real", n_real, Source.REAL, ToyConfig(gap=0.0), seed=seed, prefix="real"
    )
    closed = make_toy_corpus(
        root / "synth_closed",
        n_synth,
        Source.SYNTH_CLOSED,
        ToyConfig(gap=gap),
        seed=seed + 1,
        prefix="closed",
    )
    open_ = make_toy_corpus(
        root / "synth_open",
        n_synth,
        Source.SYNTH_OPEN,
        ToyConfig(gap=min(1.0, gap * 1.5)),
        seed=seed + 2,
        prefix="open",
    )
    return {"real": real, "synth_closed": closed, "synth_open": open_}


__all__ = [
    "ToyConfig",
    "SEVERITY_BINS",
    "CLASS_PRIOR",
    "NUM_CLASSES",
    "render",
    "make_toy_corpus",
    "make_toy_study",
]
