"""Prompt construction for both generator arms.

Prompt diversity is not a cosmetic choice. He et al. (ICLR 2023) had to add a
sentence-expansion model because naive class-name prompting produces images that are
individually fine and collectively near-identical, and the fidelity-diversity
trade-off is the mechanism most often blamed for the synthetic-to-real gap. So the
open-set arm samples over demographic, capture and lighting axes rather than
repeating one template, and we report the two prompt regimes as an ablation.

The closed-set arm does the opposite: a single rare token per severity grade, with
the visual content carried by the fine-tuned weights rather than by the text.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Iterator, Sequence

from ..data.records import NUM_CLASSES, SEVERITY_NAMES
from ..utils.seeding import rng

#: Clinical descriptions of the four Hayashi grades, phrased for a text encoder.
SEVERITY_DESCRIPTIONS = {
    0: "a few scattered comedones and one or two small papules, mild acne vulgaris",
    1: "numerous papules and pustules across the cheek, moderate acne vulgaris",
    2: "dense inflammatory papules and pustules with several nodules, severe acne vulgaris",
    3: "confluent inflammatory nodules and cysts with scarring, very severe acne vulgaris",
}

#: Rare tokens for the closed-set arm, one per grade.
SEVERITY_TOKENS = {0: "sks-acne-a", 1: "sks-acne-b", 2: "sks-acne-c", 3: "sks-acne-d"}

AGES = ("a teenage", "a young adult", "an adult")
SEXES = ("male", "female")
#: Fitzpatrick I-VI. The open-set arm can sample tones the real corpus does not
#: contain, which is the one place synthetic data has a clear a-priori advantage.
SKIN_TONES = (
    "very fair skin", "fair skin", "light olive skin",
    "olive skin", "brown skin", "deeply pigmented skin",
)
VIEWS = (
    "photographed at a three-quarter angle from the front",
    "photographed in profile",
    "photographed frontally",
)
LIGHTING = (
    "even clinical lighting", "diffuse daylight", "overhead fluorescent lighting",
    "soft window light",
)
#: Every template says "colour" explicitly. A CPU probe of the untouched prompts
#: (docs/examples/guidance_probe_cpu.png) returned mostly black-and-white, vintage-looking
#: portraits: Stable Diffusion's prior for "documentation photograph" pulls towards
#: archival imagery, and a monochrome training pool would be a confound with nothing to do
#: with acne.
CAPTURE = (
    "colour clinical dermatology photograph",
    "colour close-up medical photograph",
    "high-resolution colour dermatological documentation photograph",
)

#: Extended after the CPU probe. The monochrome and archival terms address Stable
#: Diffusion's pull towards vintage documentation imagery; the collage and border terms
#: address multi-panel outputs with caption strips, which appeared at low guidance where
#: the negative prompt also carries less weight. Freckles and moles are named because
#: they are what the model reaches for when asked for facial lesions, and a pool of
#: freckled faces labelled "moderate acne" is worse than no pool at all.
NEGATIVE_PROMPT = (
    "illustration, drawing, cartoon, 3d render, cgi, painting, text, watermark, "
    "logo, caption, collage, multiple panels, split screen, border, frame, "
    "black and white, monochrome, greyscale, sepia, vintage, film grain, "
    "freckles, moles, birthmark, "
    "blurry, distorted anatomy, deformed face, deformed teeth, extra teeth, "
    "oversaturated, beauty filter"
)


@dataclass(frozen=True)
class Prompt:
    text: str
    label: int
    negative: str = NEGATIVE_PROMPT
    meta: dict | None = None


def closed_set_prompt(label: int) -> Prompt:
    """Minimal text; the fine-tuned weights carry the content."""
    token = SEVERITY_TOKENS[label]
    return Prompt(
        text=f"a clinical photograph of a face with {token} acne vulgaris",
        label=label,
        meta={"regime": "closed_set", "token": token},
    )


def open_set_prompt(label: int, seed: int) -> Prompt:
    """One sample from the diversified open-set prompt distribution."""
    gen = rng(seed)
    age = AGES[gen.integers(len(AGES))]
    sex = SEXES[gen.integers(len(SEXES))]
    tone = SKIN_TONES[gen.integers(len(SKIN_TONES))]
    view = VIEWS[gen.integers(len(VIEWS))]
    light = LIGHTING[gen.integers(len(LIGHTING))]
    capture = CAPTURE[gen.integers(len(CAPTURE))]
    description = SEVERITY_DESCRIPTIONS[label]
    text = (
        f"{capture} of the face of {age} {sex} person with {tone}, "
        f"showing {description}, {view}, {light}"
    )
    return Prompt(
        text=text,
        label=label,
        meta={
            "regime": "open_set", "age": age, "sex": sex, "skin_tone": tone,
            "view": view, "lighting": light, "capture": capture,
        },
    )


def minimal_open_set_prompt(label: int) -> Prompt:
    """The ablation baseline: class name only, no diversification."""
    return Prompt(
        text=f"a photograph of a face with {SEVERITY_NAMES[label].replace('_', ' ')} acne",
        label=label,
        meta={"regime": "open_set_minimal"},
    )


def prompt_stream(
    labels: Sequence[int],
    counts: Sequence[int],
    regime: str = "open_set",
    seed: int = 0,
) -> Iterator[Prompt]:
    """Yield `counts[i]` prompts for `labels[i]`, in class order."""
    builders = {
        "closed_set": lambda label, s: closed_set_prompt(label),
        "open_set": open_set_prompt,
        "open_set_minimal": lambda label, s: minimal_open_set_prompt(label),
    }
    if regime not in builders:
        raise ValueError(f"unknown prompt regime {regime!r}; expected one of {sorted(builders)}")
    build = builders[regime]
    counter = 0
    for label, count in zip(labels, counts):
        for _ in range(count):
            yield build(label, seed * 1_000_003 + counter)
            counter += 1


def prompt_grid_size() -> int:
    """How many distinct open-set prompts the axes can produce."""
    return len(list(product(AGES, SEXES, SKIN_TONES, VIEWS, LIGHTING, CAPTURE))) * NUM_CLASSES
