"""Construction of the training mixtures. This is the experiment.

Three families of mixture, matching protocol §5.1, §5.2 and §8.4:

* `substitution`  — fixed total budget N, synthetic fraction p replaces real.
* `additive`      — real budget n held fixed, synthetic added at k x n on top.
* `real_subset`   — real-only at size (1-p)N, the control that says whether a
                    synthetic arm beats "just having fewer real images".

Two properties matter more than they look:

**Nesting.** Real subsets are prefixes of a single seeded per-class permutation, so
the real half of the p=75 arm is a subset of the p=50 arm's. Arms then differ by
*which images were removed*, not by an independent redraw, which strips a large
variance term out of the arm-to-arm comparison.

**Class histogram preservation.** Every arm carries the same class distribution as
the real training split. Otherwise a synthetic arm could win or lose on prior shift
alone and we would learn nothing about image content.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from ..utils.seeding import rng
from .records import NUM_CLASSES, Corpus, Record

MixKind = Literal["substitution", "additive", "real_subset"]


class InsufficientSyntheticError(RuntimeError):
    """Raised when the synthetic pool cannot fill a requested arm."""


@dataclass(frozen=True)
class MixSpec:
    kind: MixKind = "substitution"
    #: Synthetic fraction in [0, 1]. Used by `substitution` and `real_subset`.
    synthetic_fraction: float = 0.0
    #: Fraction of the real training split available to this arm (protocol §5.2).
    real_budget_fraction: float = 1.0
    #: Synthetic multiplier k relative to the real budget. Used by `additive`.
    synthetic_multiplier: float = 0.0
    seed: int = 0

    def __post_init__(self) -> None:
        if not 0.0 <= self.synthetic_fraction <= 1.0:
            raise ValueError("synthetic_fraction must lie in [0, 1]")
        if not 0.0 < self.real_budget_fraction <= 1.0:
            raise ValueError("real_budget_fraction must lie in (0, 1]")
        if self.synthetic_multiplier < 0:
            raise ValueError("synthetic_multiplier must be non-negative")

    @property
    def name(self) -> str:
        if self.kind == "substitution":
            return f"sub_p{int(round(self.synthetic_fraction * 100)):03d}"
        if self.kind == "additive":
            return (
                f"add_n{int(round(self.real_budget_fraction * 100)):03d}"
                f"_k{self.synthetic_multiplier:g}"
            )
        return f"realonly_p{int(round(self.synthetic_fraction * 100)):03d}"


def largest_remainder(total: int, weights: np.ndarray) -> np.ndarray:
    """Apportion `total` across bins in proportion to `weights`, exactly.

    Naive rounding of per-class quotas loses or gains images and silently changes
    the training budget, which is the one thing every arm must share.
    """
    weights = np.asarray(weights, dtype=float)
    if total <= 0 or weights.sum() <= 0:
        return np.zeros(len(weights), dtype=int)
    exact = weights / weights.sum() * total
    base = np.floor(exact).astype(int)
    remainder = total - int(base.sum())
    if remainder > 0:
        order = np.argsort(-(exact - base))
        base[order[:remainder]] += 1
    return base


def _ordered_by_class(corpus: Corpus, seed: int) -> dict[int, list[Record]]:
    """A fixed seeded permutation per class. Prefixes of these give nested subsets."""
    gen = rng(seed)
    out: dict[int, list[Record]] = {}
    for cls, sub in corpus.by_class().items():
        records = list(sub)
        idx = gen.permutation(len(records))
        out[cls] = [records[i] for i in idx]
    return out


def _ordered_disjoint_by_seed(
    corpus: Corpus, seed: int, n_seeds: int, base_seed: int = 20_000
) -> dict[int, list[Record]]:
    """Order the pool so that different seeds take *disjoint* prefixes.

    Seeds exist to estimate the variance of "train on N synthetic images". The closest
    thing to five independent generations of N images is five disjoint slices of a pool of
    5N; drawing five independent random subsets instead makes them overlap -- at a 4,740
    pool with 948 drawn per seed, two seeds share about 190 images by chance -- which
    correlates the seeds and understates the variance at exactly the p=1 arm carrying the
    headline claim.

    One permutation, shared across seeds, rotated by seed. Seed *i* starts at its own
    slice and then continues through the others, so:

    - at the full budget the slices are disjoint by construction;
    - below it each seed still takes a *prefix* of one fixed ordering, so the nesting
      property that makes the mixing fractions comparable is preserved;
    - above it (the additive arms, k > 1) seeds necessarily overlap, which is unavoidable
      and harmless -- there is only one pool.
    """
    gen = rng(base_seed)
    out: dict[int, list[Record]] = {}
    for cls, sub in corpus.by_class().items():
        records = list(sub)
        order = [records[i] for i in gen.permutation(len(records))]
        offset = (seed % max(n_seeds, 1)) * (len(order) // max(n_seeds, 1))
        out[cls] = order[offset:] + order[:offset]
    return out


def _take(pool: dict[int, list[Record]], counts: np.ndarray, what: str) -> list[Record]:
    taken: list[Record] = []
    for cls in range(NUM_CLASSES):
        need = int(counts[cls])
        have = len(pool[cls])
        if need > have:
            raise InsufficientSyntheticError(
                f"{what}: need {need} images of class {cls} but pool holds {have}. "
                "Generate more, or lower the budget."
            )
        taken.extend(pool[cls][:need])
    return taken


def build_mixture(
    real_train: Corpus,
    synthetic_pool: Corpus,
    spec: MixSpec,
    n_seeds: int = 5,
) -> Corpus:
    """Materialise one training arm.

    `real_train` is the real training split; `synthetic_pool` is the generated
    corpus for one generator arm. The returned corpus is what a run trains on.
    """
    if len(real_train) == 0:
        raise ValueError("real_train is empty")

    hist = np.array([real_train.class_counts()[c] for c in range(NUM_CLASSES)], dtype=float)

    # The nesting property comes from ordering once, with a spec-independent seed,
    # and then only ever taking prefixes.
    real_order = _ordered_by_class(real_train, spec.seed)
    # The real split is small and every arm draws a nested prefix of it, so a per-seed
    # permutation is right there. The synthetic pool is sized so seeds can be disjoint,
    # and only a rotation of one shared ordering delivers that.
    synth_order = _ordered_disjoint_by_seed(synthetic_pool, spec.seed, n_seeds)

    budget = int(round(len(real_train) * spec.real_budget_fraction))

    if spec.kind == "substitution":
        n_real = int(round(budget * (1.0 - spec.synthetic_fraction)))
        n_synth = budget - n_real
    elif spec.kind == "real_subset":
        n_real = int(round(budget * (1.0 - spec.synthetic_fraction)))
        n_synth = 0
    elif spec.kind == "additive":
        n_real = budget
        n_synth = int(round(budget * spec.synthetic_multiplier))
    else:  # pragma: no cover - guarded by the Literal type
        raise ValueError(f"unknown mix kind {spec.kind!r}")

    real_counts = largest_remainder(n_real, hist)
    synth_counts = largest_remainder(n_synth, hist)

    records = _take(real_order, real_counts, f"{spec.name} (real)")
    records += _take(synth_order, synth_counts, f"{spec.name} (synthetic)")
    if not records:
        raise ValueError(
            f"arm {spec.name} resolves to an empty training set "
            f"(budget={budget}, n_real={n_real}, n_synth={n_synth})"
        )
    return Corpus(records)


def substitution_arms(fractions=(0.0, 0.25, 0.5, 0.75, 1.0), seed: int = 0) -> list[MixSpec]:
    """The headline sweep: 0 / 25 / 50 / 75 / 100 percent synthetic at matched budget."""
    return [MixSpec(kind="substitution", synthetic_fraction=f, seed=seed) for f in fractions]


def control_arms(fractions=(0.25, 0.5, 0.75, 1.0), seed: int = 0) -> list[MixSpec]:
    """Protocol §8.4. The bar a synthetic arm must clear is this, not the p=0 arm."""
    return [MixSpec(kind="real_subset", synthetic_fraction=f, seed=seed) for f in fractions]


def additive_arms(
    real_budgets=(0.10, 0.25, 0.50, 1.00),
    multipliers=(0, 1, 2, 4, 8, 16),
    seed: int = 0,
) -> list[MixSpec]:
    """Protocol §5.2, from which the exchange rate k(n) is read off."""
    return [
        MixSpec(
            kind="additive",
            real_budget_fraction=n,
            synthetic_multiplier=float(k),
            seed=seed,
        )
        for n in real_budgets
        for k in multipliers
    ]
