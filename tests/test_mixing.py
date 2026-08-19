"""The mixing sampler is the experiment; these are the properties it must have."""

import numpy as np
import pytest

from fitymi.data.mixing import (
    InsufficientSyntheticError,
    MixSpec,
    build_mixture,
    largest_remainder,
    substitution_arms,
)
from fitymi.data.records import NUM_CLASSES, Source


def test_largest_remainder_is_exact():
    for total in (0, 1, 7, 100, 1457):
        counts = largest_remainder(total, np.array([0.4, 0.33, 0.19, 0.08]))
        assert counts.sum() == total
        assert (counts >= 0).all()


def test_budget_is_matched_across_arms(toy_real, toy_synth):
    sizes = {
        spec.synthetic_fraction: len(build_mixture(toy_real, toy_synth, spec))
        for spec in substitution_arms()
    }
    assert len(set(sizes.values())) == 1, f"budgets differ across arms: {sizes}"


def test_synthetic_fraction_is_honoured(toy_real, toy_synth):
    for spec in substitution_arms():
        mixture = build_mixture(toy_real, toy_synth, spec)
        observed = len(mixture.synthetic) / len(mixture)
        assert abs(observed - spec.synthetic_fraction) < 0.02


def test_class_histogram_preserved(toy_real, toy_synth):
    reference = toy_real.class_counts()
    for spec in substitution_arms():
        counts = build_mixture(toy_real, toy_synth, spec).class_counts()
        for cls in range(NUM_CLASSES):
            assert abs(counts[cls] - reference[cls]) <= 1


def test_real_subsets_are_nested(toy_real, toy_synth):
    """The real half of a higher-synthetic arm must be a subset of a lower one's.

    Without this the arms differ by an independent redraw of real images, and the
    arm-to-arm comparison carries a variance term that has nothing to do with the
    synthetic data.
    """
    kept = {}
    for spec in substitution_arms():
        mixture = build_mixture(toy_real, toy_synth, spec)
        kept[spec.synthetic_fraction] = {r.path for r in mixture.real}
    fractions = sorted(kept)
    for lower, higher in zip(fractions, fractions[1:]):
        assert kept[higher] <= kept[lower], f"p={higher} real set is not nested in p={lower}"


def test_all_synthetic_arm_has_no_real_images(toy_real, toy_synth):
    mixture = build_mixture(toy_real, toy_synth, MixSpec(synthetic_fraction=1.0))
    assert len(mixture.real) == 0
    assert all(r.source is Source.SYNTH_CLOSED for r in mixture)


def test_real_only_arm_has_no_synthetic_images(toy_real, toy_synth):
    mixture = build_mixture(toy_real, toy_synth, MixSpec(synthetic_fraction=0.0))
    assert len(mixture.synthetic) == 0


def test_exhausted_synthetic_pool_raises(toy_real):
    from fitymi.data.records import Corpus

    with pytest.raises(InsufficientSyntheticError):
        build_mixture(toy_real, Corpus([]), MixSpec(synthetic_fraction=1.0))


def test_additive_arm_adds_on_top(toy_real, toy_synth):
    spec = MixSpec(kind="additive", real_budget_fraction=0.5, synthetic_multiplier=2.0)
    mixture = build_mixture(toy_real, toy_synth, spec)
    expected_real = round(len(toy_real) * 0.5)
    assert abs(len(mixture.real) - expected_real) <= 1
    assert abs(len(mixture.synthetic) - 2 * expected_real) <= 2


def test_empty_arm_is_rejected(toy_real, toy_synth):
    with pytest.raises(ValueError, match="empty training set"):
        build_mixture(toy_real, toy_synth, MixSpec(kind="real_subset", synthetic_fraction=1.0))


def test_synthetic_draws_are_disjoint_across_seeds_at_the_full_budget():
    """Seeds estimate the variance of 'train on N synthetic images'.

    With a pool of exactly n_seeds x N, the p=1 arm must give each seed its own slice.
    Overlapping draws correlate the seeds and understate the variance at the arm that
    carries the headline substitution claim.
    """
    from fitymi.data.mixing import MixSpec, build_mixture
    from fitymi.data.records import Corpus, Record, Source

    n_seeds, per_class_real = 5, 4
    real = Corpus(
        Record(path=f"r{c}_{i}.jpg", label=c, source=Source.REAL)
        for c in range(NUM_CLASSES)
        for i in range(per_class_real)
    )
    pool = Corpus(
        Record(path=f"s{c}_{i}.jpg", label=c, source=Source.SYNTH_CLOSED)
        for c in range(NUM_CLASSES)
        for i in range(per_class_real * n_seeds)
    )

    spec = MixSpec(kind="substitution", synthetic_fraction=1.0)
    draws = []
    for seed in range(n_seeds):
        mixture = build_mixture(real, pool, MixSpec(**{**spec.__dict__, "seed": seed}),
                                n_seeds=n_seeds)
        draws.append({r.path for r in mixture})

    assert all(len(d) == len(real) for d in draws)
    for i in range(n_seeds):
        for j in range(i + 1, n_seeds):
            assert not (draws[i] & draws[j]), f"seeds {i} and {j} share synthetic images"
    assert len(set().union(*draws)) == len(pool)


def test_nesting_across_fractions_survives_the_disjoint_ordering():
    """Each seed must still take prefixes, so p=0.5 is a subset of p=1.0."""
    from fitymi.data.mixing import MixSpec, build_mixture
    from fitymi.data.records import Corpus, Record, Source

    real = Corpus(
        Record(path=f"r{c}_{i}.jpg", label=c, source=Source.REAL)
        for c in range(NUM_CLASSES) for i in range(8)
    )
    pool = Corpus(
        Record(path=f"s{c}_{i}.jpg", label=c, source=Source.SYNTH_CLOSED)
        for c in range(NUM_CLASSES) for i in range(40)
    )
    half = build_mixture(real, pool, MixSpec(kind="substitution", synthetic_fraction=0.5,
                                             seed=2), n_seeds=5)
    full = build_mixture(real, pool, MixSpec(kind="substitution", synthetic_fraction=1.0,
                                             seed=2), n_seeds=5)
    half_synth = {r.path for r in half if r.source.is_synthetic}
    full_synth = {r.path for r in full if r.source.is_synthetic}
    assert half_synth <= full_synth
