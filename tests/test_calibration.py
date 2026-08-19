"""A threshold must come from the corpus, not from the paper it was published in."""

import numpy as np
import pytest

from fitymi.controls.calibration import calibrate


def embedder_for(vectors):
    lookup = {f"x{i}.jpg": v for i, v in enumerate(vectors)}
    return lambda paths: np.stack([lookup[p] for p in paths])


def paths_for(n):
    return [f"x{i}.jpg" for i in range(n)]


def test_homogeneous_corpus_needs_a_higher_threshold():
    """The whole point: a corpus where everything looks alike shifts the null right."""
    rng = np.random.default_rng(0)
    diverse = rng.normal(size=(150, 64)).astype(np.float32)
    homogeneous = diverse + 8.0 * rng.normal(size=(1, 64)).astype(np.float32)

    a = calibrate(paths_for(150), embedder_for(diverse), target_fpr=0.05)
    b = calibrate(paths_for(150), embedder_for(homogeneous), target_fpr=0.05)

    assert b.null.median > a.null.median
    assert b.threshold > a.threshold


def test_achieved_error_rate_meets_the_target():
    rng = np.random.default_rng(1)
    vectors = rng.normal(size=(200, 32)).astype(np.float32)
    for target in (0.10, 0.05, 0.01):
        result = calibrate(paths_for(200), embedder_for(vectors), target_fpr=target)
        assert result.achieved_fpr <= target + 1e-9


def test_a_reference_threshold_is_reported_and_judged_not_applied():
    """The literature's number appears in the record next to what the corpus supports."""
    rng = np.random.default_rng(2)
    vectors = rng.normal(size=(150, 32)).astype(np.float32) + 8.0 * rng.normal(size=(1, 32)).astype(np.float32)
    result = calibrate(paths_for(150), embedder_for(vectors),
                       target_fpr=0.01, reference_threshold=0.5)

    assert result.threshold != 0.5                     # not applied
    assert result.reference_threshold == 0.5           # but recorded
    assert result.fpr_at_reference > 0.01     # and shown to be wrong here
    assert not result.reference_is_usable


def test_a_reference_threshold_that_does_transfer_is_not_flagged():
    rng = np.random.default_rng(3)
    vectors = rng.normal(size=(150, 64)).astype(np.float32)
    result = calibrate(paths_for(150), embedder_for(vectors),
                       target_fpr=0.05, reference_threshold=0.95)
    assert result.reference_is_usable


def test_known_pairs_are_excluded_from_the_null():
    """A true duplicate inside the corpus is not evidence about the null."""
    rng = np.random.default_rng(4)
    vectors = rng.normal(size=(80, 32)).astype(np.float32)
    vectors[1] = vectors[0]
    naive = calibrate(paths_for(80), embedder_for(vectors), target_fpr=0.05)
    fixed = calibrate(paths_for(80), embedder_for(vectors), target_fpr=0.05,
                      known_pairs={(0, 1)})
    assert naive.null.maximum > fixed.null.maximum


def test_percolation_is_monotone_and_reaches_the_whole_corpus():
    """Lowering the threshold can only merge groups, never split them."""
    rng = np.random.default_rng(5)
    vectors = rng.normal(size=(120, 16)).astype(np.float32)
    result = calibrate(paths_for(120), embedder_for(vectors), target_fpr=0.05,
                       percolation_grid=[0.9, 0.7, 0.5, 0.3, 0.0, -1.0])
    sizes = [largest for _, largest, _ in result.percolation]
    assert sizes == sorted(sizes)
    assert sizes[-1] == 120


def test_too_few_items_is_an_error_not_a_silent_number():
    rng = np.random.default_rng(6)
    with pytest.raises(ValueError):
        calibrate(paths_for(2), embedder_for(rng.normal(size=(2, 8)).astype(np.float32)))


def test_per_pair_mode_is_not_the_same_as_per_item():
    """When most items have a true match, per-item measures the true positive, not error.

    Twenty identities of three images each. Every item has a true neighbour at ~0.9, so a
    per-item rate cannot fall below 100% until the threshold clears that. Same-identity
    pairs are only 3.4% of all pairs, so the per-pair rate is already under a 5% target at
    a much lower threshold. The two modes must therefore disagree, and by a lot.
    """
    rng = np.random.default_rng(11)
    n_id, per_id, dim = 20, 3, 64
    centres = rng.normal(size=(n_id, dim)).astype(np.float32)
    centres /= np.linalg.norm(centres, axis=1, keepdims=True)
    vectors = np.repeat(centres, per_id, axis=0) + 0.25 * rng.normal(
        size=(n_id * per_id, dim)
    ).astype(np.float32)
    paths = paths_for(n_id * per_id)

    per_item = calibrate(paths, embedder_for(vectors), target_fpr=0.05, mode="per_item")
    per_pair = calibrate(paths, embedder_for(vectors), target_fpr=0.05, mode="per_pair")

    assert per_item.threshold > per_pair.threshold
    assert per_pair.achieved_fpr <= 0.05 + 1e-9
    # The per-item threshold has been pushed up past the true within-identity similarity,
    # which is exactly the failure mode this mode exists to avoid.
    assert per_item.threshold > per_pair.null.p99


def test_mode_must_be_one_of_the_two():
    rng = np.random.default_rng(7)
    with pytest.raises(ValueError, match="per_item"):
        calibrate(paths_for(10), embedder_for(rng.normal(size=(10, 8)).astype(np.float32)),
                  mode="whatever")


def test_grouping_needs_a_percolation_cap_as_well_as_an_error_rate():
    """A tolerable pairwise error rate can still merge the whole corpus.

    Grouping takes the transitive closure of the links, so individually rare false links
    chain. The recommended threshold must sit above the error-rate threshold whenever the
    latter percolates.
    """
    rng = np.random.default_rng(12)
    n_id, per_id, dim = 30, 3, 48
    centres = rng.normal(size=(n_id, dim)).astype(np.float32)
    centres /= np.linalg.norm(centres, axis=1, keepdims=True)
    vectors = np.repeat(centres, per_id, axis=0) + 0.45 * rng.normal(
        size=(n_id * per_id, dim)
    ).astype(np.float32)
    paths = paths_for(n_id * per_id)

    result = calibrate(paths, embedder_for(vectors), target_fpr=0.05, mode="per_pair",
                       max_component_fraction=0.10)
    assert result.recommended_threshold is not None
    assert result.recommended_threshold >= result.threshold
    largest = [f for t, _, f in result.percolation if t >= result.recommended_threshold]
    assert all(f <= 0.10 + 1e-9 for f in largest)


def test_percolation_cap_can_be_switched_off():
    rng = np.random.default_rng(13)
    vectors = rng.normal(size=(60, 32)).astype(np.float32)
    result = calibrate(paths_for(60), embedder_for(vectors), target_fpr=0.05,
                       max_component_fraction=None)
    assert result.recommended_threshold is None
    assert result.max_component_fraction is None
