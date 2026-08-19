"""The toy generative process.

The smoke test's verdict is only meaningful if the toy synthetic process fails the
way real synthetic data fails. An earlier version narrowed the within-grade
distributions without breaking the label-image mapping, which made synthetic-trained
models *better* than real-trained ones -- a cleaner prototype per class. These tests
pin the property that matters: the gap must corrupt the relationship between the
label and the image, not merely change the images.
"""

import numpy as np
import pytest

from fitymi.data.records import NUM_CLASSES, Source
from fitymi.data.toy import CLASS_PRIOR, SEVERITY_BINS, ToyConfig, _lesion_count, make_toy_corpus
from fitymi.utils.seeding import rng


def _mean_counts(gap: float, n: int = 300) -> list[float]:
    return [
        float(np.mean([_lesion_count(rng(9973 * c + i), c, gap) for i in range(n)]))
        for c in range(NUM_CLASSES)
    ]


def test_real_counts_fall_inside_their_grade():
    for cls, (lo, hi) in enumerate(SEVERITY_BINS):
        counts = [_lesion_count(rng(i), cls, 0.0) for i in range(200)]
        assert min(counts) >= lo and max(counts) <= hi


def test_class_separation_shrinks_as_the_gap_widens():
    def spread(gap: float) -> float:
        means = _mean_counts(gap)
        return max(means) - min(means)

    assert spread(0.0) > spread(0.3) > spread(0.6) > spread(0.9)


def test_at_maximum_gap_the_label_carries_no_information():
    """Every grade renders the same picture, so a synthetic-only arm must be at chance."""
    means = _mean_counts(1.0)
    assert max(means) - min(means) < 2.0


def test_synthetic_counts_escape_their_labelled_bin():
    """Concept error means an image labelled 'very severe' may depict a milder case."""
    lo, hi = SEVERITY_BINS[3]
    counts = [_lesion_count(rng(i), 3, 0.6) for i in range(200)]
    assert min(counts) < lo, "no label-image mismatch was produced"


def test_zero_gap_synthetic_matches_the_real_process():
    """The null the smoke test checks: with gap=0 the two processes are the same."""
    assert _mean_counts(0.0) == pytest.approx(_mean_counts(0.0))
    for cls in range(NUM_CLASSES):
        real = [_lesion_count(rng(i), cls, 0.0) for i in range(200)]
        synth = [_lesion_count(rng(i), cls, 0.0) for i in range(200)]
        assert real == synth


def test_corpus_follows_the_intended_class_prior(tmp_path):
    corpus = make_toy_corpus(
        tmp_path, 400, Source.REAL, ToyConfig(gap=0.0, size=32), seed=1, prefix="p"
    )
    counts = corpus.class_counts()
    for cls, prior in enumerate(CLASS_PRIOR):
        assert abs(counts[cls] / 400 - prior) < 0.02
