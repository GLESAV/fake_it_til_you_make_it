"""Identity diversity must separate 'many people' from 'one person, many pictures'."""

import numpy as np
import pytest

from fitymi.controls.identity_diversity import identity_diversity
from fitymi.data.records import Corpus, Record, Source


def corpus_of(n: int) -> Corpus:
    return Corpus(Record(path=f"img_{i}.jpg", label=i % 4, source=Source.SYNTH_CLOSED)
                  for i in range(n))


def embedder_from(vectors: np.ndarray):
    lookup = {f"img_{i}.jpg": v for i, v in enumerate(vectors)}
    ref = {f"real_{i}.jpg": v for i, v in enumerate(vectors)}
    lookup.update(ref)

    def embed(paths):
        return np.stack([lookup[p] for p in paths])

    return embed


def one_hot_identities(n_images: int, n_identities: int, dim: int = 16) -> np.ndarray:
    """Each identity is its own basis vector, so within-identity cosine is exactly 1."""
    out = np.zeros((n_images, dim), dtype=np.float32)
    for i in range(n_images):
        out[i, i % n_identities] = 1.0
    return out


def test_counts_distinct_identities():
    vectors = one_hot_identities(40, 8)
    result = identity_diversity(corpus_of(40), embedder_from(vectors), threshold=0.6)
    assert result.n_identities == 8
    assert result.effective_identities == pytest.approx(8.0, abs=1e-6)
    assert result.mean_images_per_identity == pytest.approx(5.0)


def test_collapse_to_one_identity_is_visible():
    vectors = one_hot_identities(40, 1)
    result = identity_diversity(corpus_of(40), embedder_from(vectors), threshold=0.6)
    assert result.n_identities == 1
    assert result.largest_identity_share == pytest.approx(1.0)


def test_effective_count_is_below_raw_count_when_imbalanced():
    """The raw count cannot tell a balanced pool from one dominated by a single face."""
    dim = 16
    vectors = np.zeros((40, dim), dtype=np.float32)
    vectors[:37, 0] = 1.0                      # one identity holds 37 of 40 images
    for k, i in enumerate(range(37, 40)):
        vectors[i, k + 1] = 1.0
    result = identity_diversity(corpus_of(40), embedder_from(vectors), threshold=0.6)
    assert result.n_identities == 4
    assert result.effective_identities < 2.0
    assert result.largest_identity_share == pytest.approx(37 / 40)


def test_coverage_of_reference_detects_a_missed_subset():
    """A generator reproducing half the real identities covers half of them."""
    dim = 16
    ref = np.zeros((8, dim), dtype=np.float32)
    for i in range(8):
        ref[i, i] = 1.0
    synth = ref[:4].copy()

    lookup = {f"img_{i}.jpg": v for i, v in enumerate(synth)}
    lookup.update({f"real_{i}.jpg": v for i, v in enumerate(ref)})

    def embed(paths):
        return np.stack([lookup[p] for p in paths])

    reference = Corpus(Record(path=f"real_{i}.jpg", label=i % 4, source=Source.REAL)
                       for i in range(8))
    result = identity_diversity(corpus_of(4), embed, threshold=0.6, reference=reference)
    assert result.n_reference_identities == 8
    assert result.coverage_of_reference == pytest.approx(0.5)


def test_undetected_faces_do_not_crash():
    vectors = np.zeros((10, 16), dtype=np.float32)   # no face found anywhere
    result = identity_diversity(corpus_of(10), embedder_from(vectors), threshold=0.6)
    assert result.n_with_face == 0
    assert result.n_identities == 0
