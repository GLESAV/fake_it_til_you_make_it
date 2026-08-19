"""The §8 controls. Without these the mixing curve is not interpretable."""

import pytest

from fitymi.controls.memorization import audit, nearest_neighbours, pixel_embedder
from fitymi.controls.skintone import estimate_ita
from fitymi.data.records import Corpus, Source
from fitymi.data.toy import ToyConfig, make_toy_corpus


def test_memorization_audit_flags_exact_copies(toy_real):
    """If the 'generated' pool is literally the training set, the audit must say so."""
    copies = Corpus(
        type(r)(path=r.path, label=r.label, source=Source.SYNTH_CLOSED, group=r.group)
        for r in toy_real
    )
    result = audit(copies, toy_real, pixel_embedder(), threshold=0.95)
    assert result.replication_rate == pytest.approx(1.0)
    assert result.release_recommended is False
    assert "not release" in result.note or "removed" in result.note


def test_memorization_audit_is_quiet_on_independent_images(toy_real, tmp_path):
    other = make_toy_corpus(
        tmp_path / "other", 60, Source.SYNTH_OPEN, ToyConfig(gap=0.9, size=32), seed=99, prefix="o"
    )
    result = audit(other, toy_real, pixel_embedder(), threshold=0.99)
    assert result.replication_rate < 0.05
    assert result.max_nn_similarity < 0.999


def test_nearest_neighbours_returns_one_match_per_generated_image(toy_real, toy_synth):
    sims, idxs = nearest_neighbours(toy_synth[:20], toy_real, pixel_embedder())
    assert len(sims) == 20
    assert idxs.max() < len(toy_real)


def test_ita_orders_light_and_dark_patches(tmp_path):
    from PIL import Image

    light = tmp_path / "light.png"
    dark = tmp_path / "dark.png"
    Image.new("RGB", (64, 64), (235, 205, 185)).save(light)
    Image.new("RGB", (64, 64), (95, 65, 50)).save(dark)
    assert estimate_ita(str(light)).ita > estimate_ita(str(dark)).ita


def test_ita_bins_are_reported_coarsely(tmp_path):
    from PIL import Image

    path = tmp_path / "mid.png"
    Image.new("RGB", (64, 64), (200, 160, 130)).save(path)
    estimate = estimate_ita(str(path))
    assert estimate.coarse_bin in {"light", "intermediate", "dark"}
    assert 0.0 <= estimate.coverage <= 1.0


def test_calibrate_threshold_adapts_to_a_homogeneous_corpus():
    """A corpus where everything looks alike needs a higher threshold, not the paper's.

    Two synthetic corpora with the same instrument: one where distinct items are nearly
    orthogonal, one where they all share a dominant component. The calibrated threshold
    must rise for the second, which is the whole point.
    """
    import numpy as np

    from fitymi.controls.memorization import calibrate_threshold

    rng = np.random.default_rng(0)

    def embedder_for(vectors):
        lookup = {f"x{i}.jpg": v for i, v in enumerate(vectors)}
        return lambda paths: np.stack([lookup[p] for p in paths])

    diverse = rng.normal(size=(120, 64)).astype(np.float32)
    shared = diverse + 6.0 * rng.normal(size=(1, 64)).astype(np.float32)

    paths = [f"x{i}.jpg" for i in range(120)]
    a = calibrate_threshold(paths, embedder_for(diverse), target_per_image_fpr=0.05)
    b = calibrate_threshold(paths, embedder_for(shared), target_per_image_fpr=0.05)

    assert b["null_median"] > a["null_median"]
    assert b["threshold"] > a["threshold"]
    assert a["achieved_per_image_fpr"] <= 0.05 + 1e-9
    assert b["achieved_per_image_fpr"] <= 0.05 + 1e-9


def test_calibrate_threshold_excludes_known_duplicates_from_the_null():
    """A duplicate inside the reference set is a true positive, not part of the null."""
    import numpy as np

    from fitymi.controls.memorization import calibrate_threshold

    rng = np.random.default_rng(1)
    vectors = rng.normal(size=(60, 32)).astype(np.float32)
    vectors[1] = vectors[0]                      # an exact duplicate pair
    paths = [f"x{i}.jpg" for i in range(60)]
    lookup = {p: v for p, v in zip(paths, vectors)}
    embed = lambda ps: np.stack([lookup[p] for p in ps])

    naive = calibrate_threshold(paths, embed, target_per_image_fpr=0.05)
    fixed = calibrate_threshold(paths, embed, target_per_image_fpr=0.05,
                                exclude_pairs={(0, 1)})
    assert naive["null_max"] > fixed["null_max"]
