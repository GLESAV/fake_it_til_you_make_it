"""The §8 controls. Without these the mixing curve is not interpretable."""

import numpy as np
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
