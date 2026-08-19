"""Backbones, and the pretraining condition the study treats as a variable."""

import pytest
import torch

from fitymi.train.models import build_model, count_parameters


def test_tinycnn_forward_shape():
    model = build_model("tinycnn", "scratch")
    out = model(torch.randn(2, 3, 64, 64))
    assert out.shape == (2, 4)


def test_tinycnn_has_no_pretrained_weights():
    with pytest.raises(ValueError, match="no pretrained weights"):
        build_model("tinycnn", "imagenet")


def test_scratch_init_is_not_silently_pretrained():
    """A '100% synthetic' arm on pretrained weights is not 100% synthetic."""
    a = build_model("resnet50", "scratch")
    b = build_model("resnet50", "scratch")
    same = torch.equal(a.conv1.weight, b.conv1.weight)
    assert not same, "scratch models should differ between constructions"


def test_head_is_resized_to_the_severity_grades():
    model = build_model("resnet50", "scratch")
    assert model.fc.out_features == 4
    assert count_parameters(model) > 0


def test_unknown_architecture_is_rejected():
    with pytest.raises(ValueError, match="unknown architecture"):
        build_model("not_a_model", "scratch")
