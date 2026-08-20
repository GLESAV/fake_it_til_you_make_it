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


def test_class_weights_ignore_absent_classes():
    """An absent class must get weight zero, not the weight of a singleton class.

    Clamping an absent class to a count of one hands it the largest weight in the vector,
    and training collapses onto labels that never occur. A two-class probe on this codebase
    returned exactly 0.0000 accuracy that way -- every prediction in an empty class -- which
    looks like a broken model rather than a broken loss function.
    """
    from fitymi.data.records import NUM_CLASSES, Corpus, Record, Source
    from fitymi.data.torchds import class_weights

    corpus = Corpus(
        [Record(path=f"a{i}.jpg", label=0, source=Source.REAL) for i in range(80)]
        + [Record(path=f"b{i}.jpg", label=1, source=Source.REAL) for i in range(20)]
    )
    w = class_weights(corpus).tolist()
    assert w[2] == 0.0 and w[3] == 0.0, "absent classes must not be weighted"
    assert w[0] > 0 and w[1] > 0
    assert w[1] > w[0], "the rarer present class should still be up-weighted"
    # Present-class weights should average to one, as inverse-frequency weighting intends.
    assert abs((w[0] * 80 + w[1] * 20) / 100 - 1.0) < 1e-5


def test_class_weights_balance_a_full_corpus():
    from fitymi.data.records import NUM_CLASSES, Corpus, Record, Source
    from fitymi.data.torchds import class_weights

    counts = [40, 30, 20, 10]
    corpus = Corpus([Record(path=f"{c}_{i}.jpg", label=c, source=Source.REAL)
                     for c, n in enumerate(counts) for i in range(n)])
    w = class_weights(corpus).tolist()
    assert all(x > 0 for x in w)
    products = [x * n for x, n in zip(w, counts)]
    assert max(products) - min(products) < 1e-4, "weighted class mass should be equal"
