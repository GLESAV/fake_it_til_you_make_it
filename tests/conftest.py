import pytest

from fitymi.data.records import Source
from fitymi.data.toy import ToyConfig, make_toy_corpus


@pytest.fixture(scope="session")
def toy_real(tmp_path_factory) -> "object":
    root = tmp_path_factory.mktemp("toy_real")
    return make_toy_corpus(root, 120, Source.REAL, ToyConfig(gap=0.0, size=32), seed=1, prefix="r")


@pytest.fixture(scope="session")
def toy_synth(tmp_path_factory) -> "object":
    root = tmp_path_factory.mktemp("toy_synth")
    return make_toy_corpus(
        root, 240, Source.SYNTH_CLOSED, ToyConfig(gap=0.7, size=32), seed=2, prefix="s"
    )
