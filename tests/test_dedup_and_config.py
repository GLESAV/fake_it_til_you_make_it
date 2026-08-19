"""Deduplication, records, and config round-tripping."""

import pytest

from fitymi.config import ExperimentConfig
from fitymi.data.dedup import DedupConfig, assign_groups
from fitymi.data.records import Corpus, Record, Source


def test_identical_files_land_in_one_group(tmp_path):
    from fitymi.data.toy import ToyConfig, render

    image = render(1, 42, ToyConfig(gap=0.0, size=32), synthetic=False)
    paths = []
    for i in range(3):
        p = tmp_path / f"dup_{i}.png"
        image.save(p)
        paths.append(str(p))
    distinct = tmp_path / "other.png"
    render(3, 7, ToyConfig(gap=0.0, size=32), synthetic=False).save(distinct)

    corpus = Corpus(
        [Record(p, 1, Source.REAL) for p in paths] + [Record(str(distinct), 3, Source.REAL)]
    )
    regrouped, report = assign_groups(corpus, DedupConfig(use_embeddings=False))

    groups = regrouped.groups
    assert len(set(groups[:3])) == 1, "identical images were not clustered"
    assert groups[3] not in groups[:3]
    assert report["n_duplicate_clusters"] == 1
    assert report["n_images_in_duplicate_clusters"] == 3


def test_tighter_phash_threshold_yields_more_groups(toy_real):
    """Sanity on the knob, and a caution about it.

    Low-texture 32px toy images collide readily under pHash, so a permissive
    Hamming threshold over-clusters them. That is a property of the images, not a
    bug, but it is why `DedupConfig` also carries an embedding signal and why the
    dedup report is a paper artefact rather than something to run and forget.
    """
    loose = assign_groups(toy_real[:30], DedupConfig(phash_max_hamming=8, use_embeddings=False))[1]
    tight = assign_groups(toy_real[:30], DedupConfig(phash_max_hamming=0, use_embeddings=False))[1]
    assert tight["n_groups"] > loose["n_groups"]
    # Not 30: even at Hamming 0 a couple of these low-texture renders share a pHash,
    # which is precisely the over-clustering the docstring warns about.
    assert tight["n_groups"] >= 28


def test_record_rejects_an_out_of_range_label():
    with pytest.raises(ValueError, match="outside"):
        Record("x.png", 9, Source.REAL)


def test_record_accepts_a_source_string():
    assert Record("x.png", 0, "synth_open").source is Source.SYNTH_OPEN


def test_corpus_jsonl_roundtrip(tmp_path, toy_real):
    path = tmp_path / "corpus.jsonl"
    toy_real.to_jsonl(path)
    reloaded = Corpus.from_jsonl(path)
    assert len(reloaded) == len(toy_real)
    assert reloaded[0].source is toy_real[0].source
    assert reloaded.class_counts() == toy_real.class_counts()


def test_config_roundtrip(tmp_path):
    config = ExperimentConfig(name="t", output_dir=str(tmp_path / "runs"))
    config.sweep.fractions = (0.0, 0.5, 1.0)
    config.train.arch = "efficientnet_b0"
    path = tmp_path / "c.yaml"
    config.save(path)
    reloaded = ExperimentConfig.load(path)
    assert reloaded.name == "t"
    assert reloaded.sweep.fractions == (0.0, 0.5, 1.0)
    assert reloaded.train.arch == "efficientnet_b0"


def test_shipped_configs_parse():
    from pathlib import Path

    configs = sorted(Path("configs").glob("*.yaml"))
    assert configs, "no configs to validate"
    for path in configs:
        ExperimentConfig.load(path)


def test_duplicate_run_records_are_counted_once(tmp_path):
    """Identical control-arm records under two output dirs must not halve the CIs."""
    import json

    from fitymi.analysis.aggregate import load_runs

    record = {
        "run_id": "shared_control_s0",
        "arm": "realonly_p050",
        "config": {"mix_kind": "real_subset", "synthetic_fraction": 0.5, "seed": 0},
        "val": {"balanced_accuracy": 0.5, "per_class_recall": {}},
    }
    for sub in ("closed", "open"):
        directory = tmp_path / sub
        directory.mkdir()
        (directory / "run_shared_control_s0.json").write_text(json.dumps(record))

    runs = load_runs(tmp_path)
    assert len(runs) == 1
