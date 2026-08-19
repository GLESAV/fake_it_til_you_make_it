"""Orchestration: end-to-end on a handful of toy images, plus resume behaviour."""

from pathlib import Path

import pytest

from fitymi.config import DataConfig, ExperimentConfig, GeneratorConfig, SweepConfig
from fitymi.data.mixing import MixSpec
from fitymi.data.records import Source
from fitymi.data.toy import ToyConfig, make_toy_corpus
from fitymi.experiment import load_synthetic_pool, prepare_data, run_arm
from fitymi.train.loop import TrainConfig


@pytest.fixture(scope="module")
def tiny_study(tmp_path_factory):
    root = tmp_path_factory.mktemp("tiny")
    config = ExperimentConfig(
        name="tiny",
        output_dir=str(root / "runs"),
        data=DataConfig(
            dataset="toy",
            root=str(root / "data"),
            splits_dir=str(root / "data" / "splits"),
            audit_log=str(root / "audit.jsonl"),
            toy_n_real=80,
            toy_image_size=32,
            dedup=False,
        ),
        generator=GeneratorConfig(
            source=Source.SYNTH_CLOSED.value, pool_dir=str(root / "pool")
        ),
        train=TrainConfig(
            arch="tinycnn", init="scratch", image_size=32, batch_size=16, epochs=1,
            patience=1, num_workers=0, device="cpu", amp=False,
        ),
        sweep=SweepConfig(fractions=(0.0, 1.0), seeds=(0,)),
        final_eval=True,
    )
    bundle = prepare_data(config)
    pool = make_toy_corpus(
        Path(config.generator.pool_dir) / "images", 120, Source.SYNTH_CLOSED,
        ToyConfig(gap=0.8, size=32), seed=77, prefix="s",
    )
    pool.to_jsonl(Path(config.generator.pool_dir) / "corpus.jsonl")
    return config, bundle


def test_pool_loader_rejects_a_mislabelled_pool(tiny_study, tmp_path):
    config, _ = tiny_study
    from dataclasses import replace

    wrong = replace(config, generator=GeneratorConfig(
        source=Source.SYNTH_OPEN.value, pool_dir=config.generator.pool_dir
    ))
    with pytest.raises(ValueError, match="not tagged"):
        load_synthetic_pool(wrong)


def test_run_arm_produces_a_record_with_test_metrics(tiny_study):
    config, bundle = tiny_study
    pool = load_synthetic_pool(config)
    record = run_arm(MixSpec(synthetic_fraction=1.0), bundle, pool, config, seed=0)

    assert record.test is not None, "final_eval was set, so the test block must exist"
    assert record.train_summary["sources"] == {"synth_closed": record.train_summary["n"]}
    assert Path(config.output_dir, f"run_{record.run_id}.json").exists()
    assert Path(config.data.audit_log).exists(), "unsealing must be logged"


def test_run_arm_resumes_instead_of_retraining(tiny_study):
    """A 250-run sweep must survive a crash without restarting from zero."""
    config, bundle = tiny_study
    pool = load_synthetic_pool(config)
    spec = MixSpec(synthetic_fraction=0.0)

    first = run_arm(spec, bundle, pool, config, seed=0)
    second = run_arm(spec, bundle, pool, config, seed=0)

    assert second.run_id == first.run_id
    # Wall-clock is recorded per run; an identical value means no retraining happened.
    assert second.history["seconds"] == first.history["seconds"]

    third = run_arm(spec, bundle, pool, config, seed=0, overwrite=True)
    assert third.history["seconds"] != first.history["seconds"]


def test_provenance_is_recorded(tiny_study):
    config, bundle = tiny_study
    pool = load_synthetic_pool(config)
    record = run_arm(MixSpec(synthetic_fraction=0.5), bundle, pool, config, seed=0)
    provenance = record.provenance
    assert provenance["config_hash"]
    assert provenance["dataset_hash"]
    assert "python" in provenance["packages"]


def _run_config(config, seed=0, label="none"):
    from dataclasses import asdict, replace

    from fitymi.data.mixing import MixSpec
    from fitymi.experiment import build_run_config
    from fitymi.train.loop import TrainConfig

    spec = MixSpec(kind="substitution", synthetic_fraction=0.0)
    train_config = TrainConfig(**{**asdict(config.train), "seed": seed})
    return build_run_config(spec, train_config, config, seed, label)


def test_run_id_separates_arms_that_differ_only_in_the_split():
    """Two arms differing only in how the corpus was split are different experiments.

    Without the data block in the run config they hash identically, and the resume path
    would skip the second as already done if they ever shared an output directory.
    """
    from dataclasses import replace

    from fitymi.config import ExperimentConfig
    from fitymi.utils.provenance import hash_obj

    image_level = ExperimentConfig()
    subject_disjoint = replace(
        image_level,
        data=replace(image_level.data, subject_grouping=True,
                     splits_dir="data/splits_subject"),
    )
    assert hash_obj(_run_config(image_level)) != hash_obj(_run_config(subject_disjoint))


def test_run_id_separates_the_two_label_sets():
    """Protocol §8.9 repeats the headline under the second annotation team's labels."""
    from dataclasses import replace

    from fitymi.config import ExperimentConfig
    from fitymi.utils.provenance import hash_obj

    v1 = ExperimentConfig()
    v2 = replace(v1, data=replace(v1.data, label_files=("NNEW_v2_all.txt",)))
    assert hash_obj(_run_config(v1)) != hash_obj(_run_config(v2))


def test_run_id_ignores_the_config_filename():
    """The p=0 arm must resolve to one run across the closed- and open-set sweeps."""
    from dataclasses import replace

    from fitymi.config import ExperimentConfig
    from fitymi.utils.provenance import hash_obj

    closed = ExperimentConfig(name="acne04_closed")
    reopened = replace(closed, name="acne04_open")
    assert hash_obj(_run_config(closed)) == hash_obj(_run_config(reopened))
