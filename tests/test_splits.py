"""Splitting, and the mechanical seal on the test set."""

import pytest

from fitymi.data.records import Corpus, Record, Source
from fitymi.data.splits import (
    SealedSplitError,
    SplitBundle,
    assert_disjoint,
    group_stratified_split,
)


def test_split_fractions_are_approximately_honoured(toy_real):
    parts = group_stratified_split(toy_real)
    total = sum(len(p) for p in parts.values())
    assert total == len(toy_real)
    assert abs(len(parts["train"]) / total - 0.65) < 0.08
    assert abs(len(parts["test"]) / total - 0.20) < 0.08


def test_split_is_deterministic(toy_real):
    a = group_stratified_split(toy_real, seed=7)
    b = group_stratified_split(toy_real, seed=7)
    for name in a:
        assert [r.path for r in a[name]] == [r.path for r in b[name]]


def test_groups_never_straddle_splits():
    """A duplicate cluster split across train and test inflates every arm."""
    records = []
    for i in range(60):
        group = f"grp{i // 3}"
        for j in range(3):
            records.append(
                Record(path=f"{group}_{j}.png", label=i % 4, source=Source.REAL, group=group)
            )
    parts = group_stratified_split(Corpus(records))
    seen: dict[str, str] = {}
    for name, part in parts.items():
        for record in part:
            assert seen.setdefault(record.group, name) == name


def test_test_split_is_sealed(toy_real):
    bundle = SplitBundle.from_corpus(toy_real)
    with pytest.raises(SealedSplitError):
        _ = bundle.test


def test_unsealing_requires_a_reason_and_logs_it(toy_real, tmp_path):
    log = tmp_path / "audit.jsonl"
    bundle = SplitBundle.from_corpus(toy_real, audit_log=log)

    with pytest.raises(ValueError):
        bundle.unseal_test(reason="", run_id="r1")

    test = bundle.unseal_test(reason="final evaluation", run_id="r1")
    assert len(test) == bundle.test_size
    assert log.exists()
    assert "final evaluation" in log.read_text()

    bundle.unseal_test(reason="second model", run_id="r2")
    assert len(log.read_text().strip().splitlines()) == 2


def test_summary_does_not_leak_test_composition(toy_real):
    summary = SplitBundle.from_corpus(toy_real).summary()
    assert summary["test"]["classes"] == "<sealed>"


def test_assert_disjoint_catches_overlap(toy_real):
    bundle = SplitBundle.from_corpus(toy_real)
    assert_disjoint(bundle.train, bundle.val)
    with pytest.raises(AssertionError):
        assert_disjoint(bundle.train, bundle.train)


def test_roundtrip_through_disk(toy_real, tmp_path):
    bundle = SplitBundle.from_corpus(toy_real)
    bundle.save(tmp_path)
    reloaded = SplitBundle.load(tmp_path)
    assert [r.path for r in reloaded.train] == [r.path for r in bundle.train]
    assert reloaded.test_size == bundle.test_size


def test_degenerate_split_is_rejected():
    """The failure this guards against: one duplicate cluster swallowing the corpus."""
    from fitymi.data.splits import DegenerateSplitError, check_split_quality

    everything = Corpus(
        Record(f"a{i}.png", i % 4, Source.REAL, group="one_big_cluster") for i in range(100)
    )
    nothing = Corpus([])
    fractions = {"train": 0.65, "val": 0.15, "test": 0.20}

    with pytest.raises(DegenerateSplitError, match="dedup report"):
        check_split_quality({"train": everything, "val": nothing, "test": nothing}, fractions)

    report = check_split_quality(
        {"train": everything, "val": nothing, "test": nothing}, fractions, strict=False
    )
    assert report["problems"]


def test_healthy_split_passes_the_quality_check(toy_real):
    from fitymi.data.splits import check_split_quality

    fractions = {"train": 0.65, "val": 0.15, "test": 0.20}
    parts = group_stratified_split(toy_real, fractions)
    report = check_split_quality(parts, fractions)
    assert report["problems"] == []


def test_over_clustering_is_flagged_in_the_dedup_report(tmp_path):
    """Low-texture images collide under pHash; the report must say so, loudly."""
    from fitymi.data.dedup import DedupConfig, assign_groups
    from fitymi.data.toy import ToyConfig, make_toy_corpus

    corpus = make_toy_corpus(
        tmp_path / "flat", 40, Source.REAL, ToyConfig(gap=0.0, size=32), seed=3, prefix="f"
    )
    _, report = assign_groups(corpus, DedupConfig(phash_max_hamming=32, use_embeddings=False))
    assert report["over_clustered"] is True
    assert report["largest_cluster_fraction"] > 0.05
