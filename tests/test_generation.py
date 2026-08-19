"""Prompt construction and the generator's no-leakage precondition."""

import pytest

from fitymi.data.records import Corpus, Record, Source
from fitymi.generate.finetune import LeakageError, assert_no_leakage, build_caption_manifest
from fitymi.generate.prompts import (
    NEGATIVE_PROMPT,
    SEVERITY_TOKENS,
    closed_set_prompt,
    prompt_stream,
)
from fitymi.generate.sample import target_counts


def test_open_set_prompts_are_diverse():
    """Naive class-name prompting is the documented cause of low-diversity pools."""
    prompts = [p.text for p in prompt_stream([0], [64], regime="open_set", seed=3)]
    assert len(set(prompts)) > 40


def test_minimal_regime_is_deliberately_undiverse():
    prompts = [p.text for p in prompt_stream([0], [16], regime="open_set_minimal", seed=3)]
    assert len(set(prompts)) == 1


def test_closed_set_prompts_use_one_rare_token_per_grade():
    tokens = {closed_set_prompt(c).meta["token"] for c in range(4)}
    assert tokens == set(SEVERITY_TOKENS.values())


def test_prompts_carry_a_negative_prompt():
    prompt = next(iter(prompt_stream([2], [1], regime="open_set", seed=0)))
    assert prompt.negative == NEGATIVE_PROMPT


def test_unknown_regime_is_rejected():
    with pytest.raises(ValueError, match="unknown prompt regime"):
        list(prompt_stream([0], [1], regime="nonsense"))


def test_generation_targets_mirror_the_real_histogram(toy_real):
    counts = target_counts(toy_real, 1000)
    assert sum(counts) == 1000
    reference = toy_real.class_counts()
    total = len(toy_real)
    for cls, count in enumerate(counts):
        assert abs(count / 1000 - reference[cls] / total) < 0.02


def test_leakage_check_catches_a_shared_image():
    a = Corpus([Record("x.png", 0, Source.REAL)])
    b = Corpus([Record("x.png", 0, Source.REAL)])
    with pytest.raises(LeakageError):
        assert_no_leakage(a, b)


def test_leakage_check_catches_a_shared_dedup_group():
    a = Corpus([Record("a.png", 0, Source.REAL, group="g1")])
    b = Corpus([Record("b.png", 0, Source.REAL, group="g1")])
    with pytest.raises(LeakageError, match="deduplication groups"):
        assert_no_leakage(a, b)


def test_leakage_check_passes_on_disjoint_splits(toy_real):
    from fitymi.data.splits import SplitBundle

    bundle = SplitBundle.from_corpus(toy_real)
    assert_no_leakage(bundle.train, bundle.val)


def test_closed_set_generator_refuses_synthetic_training_data(tmp_path, toy_synth):
    with pytest.raises(ValueError, match="real images only"):
        build_caption_manifest(toy_synth, tmp_path / "captions.jsonl")
