"""Prompts must fit CLIP's context, because exceeding it fails silently.

Stable Diffusion's text encoder truncates at 77 tokens without raising. An over-long
negative prompt therefore keeps working while quietly no longer suppressing whatever fell
off the end, and the generated pool carries the artefacts you thought you had excluded.
Nothing in the output says so.

Marked so it can be skipped where the tokenizer is unavailable; the check needs the real
tokenizer because a word-count approximation is exactly what let the bug through.
"""

import pytest

from fitymi.generate.prompts import NEGATIVE_PROMPT, closed_set_prompt, open_set_prompt

CLIP_MAX_TOKENS = 77


@pytest.fixture(scope="module")
def tokenizer():
    transformers = pytest.importorskip("transformers")
    try:
        return transformers.CLIPTokenizer.from_pretrained(
            "runwayml/stable-diffusion-v1-5", subfolder="tokenizer"
        )
    except Exception as exc:                                   # offline, gated, rate-limited
        pytest.skip(f"CLIP tokenizer unavailable: {exc}")


def n_tokens(tokenizer, text: str) -> int:
    return len(tokenizer(text)["input_ids"])


def test_negative_prompt_fits(tokenizer):
    count = n_tokens(tokenizer, NEGATIVE_PROMPT)
    assert count <= CLIP_MAX_TOKENS, (
        f"the negative prompt is {count} tokens; CLIP truncates at {CLIP_MAX_TOKENS} "
        "silently, so the tail would stop suppressing anything"
    )


def test_closed_set_prompts_fit(tokenizer):
    for label in range(4):
        assert n_tokens(tokenizer, closed_set_prompt(label).text) <= CLIP_MAX_TOKENS


def test_open_set_prompts_fit_across_the_whole_sampling_space(tokenizer):
    """Open-set prompts are assembled from sampled axes, so the worst case is what counts."""
    worst = max(
        (n_tokens(tokenizer, open_set_prompt(label, seed).text), label, seed)
        for label in range(4)
        for seed in range(200)
    )
    count, label, seed = worst
    assert count <= CLIP_MAX_TOKENS, (
        f"open-set prompt for grade {label}, seed {seed} is {count} tokens"
    )
