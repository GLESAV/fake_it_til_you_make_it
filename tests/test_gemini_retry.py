"""Retry accounting in the Gemini backend.

Both tests exist because of a defect in the artefact rather than in the images. The
manifest is a paper artefact -- it is what the per-substrate refusal rates and the pool's
request cost are read from -- and `generate_one` reported the retry *cap* rather than the
number of requests it actually made whenever it gave up. Every content refusal therefore
went into the record as costing six requests when it cost two, and the `max_empty_attempts`
cap could not be shown to be in force from the record it was supposed to govern.
"""

from __future__ import annotations

import json

import pytest

from fitymi.generate.gemini import GeminiConfig, generate_one

pytest.importorskip("google.genai")


class _Models:
    def __init__(self, behaviour):
        self.behaviour = behaviour
        self.calls = 0

    def generate_content(self, **kwargs):
        self.calls += 1
        return self.behaviour(self.calls)


class _Client:
    def __init__(self, behaviour):
        self.models = _Models(behaviour)


def _empty(_call):
    return type("R", (), {"candidates": [], "usage_metadata": None})()


def _rate_limited(_call):
    raise RuntimeError("ClientError: 429 RESOURCE_EXHAUSTED")


def test_an_empty_response_stops_at_the_empty_cap_and_reports_the_real_count():
    client = _Client(_empty)
    config = GeminiConfig(max_attempts=6, max_empty_attempts=2, backoff=0.0)
    result = generate_one(client, "a prompt", config)

    assert result.image_bytes is None
    assert client.models.calls == 2, "the empty cap, not the retry cap, bounds the requests"
    assert result.attempts == 2, "the record must report requests made, not the cap"
    assert result.blocked_reason == "no candidates"


def test_a_rate_limit_is_retried_to_the_full_cap():
    """A 429 says nothing about the prompt, so it does not share the empty cap."""
    client = _Client(_rate_limited)
    config = GeminiConfig(max_attempts=3, max_empty_attempts=2,
                          backoff=0.0, rate_limit_backoff=0.0)
    result = generate_one(client, "a prompt", config)

    assert client.models.calls == 3
    assert result.attempts == 3
    assert "429" in result.blocked_reason


def test_persistently_refused_skips_refusals_and_keeps_rate_limits(tmp_path):
    """The resume filter must distinguish a content decision from a transient fault."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "gwd", "scripts/generate_wide_domain.py")
    gwd = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gwd)

    manifest = tmp_path / "manifest.jsonl"
    rows = [
        # refused twice, never rate-limited -> skip
        {"name": "refused", "path": None, "blocked_reason": "no candidates"},
        {"name": "refused", "path": None, "blocked_reason": "no candidates"},
        # refused once -> not yet enough evidence
        {"name": "once", "path": None, "blocked_reason": "no candidates"},
        # a 429 in the history -> always retry
        {"name": "throttled", "path": None, "blocked_reason": "no candidates"},
        {"name": "throttled", "path": None, "blocked_reason": "ClientError: 429 ..."},
        # refused, then eventually succeeded -> never skip
        {"name": "recovered", "path": None, "blocked_reason": "no candidates"},
        {"name": "recovered", "path": None, "blocked_reason": "no candidates"},
        {"name": "recovered", "path": "recovered.png", "blocked_reason": None},
    ]
    manifest.write_text("\n".join(json.dumps(r) for r in rows) + "\n")

    assert gwd.persistently_refused(manifest) == {"refused"}
    assert gwd.persistently_refused(tmp_path / "absent.jsonl") == set()
