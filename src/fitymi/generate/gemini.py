"""Gemini image generation backend.

The point of this arm, and the reason the study exists: a large prompted model can be
asked for coverage that no real dermatology dataset has. ACNE04 is roughly 600 people, one
cohort, one capture protocol, overwhelmingly one skin tone; a generator fine-tuned on it
inherits all of that by construction and cannot exceed it. A prompted frontier model is not
bounded that way, and whether the breadth it can produce is *useful* -- whether a classifier
trained on it works on real patients -- is the question.

Real data is validation here, not training material. Nothing in this module reads the real
corpus.

Two practical notes that cost time to learn:

- **Vertex works with plain application-default credentials.** No API key is needed if
  `gcloud auth application-default login` has been run; set `GOOGLE_GENAI_USE_VERTEXAI=true`
  and a project. The `global` location resolves the image models when a regional one does
  not.
- **Clinical dermatology prompts are not safety-blocked**, but individual generations do
  come back empty. The caller must treat a missing image as normal and retry, not as an
  error, or a long run dies partway with a confusing message.
"""

from __future__ import annotations

import logging
import os
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterator, Sequence

log = logging.getLogger(__name__)

#: "nano banana". The preview alias is kept as a fallback because the stable name has
#: moved once already.
DEFAULT_MODEL = "gemini-2.5-flash-image"
FALLBACK_MODELS = ("gemini-2.5-flash-image-preview",)


@dataclass
class GeminiConfig:
    model: str = DEFAULT_MODEL
    project: str | None = None
    location: str = "global"
    use_vertex: bool = True
    #: Attempts per prompt when the failure is a rate limit. Worth retrying: the limit is
    #: bursty rather than absolute, and a prompt that 429s now succeeds later.
    max_attempts: int = 6
    #: Attempts when the model returns no image at all. Kept much lower on purpose. An
    #: empty response is a content decision, not a transient fault, and it repeats: on the
    #: wide-domain pool the empty rate is 83% for close-up cheek macros and 0% for wax
    #: moulages and textbook plates, which is a property of the prompt rather than of the
    #: moment. Retrying those six times spends six requests of a rate-limited quota to
    #: fail six times, and those wasted requests are themselves what provokes the next 429.
    max_empty_attempts: int = 2
    #: Seconds to wait after a failure, doubled each attempt.
    backoff: float = 2.0
    #: Separate, much longer backoff for 429s. A rate limit is not a transient error and
    #: retrying it on the same schedule as one just burns the remaining quota faster.
    rate_limit_backoff: float = 20.0
    #: Requests in flight. Two, not more: measured on the default project quota, six
    #: concurrent requests produced a 429 storm that failed 12 of 14 images, while two
    #: sustained throughput without hitting the limit at all. The cheap tier is
    #: rate-limited, and going wider is slower.
    concurrency: int = 2

    def to_dict(self) -> dict:
        return asdict(self)


class GeminiUnavailableError(RuntimeError):
    pass


def build_client(config: GeminiConfig):
    try:
        from google import genai
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise GeminiUnavailableError(
            "the Gemini arm needs google-genai: pip install -e '.[gemini]'"
        ) from exc

    if config.use_vertex:
        os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "true"
        project = config.project or os.environ.get("GOOGLE_CLOUD_PROJECT")
        if not project:
            raise GeminiUnavailableError(
                "set generator.project or GOOGLE_CLOUD_PROJECT for the Vertex route"
            )
        os.environ["GOOGLE_CLOUD_PROJECT"] = project
        os.environ["GOOGLE_CLOUD_LOCATION"] = config.location
    elif not os.environ.get("GEMINI_API_KEY"):
        raise GeminiUnavailableError("set GEMINI_API_KEY for the AI Studio route")

    return genai.Client()


@dataclass
class GeminiResult:
    image_bytes: bytes | None
    model: str
    attempts: int
    seconds: float
    #: Billed token counts, so the cost of a pool is a measurement rather than an estimate.
    prompt_tokens: int | None = None
    output_tokens: int | None = None
    blocked_reason: str | None = None


def generate_one(client, prompt: str, config: GeminiConfig) -> GeminiResult:
    """One image, with retries. Returns a result whose `image_bytes` may be None.

    An empty response is a normal outcome rather than an exception: the model sometimes
    returns text instead of an image, and a caller generating thousands must be able to
    count those rather than crash on them.
    """
    from google.genai import types

    started = time.time()
    last_reason = None
    # Retry the SAME model. The fallback exists for "this model name has moved", which
    # surfaces as a 404 -- not for transient failures. Escalating on every retry turns one
    # flaky request into a guaranteed 404 against a name that does not exist on this
    # backend, which cost 14 of the first 24 images generated here.
    model = config.model
    tried_fallback = False

    empty_count = 0
    for attempt in range(1, config.max_attempts + 1):
        try:
            response = client.models.generate_content(
                model=model,
                contents=prompt,
                config=types.GenerateContentConfig(response_modalities=["IMAGE"]),
            )
            usage = getattr(response, "usage_metadata", None)
            candidates = getattr(response, "candidates", None) or []
            if not candidates:
                last_reason = "no candidates"
            for candidate in candidates:
                for part in getattr(candidate.content, "parts", None) or []:
                    blob = getattr(part, "inline_data", None)
                    if blob is not None and blob.data:
                        return GeminiResult(
                            image_bytes=blob.data, model=model, attempts=attempt,
                            seconds=time.time() - started,
                            prompt_tokens=getattr(usage, "prompt_token_count", None),
                            output_tokens=getattr(usage, "candidates_token_count", None),
                        )
                last_reason = str(getattr(candidate, "finish_reason", None) or "no image part")
            rate_limited = False
            empty_count += 1
            if empty_count >= config.max_empty_attempts:
                break
        except Exception as exc:  # noqa: BLE001 - retried, then reported
            last_reason = f"{type(exc).__name__}: {exc}"[:200]
            rate_limited = "RESOURCE_EXHAUSTED" in str(exc) or "429" in str(exc)
            if "NOT_FOUND" in str(exc) and not tried_fallback and FALLBACK_MODELS:
                model, tried_fallback = FALLBACK_MODELS[0], True
                log.warning("%s not found here; falling back to %s", config.model, model)
        if attempt < config.max_attempts:
            base = config.rate_limit_backoff if rate_limited else config.backoff
            # Jitter, so concurrent workers that were rate-limited together do not all
            # come back at the same instant and trigger the limit again.
            time.sleep(base * (2 ** (attempt - 1)) * (0.75 + 0.5 * random.random()))

    # `attempt` and not `config.max_attempts`. Reporting the cap here was wrong and it
    # was wrong in the direction that hides work: the empty-response path breaks out at
    # `max_empty_attempts` (2), so every content refusal in the manifest was recorded as
    # having cost 6 requests when it cost 2. That made the manifest's request accounting
    # an overcount and, worse, made the cap itself unverifiable from the artefact -- you
    # could not tell from the record whether it was in force.
    return GeminiResult(image_bytes=None, model=config.model,
                        attempts=attempt, seconds=time.time() - started,
                        blocked_reason=last_reason)


def generate_many(
    prompts: Sequence[tuple[str, str]],
    output_dir: str | Path,
    config: GeminiConfig | None = None,
) -> Iterator[dict]:
    """Generate `(name, prompt)` pairs concurrently, yielding one record each.

    Yields rather than returns so a long run can be checkpointed and inspected while it
    is still going, which matters when the pool takes hours and costs money per image.
    Existing files are skipped, so an interrupted run resumes.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    config = config or GeminiConfig()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    client = build_client(config)

    pending = [(name, prompt) for name, prompt in prompts
               if not (output_dir / f"{name}.png").exists()]
    log.info("generating %d images (%d already present) with %s",
             len(pending), len(prompts) - len(pending), config.model)

    with ThreadPoolExecutor(max_workers=config.concurrency) as pool:
        futures = {pool.submit(generate_one, client, prompt, config): (name, prompt)
                   for name, prompt in pending}
        for future in as_completed(futures):
            name, prompt = futures[future]
            result = future.result()
            path = output_dir / f"{name}.png"
            if result.image_bytes:
                path.write_bytes(result.image_bytes)
            yield {
                "name": name,
                "path": str(path) if result.image_bytes else None,
                "prompt": prompt,
                "model": result.model,
                "attempts": result.attempts,
                "seconds": round(result.seconds, 2),
                "prompt_tokens": result.prompt_tokens,
                "output_tokens": result.output_tokens,
                "blocked_reason": result.blocked_reason,
            }
