"""Image generation (protocol §4.4).

Two things this module insists on:

* **Guidance scale is swept, not defaulted.** Fan et al. (CVPR 2024) found it to be a
  first-order variable for training utility, and it trades fidelity against
  diversity, so the aesthetic default is the wrong choice for a data generator. The
  sweep is selected on the *validation* split only.
* **Everything is logged.** Each generated image carries its prompt, seed, guidance
  scale and generator identity in the corpus metadata, so a mixing curve can be
  re-derived from the artefacts alone.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np

from ..data.mixing import largest_remainder
from ..data.records import NUM_CLASSES, Corpus, Record, Source
from .prompts import Prompt, prompt_stream

log = logging.getLogger(__name__)

#: Protocol §4.4. Selected on validation, never on test.
GUIDANCE_SWEEP = (1.5, 3.0, 5.0, 7.5, 10.0)


@dataclass
class SampleConfig:
    model_path: str = "runwayml/stable-diffusion-v1-5"
    lora_path: str | None = None
    regime: str = "open_set"          # closed_set | open_set | open_set_minimal
    guidance_scale: float = 3.0
    steps: int = 50
    scheduler: str = "dpmpp"          # dpmpp | ddim
    resolution: int = 512
    batch_size: int = 8
    seed: int = 0
    source: str = Source.SYNTH_OPEN.value

    def to_dict(self) -> dict:
        return asdict(self)


def target_counts(reference: Corpus, total: int) -> list[int]:
    """Per-class generation targets that mirror the real class histogram.

    Matching the real prior is what lets a mixing arm differ from the real-only arm
    in image content rather than in label distribution.
    """
    hist = np.array([reference.class_counts()[c] for c in range(NUM_CLASSES)], dtype=float)
    return largest_remainder(total, hist).tolist()


def resolve_device() -> str:
    """CUDA, then Apple Silicon, then CPU.

    The training loop has resolved MPS since this repository moved to Apple Silicon; this
    one still said `"cuda" if available else "cpu"`, so every image was generated on the
    CPU while a 40-core GPU sat idle. It was not slow enough to look broken -- 50 s/image
    against a benchmarked 13.8 -- just slow enough to turn an 18-hour pool into a 66-hour
    one, which is the kind of gap that gets attributed to "MPS is slow" and never
    investigated.
    """
    import torch

    if torch.cuda.is_available():
        return "cuda"
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return "cpu"


def _build_pipeline(config: SampleConfig):
    try:
        import torch
        from diffusers import DDIMScheduler, DPMSolverMultistepScheduler, StableDiffusionPipeline
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "generation needs the optional extras: pip install -e '.[generate]'"
        ) from exc

    pipe = StableDiffusionPipeline.from_pretrained(
        config.model_path,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        safety_checker=None,
    )
    if config.lora_path:
        pipe.load_lora_weights(config.lora_path)
    scheduler_cls = (
        DPMSolverMultistepScheduler if config.scheduler == "dpmpp" else DDIMScheduler
    )
    pipe.scheduler = scheduler_cls.from_config(pipe.scheduler.config)
    pipe = pipe.to(resolve_device())
    pipe.set_progress_bar_config(disable=True)
    return pipe


def generate(
    reference: Corpus,
    total: int,
    output_dir: str | Path,
    config: SampleConfig | None = None,
) -> Corpus:
    """Generate `total` images matching `reference`'s class histogram."""
    import torch

    config = config or SampleConfig()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    counts = target_counts(reference, total)
    log.info("generating %d images %s with %s", total, counts, config.to_dict())

    pipe = _build_pipeline(config)
    prompts = list(
        prompt_stream(range(NUM_CLASSES), counts, regime=config.regime, seed=config.seed)
    )

    records: list[Record] = []
    manifest = open(output_dir / "manifest.jsonl", "w")
    try:
        for start in range(0, len(prompts), config.batch_size):
            batch: list[Prompt] = prompts[start : start + config.batch_size]
            generators = [
                torch.Generator(device=pipe.device).manual_seed(config.seed * 1_000_003 + start + i)
                for i in range(len(batch))
            ]
            images = pipe(
                prompt=[p.text for p in batch],
                negative_prompt=[p.negative for p in batch],
                guidance_scale=config.guidance_scale,
                num_inference_steps=config.steps,
                height=config.resolution,
                width=config.resolution,
                generator=generators,
            ).images

            for i, (image, prompt) in enumerate(zip(images, batch)):
                idx = start + i
                path = output_dir / f"{config.regime}_{prompt.label}_{idx:06d}.png"
                image.save(path)
                meta = {
                    "prompt": prompt.text,
                    "seed": config.seed * 1_000_003 + idx,
                    "guidance_scale": config.guidance_scale,
                    "steps": config.steps,
                    "model": config.model_path,
                    "lora": config.lora_path,
                    **(prompt.meta or {}),
                }
                records.append(
                    Record(path=str(path), label=prompt.label, source=Source(config.source), meta=meta)
                )
                manifest.write(json.dumps({"path": str(path), "label": prompt.label, **meta}) + "\n")
            log.info("generated %d/%d", min(start + config.batch_size, len(prompts)), len(prompts))
    finally:
        manifest.close()

    corpus = Corpus(records)
    corpus.to_jsonl(output_dir / "corpus.jsonl")
    with open(output_dir / "sample_config.json", "w") as fh:
        json.dump(config.to_dict(), fh, indent=2)
    return corpus
