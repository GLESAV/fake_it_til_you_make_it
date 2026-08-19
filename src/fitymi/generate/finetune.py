"""Closed-set generator fine-tuning (protocol §4.1).

LoRA on a latent diffusion UNet, trained **only** on the real training split. The
module refuses to start if the corpus it is handed contains anything from the
validation or test splits, because a generator that has seen the test set turns the
whole study into an elaborate way of copying labels.

`diffusers` and friends are optional dependencies (`pip install -e '.[generate]'`);
this module is import-safe without them so the rest of the package runs in
environments that have no model weights.
"""

from __future__ import annotations

import importlib.util
import json
import logging
from dataclasses import dataclass, asdict
from pathlib import Path

from ..data.records import NUM_CLASSES, Corpus, Source
from .prompts import SEVERITY_TOKENS, closed_set_prompt

log = logging.getLogger(__name__)


class LeakageError(RuntimeError):
    """Raised when generator training data overlaps a held-out split."""


@dataclass
class FinetuneConfig:
    base_model: str = "runwayml/stable-diffusion-v1-5"
    method: str = "lora"          # "lora" | "dreambooth"
    lora_rank: int = 16
    resolution: int = 512
    batch_size: int = 4
    gradient_accumulation: int = 4
    learning_rate: float = 1e-4
    max_steps: int = 4000
    mixed_precision: str = "fp16"
    seed: int = 0
    #: Per-class token vocabulary. One rare token per Hayashi grade.
    tokens: dict = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.tokens is None:
            self.tokens = dict(SEVERITY_TOKENS)
        if self.method not in {"lora", "dreambooth"}:
            raise ValueError(f"unknown method {self.method!r}")

    def to_dict(self) -> dict:
        return asdict(self)


def assert_no_leakage(train: Corpus, *held_out: Corpus) -> None:
    """Protocol §4.1: the generator sees the training split and nothing else."""
    train_paths = {r.path for r in train}
    train_groups = {r.group for r in train}
    for corpus in held_out:
        paths = {r.path for r in corpus}
        groups = {r.group for r in corpus}
        if train_paths & paths:
            raise LeakageError(
                f"{len(train_paths & paths)} images appear in both the generator's "
                "training data and a held-out split"
            )
        if train_groups & groups:
            raise LeakageError(
                f"{len(train_groups & groups)} deduplication groups straddle the "
                "generator training data and a held-out split"
            )


def _require_diffusers() -> None:
    """Check the optional generation stack without importing it.

    Kept as a spec probe so the rest of the package stays importable in
    environments that have no model weights and no way to fetch them.
    """
    missing = [
        name
        for name in ("accelerate", "diffusers", "peft", "transformers")
        if importlib.util.find_spec(name) is None
    ]
    if missing:
        raise ImportError(
            "generator fine-tuning needs the optional extras: "
            f"pip install -e '.[generate]' (missing: {', '.join(missing)}). Note that "
            "model weights are downloaded from HuggingFace, which some networks block."
        )


def build_caption_manifest(train: Corpus, output: str | Path) -> Path:
    """Write the image/caption manifest the fine-tuning script consumes."""
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as fh:
        for record in train:
            if record.source is not Source.REAL:
                raise ValueError(
                    "the closed-set generator must be trained on real images only; "
                    f"found {record.source.value} at {record.path}"
                )
            fh.write(
                json.dumps(
                    {
                        "file_name": record.path,
                        "text": closed_set_prompt(record.label).text,
                        "label": record.label,
                    }
                )
                + "\n"
            )
    return output


def finetune(
    train: Corpus,
    output_dir: str | Path,
    config: FinetuneConfig | None = None,
    held_out: tuple[Corpus, ...] = (),
) -> Path:
    """Fine-tune the closed-set generator. Requires a GPU and the `generate` extras."""
    config = config or FinetuneConfig()
    assert_no_leakage(train, *held_out)
    _require_diffusers()

    import torch
    from diffusers import StableDiffusionPipeline

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = build_caption_manifest(train, output_dir / "captions.jsonl")

    counts = train.class_counts()
    log.info(
        "fine-tuning %s (%s, rank %d) on %d real images %s",
        config.base_model, config.method, config.lora_rank, len(train), counts,
    )
    thin = [c for c in range(NUM_CLASSES) if counts[c] < 50]
    if thin:
        log.warning(
            "classes %s have fewer than 50 training images; expect the generator to "
            "memorise rather than generalise for those grades, and read the §8.2 "
            "memorisation audit before trusting any arm that leans on them",
            thin,
        )

    pipe = StableDiffusionPipeline.from_pretrained(
        config.base_model,
        torch_dtype=torch.float16 if config.mixed_precision == "fp16" else torch.float32,
        safety_checker=None,
    )
    _train_lora(pipe, manifest, output_dir, config)

    with open(output_dir / "finetune_config.json", "w") as fh:
        json.dump(config.to_dict(), fh, indent=2)
    return output_dir


def _train_lora(pipe, manifest: Path, output_dir: Path, config: FinetuneConfig) -> None:
    """LoRA training loop.

    Kept as a thin wrapper over `diffusers`' reference training script rather than a
    bespoke reimplementation: the point of this study is the downstream comparison,
    and a hand-rolled diffusion trainer would be one more thing reviewers would
    reasonably not trust.
    """
    from peft import LoraConfig

    lora = LoraConfig(
        r=config.lora_rank,
        lora_alpha=config.lora_rank,
        init_lora_weights="gaussian",
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
    )
    pipe.unet.add_adapter(lora)
    raise NotImplementedError(
        "Run diffusers' train_text_to_image_lora.py against "
        f"{manifest} with the settings in {output_dir / 'finetune_config.json'}; "
        "see scripts/finetune_closed_set.sh. This wrapper exists to validate the "
        "config and the no-leakage precondition, not to reimplement the trainer."
    )
