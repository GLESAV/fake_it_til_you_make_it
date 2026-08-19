"""Experiment configuration.

Every run is defined by a YAML file that is committed alongside its result. Nothing
in the study is configured by an argument typed once into a shell, because a mixing
curve assembled from 250 runs is only auditable if each point can be re-derived from
a file.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import yaml

from .data.records import Source
from .train.loop import TrainConfig


@dataclass
class DataConfig:
    #: "acne04" for the real study, "toy" for the pipeline smoke test.
    dataset: str = "acne04"
    root: str = "data/acne04"
    splits_dir: str = "data/splits"
    audit_log: str = "runs/test_unseal_audit.jsonl"
    fractions: dict[str, float] = field(
        default_factory=lambda: {"train": 0.65, "val": 0.15, "test": 0.20}
    )
    split_seed: int = 20260819
    dedup: bool = True
    #: Toy-mode knobs, ignored for acne04.
    toy_n_real: int = 600
    toy_n_synth: int = 1200
    toy_gap: float = 0.6
    toy_image_size: int = 64


@dataclass
class GeneratorConfig:
    #: Which synthetic pool this arm draws from.
    source: str = Source.SYNTH_CLOSED.value
    pool_dir: str = "data/synthetic/closed"
    model_path: str = "runwayml/stable-diffusion-v1-5"
    lora_path: str | None = None
    regime: str = "closed_set"
    guidance_scale: float = 3.0
    steps: int = 50
    n_images: int = 20_000


@dataclass
class SweepConfig:
    fractions: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0)
    seeds: tuple[int, ...] = (0, 1, 2, 3, 4)
    #: Protocol §8.4 — real-only arms at the same reduced sizes.
    include_real_subset_controls: bool = True
    #: Protocol §5.2 — the additive sweep the exchange rate is read from.
    include_additive: bool = False
    real_budgets: tuple[float, ...] = (0.10, 0.25, 0.50, 1.00)
    multipliers: tuple[float, ...] = (0, 1, 2, 4, 8, 16)


@dataclass
class ExperimentConfig:
    name: str = "unnamed"
    output_dir: str = "runs/unnamed"
    data: DataConfig = field(default_factory=DataConfig)
    generator: GeneratorConfig = field(default_factory=GeneratorConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    sweep: SweepConfig = field(default_factory=SweepConfig)
    #: Set only for the final evaluation. Guards the sealed test split.
    final_eval: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ExperimentConfig":
        payload = dict(payload)
        sub = {
            "data": DataConfig,
            "generator": GeneratorConfig,
            "train": TrainConfig,
            "sweep": SweepConfig,
        }
        kwargs: dict[str, Any] = {}
        for key, klass in sub.items():
            value = payload.pop(key, None)
            kwargs[key] = klass(**value) if value else klass()
        for key in ("fractions", "seeds", "real_budgets", "multipliers"):
            current = getattr(kwargs["sweep"], key, None)
            if isinstance(current, list):
                setattr(kwargs["sweep"], key, tuple(current))
        return cls(**payload, **kwargs)

    @classmethod
    def load(cls, path: str | Path) -> "ExperimentConfig":
        with open(path) as fh:
            return cls.from_dict(yaml.safe_load(fh) or {})

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as fh:
            yaml.safe_dump(self.to_dict(), fh, sort_keys=False)
