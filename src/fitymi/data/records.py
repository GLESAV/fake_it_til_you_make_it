"""The one data structure everything else agrees on.

A `Record` is a single image with a label and a provenance tag. The provenance tag
is load-bearing: mixing, the discriminability control and the memorisation audit all
key off whether an image is real or generated, so it is never inferred from a path.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, asdict, field
from enum import Enum
from pathlib import Path
from typing import Iterable, Iterator, Sequence


class Source(str, Enum):
    REAL = "real"
    SYNTH_CLOSED = "synth_closed"   # generator fine-tuned on the real train split
    SYNTH_OPEN = "synth_open"       # off-the-shelf generator, never saw our data
    SYNTH_EDIT = "synth_edit"       # lesions composited onto real skin

    @property
    def is_synthetic(self) -> bool:
        return self is not Source.REAL


# Hayashi severity, as distributed with ACNE04.
SEVERITY_NAMES = ("mild", "moderate", "severe", "very_severe")
NUM_CLASSES = len(SEVERITY_NAMES)


@dataclass(frozen=True)
class Record:
    path: str
    label: int
    source: Source
    #: Deduplication cluster. Records sharing a group never straddle a split.
    group: str = ""
    meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not 0 <= self.label < NUM_CLASSES:
            raise ValueError(f"label {self.label} outside 0..{NUM_CLASSES - 1}")
        if not isinstance(self.source, Source):
            object.__setattr__(self, "source", Source(self.source))
        if not self.group:
            object.__setattr__(self, "group", self.path)


class Corpus(Sequence[Record]):
    """An ordered, immutable collection of records."""

    def __init__(self, records: Iterable[Record]):
        self._records: tuple[Record, ...] = tuple(records)

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, idx):  # type: ignore[override]
        if isinstance(idx, slice):
            return Corpus(self._records[idx])
        return self._records[idx]

    def __iter__(self) -> Iterator[Record]:
        return iter(self._records)

    def __repr__(self) -> str:
        return f"Corpus(n={len(self)}, {self.class_counts()}, {self.source_counts()})"

    # -- views -------------------------------------------------------------

    def filter(self, **kwargs) -> "Corpus":
        def keep(r: Record) -> bool:
            return all(getattr(r, k) == v for k, v in kwargs.items())

        return Corpus(r for r in self._records if keep(r))

    @property
    def real(self) -> "Corpus":
        return Corpus(r for r in self._records if r.source is Source.REAL)

    @property
    def synthetic(self) -> "Corpus":
        return Corpus(r for r in self._records if r.source.is_synthetic)

    def by_class(self) -> dict[int, "Corpus"]:
        buckets: dict[int, list[Record]] = {c: [] for c in range(NUM_CLASSES)}
        for r in self._records:
            buckets[r.label].append(r)
        return {c: Corpus(v) for c, v in buckets.items()}

    # -- summaries ---------------------------------------------------------

    def class_counts(self) -> dict[int, int]:
        counts = {c: 0 for c in range(NUM_CLASSES)}
        for r in self._records:
            counts[r.label] += 1
        return counts

    def source_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for r in self._records:
            counts[r.source.value] = counts.get(r.source.value, 0) + 1
        return counts

    @property
    def labels(self) -> list[int]:
        return [r.label for r in self._records]

    @property
    def groups(self) -> list[str]:
        return [r.group for r in self._records]

    def concat(self, other: "Corpus") -> "Corpus":
        return Corpus([*self._records, *other])

    # -- persistence -------------------------------------------------------

    def to_jsonl(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as fh:
            for r in self._records:
                d = asdict(r)
                d["source"] = r.source.value
                fh.write(json.dumps(d) + "\n")

    @classmethod
    def from_jsonl(cls, path: str | Path) -> "Corpus":
        records = []
        with open(path) as fh:
            for line in fh:
                line = line.strip()
                if line:
                    records.append(Record(**json.loads(line)))
        return cls(records)

    def to_csv(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(["path", "label", "source", "group"])
            for r in self._records:
                writer.writerow([r.path, r.label, r.source.value, r.group])
