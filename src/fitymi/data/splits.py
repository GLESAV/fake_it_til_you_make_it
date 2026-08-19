"""Splitting, and the mechanical enforcement of a sealed test set.

Protocol §3.3 says the real test split touches nothing but the final evaluation.
Stating that in a paper is cheap; the only version anyone should believe is one the
code refuses to violate. `SplitBundle.test` therefore raises unless the caller has
explicitly unsealed it, and every unsealing is appended to an audit log that ships
with the paper.
"""

from __future__ import annotations

import datetime as _dt
import json
from dataclasses import dataclass
from pathlib import Path
import numpy as np

from ..utils.seeding import rng
from .records import NUM_CLASSES, Corpus, Record

#: Frozen in the protocol. Changing this invalidates every committed result.
SPLIT_SEED = 20260819

DEFAULT_FRACTIONS = {"train": 0.65, "val": 0.15, "test": 0.20}


class SealedSplitError(RuntimeError):
    """Raised when the test split is accessed without an explicit unseal."""


class DegenerateSplitError(RuntimeError):
    """Raised when a split is far smaller than requested, or is missing a class.

    Grouping is what keeps duplicates from straddling splits, but an over-broad
    duplicate threshold produces a handful of giant groups that cannot be divided,
    and the val or test split silently shrinks to a size that makes every downstream
    number noise. Failing loudly here is much cheaper than discovering it in the
    confidence intervals.
    """


def check_split_quality(
    parts: dict[str, Corpus],
    fractions: dict[str, float],
    tolerance: float = 0.5,
    strict: bool = True,
) -> dict:
    """Verify the realised splits resemble what was asked for.

    Returns a report; raises `DegenerateSplitError` when `strict` and a split came
    out below `tolerance` times its requested size, or lost a class entirely.
    """
    total = sum(len(p) for p in parts.values())
    problems: list[str] = []
    detail: dict[str, dict] = {}
    for name, part in parts.items():
        requested = fractions[name] * total
        realised = len(part)
        missing = [c for c, n in part.class_counts().items() if n == 0]
        detail[name] = {
            "n": realised,
            "requested": requested,
            "ratio": realised / requested if requested else float("nan"),
            "missing_classes": missing,
        }
        if requested > 0 and realised < tolerance * requested:
            problems.append(
                f"{name} holds {realised} images but {requested:.0f} were requested"
            )
        if missing:
            problems.append(f"{name} contains no examples of class(es) {missing}")

    report = {"total": total, "splits": detail, "problems": problems}
    if problems and strict:
        raise DegenerateSplitError(
            "; ".join(problems)
            + ". This is almost always over-broad deduplication producing groups too "
            "large to divide -- check the dedup report's largest_cluster_fraction."
        )
    return report


def group_stratified_split(
    corpus: Corpus,
    fractions: dict[str, float] | None = None,
    seed: int = SPLIT_SEED,
) -> dict[str, Corpus]:
    """Split by *group*, stratified by label.

    Grouping is what stops near-duplicates from straddling splits; stratification is
    what stops the rare severe grades from landing entirely in one split. Groups are
    assigned to the split whose quota for that group's majority label is furthest
    from being met, which keeps class balance close without ever splitting a group.
    """
    fractions = dict(fractions or DEFAULT_FRACTIONS)
    total = sum(fractions.values())
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"fractions must sum to 1.0, got {total}")

    gen = rng(seed)

    # Collect groups and give each a single representative label (its majority).
    by_group: dict[str, list[Record]] = {}
    for r in corpus:
        by_group.setdefault(r.group, []).append(r)

    group_label: dict[str, int] = {}
    for g, recs in by_group.items():
        counts = np.bincount([r.label for r in recs], minlength=NUM_CLASSES)
        group_label[g] = int(counts.argmax())

    names = list(fractions)
    assigned: dict[str, list[Record]] = {n: [] for n in names}

    for cls in range(NUM_CLASSES):
        groups = [g for g, lab in group_label.items() if lab == cls]
        gen.shuffle(groups)
        n_items = sum(len(by_group[g]) for g in groups)
        quota = {n: fractions[n] * n_items for n in names}
        filled = {n: 0 for n in names}
        # Largest groups first, so a big cluster cannot overshoot a small quota.
        for g in sorted(groups, key=lambda g: -len(by_group[g])):
            deficit = {n: quota[n] - filled[n] for n in names}
            target = max(names, key=lambda n: deficit[n])
            assigned[target].extend(by_group[g])
            filled[target] += len(by_group[g])

    return {n: Corpus(assigned[n]) for n in names}


@dataclass
class SplitBundle:
    """Train / val / test, with the test split held behind a lock.

    Attributes are deliberately awkward to reach for the test split. Use
    `bundle.train` and `bundle.val` freely; call `bundle.unseal_test(reason=...)`
    exactly once per reported model, at the very end.
    """

    train: Corpus
    val: Corpus
    _test: Corpus
    audit_log: Path | None = None
    _unsealed: bool = False

    @classmethod
    def from_corpus(
        cls,
        corpus: Corpus,
        fractions: dict[str, float] | None = None,
        seed: int = SPLIT_SEED,
        audit_log: str | Path | None = None,
    ) -> "SplitBundle":
        parts = group_stratified_split(corpus, fractions, seed)
        return cls(
            train=parts["train"],
            val=parts["val"],
            _test=parts["test"],
            audit_log=Path(audit_log) if audit_log else None,
        )

    @property
    def test(self) -> Corpus:
        raise SealedSplitError(
            "The real test split is sealed (protocol §3.3). It may not be used for "
            "training, early stopping, prompt selection, generator checkpoint "
            "selection or hyperparameter search. Call unseal_test(reason=..., "
            "run_id=...) for a final evaluation; every call is logged."
        )

    def unseal_test(self, reason: str, run_id: str) -> Corpus:
        if not reason or not run_id:
            raise ValueError("unsealing requires both a reason and a run_id")
        self._unsealed = True
        entry = {
            "timestamp": _dt.datetime.now(_dt.timezone.utc).isoformat(),
            "run_id": run_id,
            "reason": reason,
            "n_test": len(self._test),
        }
        if self.audit_log is not None:
            self.audit_log.parent.mkdir(parents=True, exist_ok=True)
            with open(self.audit_log, "a") as fh:
                fh.write(json.dumps(entry) + "\n")
        return self._test

    @property
    def test_size(self) -> int:
        """Size is not leakage — needed for power calculations."""
        return len(self._test)

    def summary(self) -> dict:
        return {
            "train": {"n": len(self.train), "classes": self.train.class_counts()},
            "val": {"n": len(self.val), "classes": self.val.class_counts()},
            "test": {"n": len(self._test), "classes": "<sealed>"},
        }

    def save(self, directory: str | Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self.train.to_jsonl(directory / "train.jsonl")
        self.val.to_jsonl(directory / "val.jsonl")
        self._test.to_jsonl(directory / "test.jsonl")

    @classmethod
    def load(cls, directory: str | Path, audit_log: str | Path | None = None) -> "SplitBundle":
        directory = Path(directory)
        return cls(
            train=Corpus.from_jsonl(directory / "train.jsonl"),
            val=Corpus.from_jsonl(directory / "val.jsonl"),
            _test=Corpus.from_jsonl(directory / "test.jsonl"),
            audit_log=Path(audit_log) if audit_log else None,
        )


def assert_disjoint(*corpora: Corpus) -> None:
    """Guard against the failure mode that invalidates everything else."""
    seen_paths: set[str] = set()
    seen_groups: set[str] = set()
    for c in corpora:
        paths = {r.path for r in c}
        groups = {r.group for r in c}
        if paths & seen_paths:
            raise AssertionError(f"overlapping paths: {sorted(paths & seen_paths)[:5]}")
        if groups & seen_groups:
            raise AssertionError(
                f"overlapping dedup groups across splits: {sorted(groups & seen_groups)[:5]}"
            )
        seen_paths |= paths
        seen_groups |= groups
