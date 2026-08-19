"""Protocol §8.1 — the real/synthetic discriminability probe.

If a classifier can separate real from generated images trivially, then any model
trained on a mixture can partition on the generation artefact instead of on
pathology. That does not by itself invalidate a mixing result, but it bounds how much
of the effect could be artefact-driven, and a study that reports mixing curves
without reporting this number is asking to be taken on faith.

This is also the control that speaks directly to the acne prior discussed in
`docs/01_related_work.md` §4.1, where a *synthetic* healthy class was added to
otherwise synthetic training data and evaluated on real images.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np

from ..data.records import Corpus, Record, Source
from ..data.torchds import make_loader
from ..train.loop import TrainConfig, predict
from ..train.models import build_model
from ..utils.seeding import rng, seed_everything


@dataclass
class DiscriminabilityResult:
    auc: float
    accuracy: float
    n_real: int
    n_synth: int
    interpretation: str

    def to_dict(self) -> dict:
        return asdict(self)


def _relabel_by_source(corpus: Corpus) -> Corpus:
    """Real -> 0, any synthetic -> 1, encoded in the ordinary label field."""
    return Corpus(
        Record(
            path=r.path,
            label=0 if r.source is Source.REAL else 1,
            source=r.source,
            group=r.group,
            meta=r.meta,
        )
        for r in corpus
    )


def _binary_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Rank-based AUC, ties handled by average rank."""
    y_true = np.asarray(y_true)
    order = np.argsort(scores)
    ranks = np.empty(len(scores), dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1)
    # Average ranks within tied score groups.
    sorted_scores = scores[order]
    start = 0
    for i in range(1, len(sorted_scores) + 1):
        if i == len(sorted_scores) or sorted_scores[i] != sorted_scores[start]:
            ranks[order[start:i]] = ranks[order[start:i]].mean()
            start = i
    n_pos = int((y_true == 1).sum())
    n_neg = int((y_true == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    return float((ranks[y_true == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def _interpret(auc: float) -> str:
    if not np.isfinite(auc):
        return "undefined (one class absent)"
    if auc >= 0.99:
        return (
            "generated images are trivially separable from real ones; any mixed-data "
            "result must be read as potentially artefact-driven"
        )
    if auc >= 0.90:
        return "strongly separable; report as a bound on artefact-driven effects"
    if auc >= 0.70:
        return "partially separable"
    return "near chance; artefact shortcuts are unlikely to explain mixing effects"


def run_probe(
    real: Corpus,
    synthetic: Corpus,
    config: TrainConfig,
    holdout_fraction: float = 0.25,
) -> DiscriminabilityResult:
    """Train a real-vs-synthetic classifier and report held-out AUC.

    Uses the same backbone family as the main study, on a balanced subsample so the
    AUC is not inflated by a class prior.
    """
    seed_everything(config.seed)
    gen = rng(config.seed)

    n = min(len(real), len(synthetic))
    if n == 0:
        raise ValueError("probe needs both real and synthetic images")
    real_idx = gen.permutation(len(real))[:n]
    synth_idx = gen.permutation(len(synthetic))[:n]
    pool = Corpus([*(real[int(i)] for i in real_idx), *(synthetic[int(i)] for i in synth_idx)])
    pool = _relabel_by_source(pool)

    order = gen.permutation(len(pool))
    n_hold = max(1, int(round(len(pool) * holdout_fraction)))
    hold = Corpus(pool[int(i)] for i in order[:n_hold])
    fit = Corpus(pool[int(i)] for i in order[n_hold:])

    # A 2-class head, fitted directly: the main training loop is hard-wired to the
    # 4-class severity task and its balanced loss would be meaningless here.
    probe_config = TrainConfig(**dict(config.__dict__))
    model = build_model(probe_config.arch, probe_config.init, n_classes=2)
    model = _fit_binary(model, fit, probe_config)

    loader = make_loader(
        hold, probe_config.batch_size, probe_config.image_size, False,
        probe_config.num_workers, probe_config.seed,
    )
    y_true, y_pred, probs = predict(model, loader, probe_config.resolve_device())
    auc = _binary_auc(y_true, probs[:, 1])
    acc = float((y_true == y_pred).mean())
    return DiscriminabilityResult(
        auc=auc,
        accuracy=acc,
        n_real=n,
        n_synth=n,
        interpretation=_interpret(auc),
    )


def _fit_binary(model, fit: Corpus, config: TrainConfig):
    import torch
    import torch.nn as nn

    device = config.resolve_device()
    model = model.to(device)
    loader = make_loader(
        fit, config.batch_size, config.image_size, True, config.num_workers, config.seed
    )
    criterion = nn.CrossEntropyLoss()
    optimiser = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    for _ in range(config.epochs):
        model.train()
        for images, labels, _ in loader:
            images = images.to(device)
            labels = labels.to(device)
            optimiser.zero_grad(set_to_none=True)
            loss = criterion(model(images), labels)
            loss.backward()
            optimiser.step()
    return model
