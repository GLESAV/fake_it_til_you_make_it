"""Metrics.

Balanced accuracy is the primary endpoint (protocol §7): acne severity is imbalanced
and plain accuracy rewards a model that predicts the majority grade. The ordinal
metrics matter because a mild/very-severe confusion is not the same error as a
mild/moderate one, and a model trained on synthetic data can fail in a specifically
ordinal way — flattening the scale — that accuracy alone hides.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np

from ..data.records import NUM_CLASSES, SEVERITY_NAMES


def confusion_matrix(y_true, y_pred, n_classes: int = NUM_CLASSES) -> np.ndarray:
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    cm = np.zeros((n_classes, n_classes), dtype=int)
    np.add.at(cm, (y_true, y_pred), 1)
    return cm


def per_class_recall(y_true, y_pred, n_classes: int = NUM_CLASSES) -> np.ndarray:
    cm = confusion_matrix(y_true, y_pred, n_classes)
    support = cm.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        recall = np.where(support > 0, np.diag(cm) / np.maximum(support, 1), np.nan)
    return recall


def balanced_accuracy(y_true, y_pred, n_classes: int = NUM_CLASSES) -> float:
    recall = per_class_recall(y_true, y_pred, n_classes)
    return float(np.nanmean(recall))


def accuracy(y_true, y_pred) -> float:
    return float(np.mean(np.asarray(y_true) == np.asarray(y_pred)))


def macro_f1(y_true, y_pred, n_classes: int = NUM_CLASSES) -> float:
    cm = confusion_matrix(y_true, y_pred, n_classes)
    tp = np.diag(cm).astype(float)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    denom = 2 * tp + fp + fn
    f1 = np.where(denom > 0, 2 * tp / np.maximum(denom, 1e-12), np.nan)
    return float(np.nanmean(f1))


def mean_absolute_error(y_true, y_pred) -> float:
    return float(np.mean(np.abs(np.asarray(y_true, float) - np.asarray(y_pred, float))))


def quadratic_weighted_kappa(y_true, y_pred, n_classes: int = NUM_CLASSES) -> float:
    """Cohen's kappa with quadratic weights — the standard ordinal agreement measure."""
    cm = confusion_matrix(y_true, y_pred, n_classes).astype(float)
    n = cm.sum()
    if n == 0:
        return float("nan")
    idx = np.arange(n_classes)
    weights = (idx[:, None] - idx[None, :]) ** 2 / max((n_classes - 1) ** 2, 1)
    row = cm.sum(axis=1, keepdims=True)
    col = cm.sum(axis=0, keepdims=True)
    expected = row @ col / n
    denom = (weights * expected).sum()
    if denom <= 0:
        return float("nan")
    return float(1.0 - (weights * cm).sum() / denom)


def expected_calibration_error(y_true, probs, n_bins: int = 15) -> float:
    """ECE. A synthetic-trained model can be accurate and badly calibrated; for any
    clinical framing the second half of that sentence is the one that matters."""
    probs = np.asarray(probs, dtype=float)
    y_true = np.asarray(y_true, dtype=int)
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    correct = (pred == y_true).astype(float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (conf > lo) & (conf <= hi)
        if not mask.any():
            continue
        ece += mask.mean() * abs(correct[mask].mean() - conf[mask].mean())
    return float(ece)


@dataclass
class EvalResult:
    balanced_accuracy: float
    accuracy: float
    macro_f1: float
    mae: float
    qwk: float
    ece: float
    per_class_recall: dict[str, float]
    confusion: list[list[int]]
    n: int

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def tail_recall(self) -> float:
        """Mean recall over severe and very-severe.

        Called out separately because this is where mode-dropping in the generator
        shows up first, and where the clinical stakes actually are.
        """
        vals = [self.per_class_recall[n] for n in SEVERITY_NAMES[2:]]
        vals = [v for v in vals if not np.isnan(v)]
        return float(np.mean(vals)) if vals else float("nan")


def evaluate(y_true, y_pred, probs=None, n_classes: int = NUM_CLASSES) -> EvalResult:
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    recall = per_class_recall(y_true, y_pred, n_classes)
    return EvalResult(
        balanced_accuracy=balanced_accuracy(y_true, y_pred, n_classes),
        accuracy=accuracy(y_true, y_pred),
        macro_f1=macro_f1(y_true, y_pred, n_classes),
        mae=mean_absolute_error(y_true, y_pred),
        qwk=quadratic_weighted_kappa(y_true, y_pred, n_classes),
        ece=(
            expected_calibration_error(y_true, probs)
            if probs is not None
            else float("nan")
        ),
        per_class_recall={
            SEVERITY_NAMES[c]: (float(recall[c]) if not np.isnan(recall[c]) else float("nan"))
            for c in range(n_classes)
        },
        confusion=confusion_matrix(y_true, y_pred, n_classes).tolist(),
        n=int(len(y_true)),
    )
