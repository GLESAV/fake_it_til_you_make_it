"""Training and evaluation loop.

One recipe, used by every arm (protocol §6.2). Hyperparameters are tuned once on the
p=0 arm and then frozen; the loop deliberately offers no per-arm hooks, because a
per-arm hook is how a study ends up tuning its preferred arm hardest.

Early stopping watches *validation* balanced accuracy. The test split is never
touched here — `evaluate_corpus` is a plain function and the caller must have
unsealed the test set explicitly (see `splits.SplitBundle.unseal_test`).
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from ..data.records import NUM_CLASSES, Corpus
from ..data.torchds import class_weights, make_loader
from ..eval.metrics import EvalResult, balanced_accuracy, evaluate
from ..utils.seeding import seed_everything
from .models import Arch, Init, build_model

log = logging.getLogger(__name__)


@dataclass
class TrainConfig:
    arch: Arch = "resnet50"
    init: Init = "imagenet"
    image_size: int = 224
    batch_size: int = 32
    epochs: int = 60
    lr: float = 3e-4
    weight_decay: float = 0.05
    label_smoothing: float = 0.05
    patience: int = 15
    num_workers: int = 4
    seed: int = 0
    device: str = "auto"
    balanced_loss: bool = True
    amp: bool = True

    def resolve_device(self) -> torch.device:
        """CUDA, then Apple Silicon, then CPU.

        MPS is a first-class target here: a 128 GB unified-memory Mac can hold this
        entire study in memory, and the alternative for someone without a CUDA box is
        not running it at all.
        """
        if self.device != "auto":
            return torch.device(self.device)
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")


@dataclass
class TrainHistory:
    train_loss: list[float] = field(default_factory=list)
    val_balanced_accuracy: list[float] = field(default_factory=list)
    best_epoch: int = -1
    best_val: float = float("-inf")
    stopped_early: bool = False
    seconds: float = 0.0


@torch.no_grad()
def predict(model: nn.Module, loader, device: torch.device) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    logits_all, labels_all = [], []
    for images, labels, _ in loader:
        images = images.to(device, non_blocking=True)
        logits = model(images)
        logits_all.append(logits.float().cpu())
        labels_all.append(labels)
    logits = torch.cat(logits_all)
    labels = torch.cat(labels_all).numpy()
    probs = torch.softmax(logits, dim=1).numpy()
    return labels, probs.argmax(axis=1), probs


def evaluate_corpus(
    model: nn.Module,
    corpus: Corpus,
    config: TrainConfig,
    device: torch.device | None = None,
    return_predictions: bool = False,
):
    """Evaluate `corpus`; optionally also return the per-image predictions.

    Aggregate metrics cannot answer questions about *which* images a model got right,
    and several of the controls are exactly that kind of question -- whether accuracy
    differs between test images whose subject also appears in training and those whose
    subject does not, or how errors distribute across skin tone. Recovering that after
    the fact means retraining, so the predictions are kept.

    The evaluation loader does not shuffle, so the returned arrays are in corpus order.
    """
    device = device or config.resolve_device()
    loader = make_loader(
        corpus,
        batch_size=config.batch_size,
        image_size=config.image_size,
        train=False,
        num_workers=config.num_workers,
        seed=config.seed,
    )
    y_true, y_pred, probs = predict(model, loader, device)
    result = evaluate(y_true, y_pred, probs, NUM_CLASSES)
    if return_predictions:
        return result, {"y_true": [int(v) for v in y_true], "y_pred": [int(v) for v in y_pred]}
    return result


def train_model(
    train_corpus: Corpus,
    val_corpus: Corpus,
    config: TrainConfig,
) -> tuple[nn.Module, TrainHistory]:
    seed_everything(config.seed)
    device = config.resolve_device()
    started = time.time()

    model = build_model(config.arch, config.init, NUM_CLASSES).to(device)

    train_loader = make_loader(
        train_corpus, config.batch_size, config.image_size, True, config.num_workers, config.seed
    )
    val_loader = make_loader(
        val_corpus, config.batch_size, config.image_size, False, config.num_workers, config.seed
    )

    weights = class_weights(train_corpus).to(device) if config.balanced_loss else None
    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=config.label_smoothing)
    optimiser = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=max(config.epochs, 1))

    # Mixed precision is CUDA-only here. MPS autocast exists but is uneven across
    # ops and silently changes numerics, which is not a trade worth making when the
    # whole point is comparing arms to each other.
    use_amp = config.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)

    history = TrainHistory()
    best_state: dict | None = None
    epochs_without_improvement = 0

    for epoch in range(config.epochs):
        model.train()
        running, seen = 0.0, 0
        for images, labels, _ in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimiser.zero_grad(set_to_none=True)
            with torch.amp.autocast(device.type, enabled=use_amp):
                loss = criterion(model(images), labels)
            scaler.scale(loss).backward()
            scaler.step(optimiser)
            scaler.update()
            running += loss.item() * labels.size(0)
            seen += labels.size(0)
        scheduler.step()

        train_loss = running / max(seen, 1)
        y_true, y_pred, _ = predict(model, val_loader, device)
        val_score = balanced_accuracy(y_true, y_pred, NUM_CLASSES)

        history.train_loss.append(train_loss)
        history.val_balanced_accuracy.append(val_score)
        log.info(
            "epoch %d/%d  loss=%.4f  val_bacc=%.4f", epoch + 1, config.epochs, train_loss, val_score
        )

        if val_score > history.best_val:
            history.best_val = val_score
            history.best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= config.patience:
                history.stopped_early = True
                log.info("early stop at epoch %d (patience %d)", epoch + 1, config.patience)
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    history.seconds = time.time() - started
    return model, history


@dataclass
class RunRecord:
    run_id: str
    arm: str
    config: dict
    provenance: dict
    train_summary: dict
    history: dict
    val: dict
    test: dict | None = None
    #: Per-image validation predictions, in corpus order: images, y_true, y_pred.
    #: Kept because several controls ask which images a model got right, not how many,
    #: and recovering that after the fact means retraining.
    val_predictions: dict | None = None

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as fh:
            json.dump(asdict(self), fh, indent=2, default=float)
