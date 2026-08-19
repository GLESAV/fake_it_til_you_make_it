"""Determinism helpers.

Every result in this study is reported as a mean over seeds, so seeding has to be
total: a run that is irreproducible cannot contribute to a confidence interval.
"""

from __future__ import annotations

import os
import random

import numpy as np


def seed_everything(seed: int, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)          # also seeds MPS
    torch.cuda.manual_seed_all(seed)  # no-op without CUDA
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def rng(seed: int) -> np.random.Generator:
    """A local generator, for code that must not disturb global state."""
    return np.random.default_rng(seed)
