"""Classifier backbones.

The classifier is the instrument, not a variable — it is held fixed across arms so
that differences are attributable to the training data. Architecture is swept only as
a robustness check (protocol §6).

`init` is a first-class experimental condition rather than a convenience flag. An
ImageNet-pretrained backbone has already consumed roughly 1.3M real photographs, so
an arm labelled "100% synthetic" on top of one is nothing of the sort. We run both
and never pool them.
"""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn

from ..data.records import NUM_CLASSES

Init = Literal["scratch", "imagenet"]
Arch = Literal["resnet50", "vit_b_16", "efficientnet_b0", "tinycnn"]


class TinyCNN(nn.Module):
    """A small convnet for the toy/smoke path.

    Not part of the study. It exists so the pipeline can be exercised end to end on
    64px procedural images without a GPU.
    """

    def __init__(self, n_classes: int = NUM_CLASSES, width: int = 32):
        super().__init__()
        def block(cin: int, cout: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv2d(cin, cout, 3, padding=1, bias=False),
                nn.BatchNorm2d(cout),
                nn.ReLU(inplace=True),
                nn.Conv2d(cout, cout, 3, padding=1, bias=False),
                nn.BatchNorm2d(cout),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            )

        self.features = nn.Sequential(
            block(3, width), block(width, width * 2), block(width * 2, width * 4)
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(0.2),
            nn.Linear(width * 4, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x))


def build_model(
    arch: Arch = "resnet50",
    init: Init = "imagenet",
    n_classes: int = NUM_CLASSES,
) -> nn.Module:
    if arch == "tinycnn":
        if init == "imagenet":
            raise ValueError("tinycnn has no pretrained weights; use init='scratch'")
        return TinyCNN(n_classes)

    from torchvision import models as tvm

    weights_arg = {"weights": "DEFAULT" if init == "imagenet" else None}

    if arch == "resnet50":
        model = tvm.resnet50(**weights_arg)
        model.fc = nn.Linear(model.fc.in_features, n_classes)
    elif arch == "vit_b_16":
        model = tvm.vit_b_16(**weights_arg)
        model.heads.head = nn.Linear(model.heads.head.in_features, n_classes)
    elif arch == "efficientnet_b0":
        model = tvm.efficientnet_b0(**weights_arg)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, n_classes)
    else:  # pragma: no cover - guarded by the Literal type
        raise ValueError(f"unknown architecture {arch!r}")
    return model


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
