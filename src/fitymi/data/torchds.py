"""Torch dataset and transforms.

The augmentation recipe is fixed across every arm (protocol §6.2). Tuning it per-arm
would let us flatter whichever arm we tuned hardest, which is the quiet way this kind
of study gets its answer.
"""

from __future__ import annotations

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from .records import NUM_CLASSES, Corpus

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transforms(image_size: int = 224, train: bool = True):
    from torchvision import transforms as T

    if train:
        return T.Compose(
            [
                T.RandomResizedCrop(image_size, scale=(0.8, 1.0), ratio=(0.9, 1.1)),
                T.RandomHorizontalFlip(),
                T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02),
                T.ToTensor(),
                T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        )
    return T.Compose(
        [
            T.Resize(int(image_size * 1.14)),
            T.CenterCrop(image_size),
            T.ToTensor(),
            T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


class CorpusDataset(Dataset):
    def __init__(self, corpus: Corpus, image_size: int = 224, train: bool = True):
        self.corpus = corpus
        self.transform = build_transforms(image_size, train)

    def __len__(self) -> int:
        return len(self.corpus)

    def __getitem__(self, idx: int):
        record = self.corpus[idx]
        with Image.open(record.path) as im:
            img = self.transform(im.convert("RGB"))
        return img, record.label, idx


def class_weights(corpus: Corpus) -> torch.Tensor:
    """Inverse-frequency weights.

    Acne severity is imbalanced and the severe tail is the clinically load-bearing
    part of the label space, so the loss is balanced rather than left to follow the
    prior.
    """
    counts = np.array([corpus.class_counts()[c] for c in range(NUM_CLASSES)], dtype=float)
    counts = np.maximum(counts, 1.0)
    w = counts.sum() / (len(counts) * counts)
    return torch.tensor(w, dtype=torch.float32)


def make_loader(
    corpus: Corpus,
    batch_size: int = 32,
    image_size: int = 224,
    train: bool = True,
    num_workers: int = 4,
    seed: int = 0,
) -> DataLoader:
    dataset = CorpusDataset(corpus, image_size=image_size, train=train)
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),  # no benefit on MPS unified memory
        drop_last=False,
        generator=generator if train else None,
        persistent_workers=num_workers > 0,
    )
