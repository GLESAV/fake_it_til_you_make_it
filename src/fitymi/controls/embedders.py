"""Real embedders for deduplication and the memorisation audit.

`memorization.pixel_embedder` is a dependency-free placeholder; it measures pixel
similarity, which is not what either job needs. Deduplication needs to know when two
photographs show *the same subject*, and the memorisation audit needs to know when a
generated image is a copy of a training image under crop, re-encode and colour shift.
Both are representation questions, not pixel questions.

These embedders are kept out of `dedup.py` and `memorization.py` on purpose: those
modules take an embedder as an argument so they stay importable with no network and no
heavyweight dependency. This module is where the dependency lives.

Why CLIP for ACNE04 specifically: every image in the dataset is a half-face shot at
roughly the same angle under similar lighting, so perceptual hashing has almost no
signal to work with -- at Hamming distance 6 it is already matching pose and background
rather than identity (see `docs/05_acne04_audit.md`). A representation trained on
natural images separates subjects that pHash cannot.
"""

from __future__ import annotations

import logging
from typing import Sequence

import numpy as np

log = logging.getLogger(__name__)

DEFAULT_CLIP_MODEL = "openai/clip-vit-base-patch32"


def _resolve_device(device: str | None) -> str:
    if device is not None:
        return device
    import torch

    if torch.cuda.is_available():
        return "cuda"
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return "cpu"


def clip_embedder(
    model_name: str = DEFAULT_CLIP_MODEL,
    device: str | None = None,
    batch_size: int = 64,
):
    """A CLIP image embedder, returning L2-normalised float32 vectors.

    Loads lazily and caches the model on the returned closure so repeated calls in one
    process pay the load cost once. Requires `transformers` and network access on first
    use; both are absent in the authoring sandbox, which is why this is optional.
    """
    import torch
    from PIL import Image
    from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection

    dev = _resolve_device(device)
    log.info("loading %s on %s", model_name, dev)
    processor = CLIPImageProcessor.from_pretrained(model_name)
    model = CLIPVisionModelWithProjection.from_pretrained(model_name).to(dev).eval()

    @torch.no_grad()
    def embed(paths: Sequence[str]) -> np.ndarray:
        out: list[np.ndarray] = []
        for start in range(0, len(paths), batch_size):
            chunk = paths[start : start + batch_size]
            images = []
            for p in chunk:
                with Image.open(p) as im:
                    images.append(im.convert("RGB"))
            inputs = processor(images=images, return_tensors="pt").to(dev)
            emb = model(**inputs).image_embeds.float().cpu().numpy()
            out.append(emb)
        arr = np.concatenate(out, axis=0)
        return arr / (np.linalg.norm(arr, axis=1, keepdims=True) + 1e-12)

    return embed


def cached_embedder(embedder, cache_path: str):
    """Wrap an embedder so repeated runs over the same paths hit an on-disk cache.

    Deduplication, the memorisation audit and the discriminability probe all embed
    overlapping path sets. Recomputing is pure waste on a study that will run these
    dozens of times.
    """
    from pathlib import Path

    cache: dict[str, np.ndarray] = {}
    path = Path(cache_path)
    if path.exists():
        with np.load(path, allow_pickle=False) as data:
            keys = data["keys"].tolist()
            vecs = data["vecs"]
        cache = {k: vecs[i] for i, k in enumerate(keys)}
        log.info("loaded %d cached embeddings from %s", len(cache), path)

    def embed(paths: Sequence[str]) -> np.ndarray:
        missing = [p for p in paths if p not in cache]
        if missing:
            fresh = embedder(missing)
            for p, v in zip(missing, fresh):
                cache[p] = v
            path.parent.mkdir(parents=True, exist_ok=True)
            keys = list(cache)
            np.savez(
                path,
                keys=np.array(keys),
                vecs=np.stack([cache[k] for k in keys]),
            )
        return np.stack([cache[p] for p in paths])

    return embed
