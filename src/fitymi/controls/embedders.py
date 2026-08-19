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


def face_embedder(
    device: str | None = None,
    det_size: int = 640,
    det_thresh: float = 0.3,
    pad_fraction: float = 0.6,
):
    """An ArcFace identity embedder, returning L2-normalised 512-d vectors.

    Deduplication by perceptual hash or CLIP answers "is this the same picture?".
    For a face dataset the question that governs train/test leakage is "is this the
    same person?", and only an identity embedding answers it. ACNE04 turns out to hold
    roughly 550-750 distinct subjects across 1,457 images, so the two questions have
    very different answers -- see docs/05_acne04_audit.md section 5.

    `pad_fraction` matters more than it looks. ACNE04 images are close crops in which
    the face fills the frame, and RetinaFace expects a face to occupy a fraction of the
    image: detection succeeds on 9% of the corpus raw and 96% after padding the border.
    Any pipeline that runs a face detector over this dataset without padding will
    silently conclude there are almost no faces in it.

    Returns zero vectors for images with no detected face. A zero vector has cosine 0
    with everything, so undetected images simply form their own singleton groups rather
    than being wrongly linked.
    """
    import cv2
    import numpy as np
    from insightface.app import FaceAnalysis

    providers = ["CPUExecutionProvider"]
    if device == "cuda":
        providers = ["CUDAExecutionProvider", *providers]
    app = FaceAnalysis(name="buffalo_l", providers=providers)
    app.prepare(ctx_id=0 if device == "cuda" else -1,
                det_size=(det_size, det_size), det_thresh=det_thresh)

    def embed(paths: Sequence[str]) -> np.ndarray:
        out = np.zeros((len(paths), 512), dtype=np.float32)
        for i, path in enumerate(paths):
            image = cv2.imread(path)
            if image is None:
                continue
            margin = int(pad_fraction * max(image.shape[:2]))
            image = cv2.copyMakeBorder(
                image, margin, margin, margin, margin, cv2.BORDER_CONSTANT, value=(0, 0, 0)
            )
            faces = app.get(image)
            if not faces:
                continue
            largest = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
            out[i] = largest.normed_embedding
        return out

    return embed


#: Facebook Research's SSCD copy-detection descriptor, the instrument Somepalli et al.
#: used to measure that 1.88% of Stable Diffusion generations exceed similarity 0.5 to a
#: training image. We adopt their instrument and their threshold so our memorisation rate
#: is comparable to the only published prior rather than to a number of our own devising.
SSCD_URL = "https://dl.fbaipublicfiles.com/sscd-copy-detection/sscd_disc_mixup.torchscript.pt"
SSCD_THRESHOLD = 0.5


def sscd_embedder(
    weights_path: str = "models/sscd_disc_mixup.torchscript.pt",
    device: str | None = None,
    batch_size: int = 32,
    size: int = 288,
):
    """The SSCD copy-detection descriptor, returning L2-normalised vectors.

    Downloads the TorchScript model on first use (94 MB) and caches it at `weights_path`.

    SSCD is trained to detect *copies under transformation* -- crop, re-encode, colour
    shift, overlay -- which is the question a memorisation audit actually asks. CLIP
    answers "is this the same kind of thing?" and a perceptual hash answers "is this the
    same pixels?"; neither is the same question, and using either would produce a rate not
    comparable to the published prior.

    Resolution 288 and bicubic resizing follow the reference implementation. Getting this
    wrong changes the similarity distribution and therefore the rate.
    """
    import urllib.request
    from pathlib import Path

    import torch
    from PIL import Image
    from torchvision import transforms

    path = Path(weights_path)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        log.info("downloading SSCD weights to %s (94 MB, once)", path)
        urllib.request.urlretrieve(SSCD_URL, path)

    dev = _resolve_device(device)
    model = torch.jit.load(str(path), map_location=dev).eval()

    preprocess = transforms.Compose([
        transforms.Resize((size, size), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    @torch.no_grad()
    def embed(paths: Sequence[str]) -> np.ndarray:
        out: list[np.ndarray] = []
        for start in range(0, len(paths), batch_size):
            batch = []
            for p in paths[start : start + batch_size]:
                with Image.open(p) as im:
                    batch.append(preprocess(im.convert("RGB")))
            vectors = model(torch.stack(batch).to(dev)).float().cpu().numpy()
            out.append(vectors)
        arr = np.concatenate(out, axis=0)
        return arr / (np.linalg.norm(arr, axis=1, keepdims=True) + 1e-12)

    return embed
