"""Protocol §8.2 — memorisation / privacy audit.

For every generated image, find its nearest neighbour in the generator's *training*
set. Two things ride on this:

1. **Validity.** If the closed-set generator is largely reproducing its training
   images, then a "100% synthetic" arm is a laundered copy of the real training split
   and the mixing curve measures nothing.
2. **Privacy.** The main published motivation for synthetic acne faces is anonymity.
   That claim is only as good as this measurement. Priors put partial replication at
   0.5-2% for web-scale generators; a LoRA fitted to ~950 faces should be expected to
   do considerably worse, and nobody has checked.

The embedder is injected rather than imported so this module carries no network
dependency. On the GPU host, pass a CLIP or SSCD encoder; both are recommended, since
CLIP measures semantic similarity and SSCD is built for copy detection.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Callable, Sequence

import numpy as np

from ..data.records import Corpus

Embedder = Callable[[Sequence[str]], np.ndarray]

#: Cosine similarity above which a generated image is treated as a probable copy.
#: Deliberately conservative; the full distribution is reported regardless.
REPLICATION_THRESHOLD = 0.95


@dataclass
class MemorizationResult:
    n_generated: int
    n_reference: int
    threshold: float
    replication_rate: float
    mean_nn_similarity: float
    median_nn_similarity: float
    p95_nn_similarity: float
    max_nn_similarity: float
    top_matches: list[dict]
    release_recommended: bool
    note: str

    def to_dict(self) -> dict:
        return asdict(self)


def _normalise(x: np.ndarray) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)


def nearest_neighbours(
    generated: Corpus,
    reference: Corpus,
    embedder: Embedder,
    batch: int = 512,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (similarity, index-into-reference) for each generated image."""
    ref_paths = [r.path for r in reference]
    gen_paths = [r.path for r in generated]
    if not ref_paths or not gen_paths:
        return np.array([]), np.array([], dtype=int)

    ref_emb = _normalise(np.asarray(embedder(ref_paths), dtype=np.float32))

    sims = np.empty(len(gen_paths), dtype=np.float32)
    idxs = np.empty(len(gen_paths), dtype=np.int64)
    for start in range(0, len(gen_paths), batch):
        chunk = gen_paths[start : start + batch]
        emb = _normalise(np.asarray(embedder(chunk), dtype=np.float32))
        scores = emb @ ref_emb.T
        idxs[start : start + len(chunk)] = scores.argmax(axis=1)
        sims[start : start + len(chunk)] = scores.max(axis=1)
    return sims, idxs


def audit(
    generated: Corpus,
    reference: Corpus,
    embedder: Embedder,
    threshold: float = REPLICATION_THRESHOLD,
    top_k: int = 25,
) -> MemorizationResult:
    sims, idxs = nearest_neighbours(generated, reference, embedder)
    if sims.size == 0:
        return MemorizationResult(
            0, len(reference), threshold, float("nan"), *([float("nan")] * 4),
            [], False, "no images to audit",
        )

    rate = float((sims >= threshold).mean())
    order = np.argsort(-sims)[:top_k]
    top = [
        {
            "generated": generated[int(i)].path,
            "nearest_real": reference[int(idxs[i])].path,
            "similarity": float(sims[i]),
        }
        for i in order
    ]

    if rate >= 0.05:
        note = (
            "material replication: the synthetic pool substantially reproduces the "
            "generator's training set. Treat synthetic-only arms as contaminated and "
            "do not release the images."
        )
        release = False
    elif rate >= 0.01:
        note = (
            "non-trivial replication, in line with or above published rates for "
            "web-scale generators. Report the rate prominently; release only with "
            "the flagged images removed."
        )
        release = False
    else:
        note = "replication below 1%; release with provenance metadata."
        release = True

    return MemorizationResult(
        n_generated=len(generated),
        n_reference=len(reference),
        threshold=threshold,
        replication_rate=rate,
        mean_nn_similarity=float(sims.mean()),
        median_nn_similarity=float(np.median(sims)),
        p95_nn_similarity=float(np.percentile(sims, 95)),
        max_nn_similarity=float(sims.max()),
        top_matches=top,
        release_recommended=release,
        note=note,
    )


def pixel_embedder(size: int = 32) -> Embedder:
    """A dependency-free fallback embedder: downsampled, contrast-normalised pixels.

    Only sensitive to near-exact copies. It is here so the audit is runnable in a
    sandbox without model weights, and so the pipeline is testable; on real data it
    must be replaced with CLIP and SSCD, which is why `audit` takes the embedder as
    an argument.
    """
    from PIL import Image

    def embed(paths: Sequence[str]) -> np.ndarray:
        out = np.empty((len(paths), size * size * 3), dtype=np.float32)
        for i, p in enumerate(paths):
            with Image.open(p) as im:
                arr = np.asarray(
                    im.convert("RGB").resize((size, size), Image.BILINEAR), dtype=np.float32
                )
            arr = arr.ravel()
            arr -= arr.mean()
            out[i] = arr
        return out

    return embed

def calibrate_threshold(
    reference: Sequence[str],
    embedder: Embedder,
    target_per_image_fpr: float = 0.01,
    exclude_pairs: set[tuple[int, int]] | None = None,
) -> dict:
    """Pick a similarity threshold from THIS corpus, not from the copy-detection paper.

    Somepalli et al. report 1.88% of Stable Diffusion generations above SSCD similarity
    0.5 to a training image. That threshold was calibrated against LAION, where two
    arbitrary web images share almost nothing. It does not transfer to a corpus whose
    images are all the same kind of picture.

    Measured on ACNE04 -- 1,457 half-face photographs at one angle under similar lighting
    -- 5.3% of pairs of *genuinely distinct* real images already exceed 0.5, and 82.6% of
    real images have some other real image above it. The maximum similarity between two
    distinct real images is 0.976. Reporting a memorisation rate at 0.5 on that corpus
    would be reporting a false-positive rate.

    So the threshold is set to a stated false-positive rate under the null distribution of
    real-versus-unrelated-real similarity from the same corpus. On ACNE04, a 1% per-image
    false-positive rate needs 0.824, not 0.5.

    `exclude_pairs` removes known-identical index pairs from the null, since a duplicate
    inside the reference set is a true positive and would otherwise inflate the threshold.

    Returns the threshold alongside the null distribution's shape, because a memorisation
    rate quoted without the null it was measured against is not interpretable.
    """
    emb = _normalise(np.asarray(embedder(list(reference)), dtype=np.float32))
    n = len(emb)
    if n < 3:
        raise ValueError("need at least three reference images to estimate a null")
    sim = emb @ emb.T

    best = sim - np.eye(n) * 2.0
    for a, b in exclude_pairs or ():
        best[a, b] = -2.0
        best[b, a] = -2.0
    per_image_best = best.max(axis=1)

    lo, hi = float(per_image_best.min()), 1.0
    for _ in range(80):
        mid = (lo + hi) / 2
        if float((per_image_best >= mid).mean()) > target_per_image_fpr:
            lo = mid
        else:
            hi = mid
    threshold = float(hi)

    upper = np.triu(np.ones((n, n), dtype=bool), k=1)
    for a, b in exclude_pairs or ():
        upper[min(a, b), max(a, b)] = False
    null = sim[upper]

    return {
        "threshold": threshold,
        "target_per_image_fpr": target_per_image_fpr,
        "achieved_per_image_fpr": float((per_image_best >= threshold).mean()),
        "n_reference": n,
        "null_median": float(np.median(null)),
        "null_p99": float(np.percentile(null, 99)),
        "null_max": float(null.max()),
        "published_threshold": 0.5,
        "per_image_fpr_at_published_threshold": float((per_image_best >= 0.5).mean()),
    }
