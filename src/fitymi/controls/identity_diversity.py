"""How many distinct people does a synthetic face dataset actually contain?

Protocol §8.10. The substitution question is usually framed as "are the synthetic images
good enough?", but for a face dataset there is a prior question with a sharper answer:
*how many different people are in them?* A generator fine-tuned on a training split
holding ~450 distinct individuals cannot invent identity diversity it never saw, and it
can easily produce far less — mode collapse in identity space is invisible to FID, to
precision/recall, and to every per-image quality metric, because each individual sample
can be flawless while the set as a whole depicts twelve people.

That failure would explain a substitution result all by itself, and it is separate from
memorisation. Memorisation asks whether a generated face is a copy of a *specific
training image*. This asks whether the generated set spans as many *people* as the real
one, which a generator can fail at while copying nothing verbatim.

The instrument is the same ArcFace embedding used to detect subject leakage, so the two
analyses are calibrated against each other: the 38 byte-identical ACNE04 pairs sit at
cosine 1.000, and fewer than 0.02% of genuinely distinct image pairs exceed 0.85.

Three numbers, each answering a different question:

- **Distinct identities** — union-find at the same threshold used for splitting. The
  headline count.
- **Effective identity count** — the exponential of the Shannon entropy of the cluster
  size distribution. A pool of 100 identities where one accounts for 90% of images is
  not the same as 100 balanced identities, and the raw count cannot tell them apart.
- **Identity coverage** — the share of *real* training identities that have at least one
  synthetic near-neighbour. Distinguishes "the generator invented a few new faces" from
  "the generator reproduced a subset of the real ones."
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, asdict
from typing import Callable, Sequence

import numpy as np

from ..data.dedup import UnionFind
from ..data.records import Corpus

log = logging.getLogger(__name__)

Embedder = Callable[[Sequence[str]], np.ndarray]


@dataclass
class IdentityDiversity:
    n_images: int
    n_with_face: int
    n_identities: int
    effective_identities: float
    largest_identity_share: float
    mean_images_per_identity: float
    threshold: float
    #: Share of reference (real) identities with a synthetic neighbour above threshold.
    coverage_of_reference: float | None = None
    n_reference_identities: int | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def _normalise(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)


def _cluster(emb: np.ndarray, threshold: float) -> np.ndarray:
    n = len(emb)
    uf = UnionFind(n)
    cos = emb @ emb.T
    ii, jj = np.where(np.triu(cos >= threshold, k=1))
    for a, b in zip(ii.tolist(), jj.tolist()):
        uf.union(a, b)
    return np.array([uf.find(i) for i in range(n)])


def _effective_count(sizes: np.ndarray) -> float:
    """exp(Shannon entropy) of the cluster-size distribution.

    Equals the raw count when identities are balanced and collapses toward 1 as one
    identity dominates, which is exactly the failure the raw count hides.
    """
    p = sizes / sizes.sum()
    p = p[p > 0]
    return float(np.exp(-(p * np.log(p)).sum()))


def identity_diversity(
    corpus: Corpus,
    embedder: Embedder,
    threshold: float = 0.60,
    reference: Corpus | None = None,
) -> IdentityDiversity:
    """Measure how many distinct people `corpus` depicts.

    If `reference` is given (normally the real training split the generator was fitted
    to), also reports what share of its identities the corpus covers.
    """
    paths = [r.path for r in corpus]
    emb = _normalise(embedder(paths))
    has_face = np.abs(emb).sum(axis=1) > 0
    kept = emb[has_face]

    if len(kept) == 0:
        log.warning("no faces detected in %d images; identity diversity is undefined", len(paths))
        return IdentityDiversity(
            n_images=len(paths), n_with_face=0, n_identities=0,
            effective_identities=0.0, largest_identity_share=0.0,
            mean_images_per_identity=0.0, threshold=threshold,
        )

    roots = _cluster(kept, threshold)
    _, sizes = np.unique(roots, return_counts=True)

    result = IdentityDiversity(
        n_images=len(paths),
        n_with_face=int(has_face.sum()),
        n_identities=int(len(sizes)),
        effective_identities=_effective_count(sizes),
        largest_identity_share=float(sizes.max() / sizes.sum()),
        mean_images_per_identity=float(sizes.mean()),
        threshold=threshold,
    )

    if reference is not None:
        ref_emb = _normalise(embedder([r.path for r in reference]))
        ref_has_face = np.abs(ref_emb).sum(axis=1) > 0
        ref_kept = ref_emb[ref_has_face]
        if len(ref_kept):
            ref_roots = _cluster(ref_kept, threshold)
            covered = set()
            best = (ref_kept @ kept.T).max(axis=1)
            for root, score in zip(ref_roots.tolist(), best.tolist()):
                if score >= threshold:
                    covered.add(root)
            n_ref = len(np.unique(ref_roots))
            result.n_reference_identities = int(n_ref)
            result.coverage_of_reference = float(len(covered) / n_ref)

    log.info(
        "identity diversity: %d images -> %d identities (effective %.1f, largest %.1f%%)",
        result.n_images, result.n_identities, result.effective_identities,
        100 * result.largest_identity_share,
    )
    return result
