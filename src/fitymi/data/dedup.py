"""Near-duplicate detection, run before splitting.

ACNE04 ships more files than labelled images and consists of half-face crops, so
near-duplicates across splits are a live risk. A duplicate that straddles train and
test inflates every arm equally, which is worse than it sounds: it compresses the
differences between arms and makes the mixing curve look flatter than it is.

Two signals are combined per protocol §3.2: perceptual hash (cheap, catches
re-encodes and crops) and an embedding cosine (catches the same subject
re-photographed). Connected components of the "duplicate" graph become groups.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import numpy as np

from .records import Corpus, Record

log = logging.getLogger(__name__)


@dataclass
class DedupConfig:
    phash_max_hamming: int = 8
    embed_min_cosine: float = 0.95
    use_embeddings: bool = True
    #: Largest tolerable share of the corpus in a single duplicate cluster. A cluster
    #: bigger than this means the thresholds are matching image *style* rather than
    #: duplication, which silently starves the val and test splits: the group-aware
    #: splitter refuses to break a cluster, so an oversized one lands whole in one
    #: split. Observed in practice on low-texture images, where pHash collides freely.
    max_cluster_fraction: float = 0.05


def phash_bits(path: str, hash_size: int = 8) -> np.ndarray:
    import imagehash
    from PIL import Image

    with Image.open(path) as im:
        h = imagehash.phash(im.convert("RGB"), hash_size=hash_size)
    return np.asarray(h.hash, dtype=bool).ravel()


def _hamming_matrix(bits: np.ndarray) -> np.ndarray:
    """Pairwise Hamming distances between packed boolean rows."""
    b = bits.astype(np.int16)
    # ||a - b||_1 for booleans is the Hamming distance.
    return (b[:, None, :] != b[None, :, :]).sum(axis=2)


class UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def assign_groups(
    corpus: Corpus,
    config: DedupConfig | None = None,
    embedder: Callable[[Sequence[str]], np.ndarray] | None = None,
    phash_fn: Callable[[str], np.ndarray] = phash_bits,
) -> tuple[Corpus, dict]:
    """Return a corpus with `group` set to a duplicate-cluster id, plus a report.

    `embedder` maps paths to L2-normalised vectors. It is injected rather than
    imported so this runs without network access; on the GPU host pass a CLIP
    encoder.
    """
    config = config or DedupConfig()
    paths = [r.path for r in corpus]
    n = len(paths)
    if n == 0:
        return corpus, {"n": 0, "n_groups": 0, "clusters": []}

    uf = UnionFind(n)

    bits = np.stack([phash_fn(p) for p in paths])
    ham = _hamming_matrix(bits)
    ii, jj = np.where(np.triu(ham <= config.phash_max_hamming, k=1))
    for i, j in zip(ii.tolist(), jj.tolist()):
        uf.union(i, j)
    n_phash_links = len(ii)

    n_embed_links = 0
    if config.use_embeddings and embedder is not None:
        emb = embedder(paths)
        emb = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12)
        cos = emb @ emb.T
        ii, jj = np.where(np.triu(cos >= config.embed_min_cosine, k=1))
        for i, j in zip(ii.tolist(), jj.tolist()):
            uf.union(i, j)
        n_embed_links = len(ii)

    roots = [uf.find(i) for i in range(n)]
    canonical: dict[int, str] = {}
    for i, root in enumerate(roots):
        canonical.setdefault(root, f"grp_{len(canonical):06d}")

    regrouped = Corpus(
        Record(path=r.path, label=r.label, source=r.source,
               group=canonical[roots[i]], meta=r.meta)
        for i, r in enumerate(corpus)
    )

    sizes: dict[str, int] = {}
    for g in regrouped.groups:
        sizes[g] = sizes.get(g, 0) + 1
    multi = {g: s for g, s in sizes.items() if s > 1}

    largest = max(sizes.values()) if sizes else 0
    largest_fraction = largest / n if n else 0.0
    over_clustered = largest_fraction > config.max_cluster_fraction

    report = {
        "n_images": n,
        "n_groups": len(canonical),
        "n_duplicate_clusters": len(multi),
        "n_images_in_duplicate_clusters": sum(multi.values()),
        "n_phash_links": int(n_phash_links),
        "n_embedding_links": int(n_embed_links),
        "largest_cluster": largest,
        "largest_cluster_fraction": largest_fraction,
        "over_clustered": over_clustered,
        "config": {
            "phash_max_hamming": config.phash_max_hamming,
            "embed_min_cosine": config.embed_min_cosine,
            "used_embeddings": config.use_embeddings and embedder is not None,
            "max_cluster_fraction": config.max_cluster_fraction,
        },
    }
    if over_clustered:
        log.warning(
            "deduplication produced a cluster holding %.1f%% of the corpus (%d of %d "
            "images). The thresholds are matching style, not duplication; tighten "
            "phash_max_hamming or raise embed_min_cosine before splitting.",
            largest_fraction * 100, largest, n,
        )
    return regrouped, report


def write_report(report: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(report, fh, indent=2)


def merge_groups_by_identity(
    corpus: Corpus,
    embedder: Callable[[Sequence[str]], np.ndarray],
    min_cosine: float,
) -> tuple[Corpus, dict]:
    """Merge existing duplicate groups further, so no subject spans two groups.

    Duplicate detection asks "is this the same picture?". On a face dataset the question
    that actually governs train/test leakage is "is this the same person?", and the two
    have very different answers: ACNE04's 1,457 images are roughly 550-750 individuals,
    each photographed two or three times from different angles. Splitting at the image
    level therefore puts the same face on both sides of the boundary -- in ACNE04's own
    published folds, for 16% of test images even at a threshold where only 0.019% of all
    image pairs qualify.

    Runs after `assign_groups` and only ever coarsens its grouping.
    """
    records = list(corpus)
    n = len(records)
    paths = [r.path for r in records]

    uf = UnionFind(n)
    by_group: dict[str, list[int]] = {}
    for i, r in enumerate(records):
        by_group.setdefault(r.group, []).append(i)
    for members in by_group.values():
        for j in members[1:]:
            uf.union(members[0], j)

    emb = np.asarray(embedder(paths), dtype=np.float32)
    emb = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12)
    cos = emb @ emb.T
    ii, jj = np.where(np.triu(cos >= min_cosine, k=1))
    for a, b in zip(ii.tolist(), jj.tolist()):
        uf.union(a, b)

    roots = [uf.find(i) for i in range(n)]
    canonical: dict[int, str] = {}
    for root in roots:
        canonical.setdefault(root, f"sub_{len(canonical):06d}")
    regrouped = Corpus(
        Record(path=r.path, label=r.label, source=r.source,
               group=canonical[roots[i]], meta=r.meta)
        for i, r in enumerate(records)
    )

    sizes: dict[str, int] = {}
    for g in regrouped.groups:
        sizes[g] = sizes.get(g, 0) + 1
    multi = {g: s for g, s in sizes.items() if s > 1}
    detected = int((np.abs(emb).sum(axis=1) > 0).sum())
    largest = max(sizes.values()) if sizes else 0

    report = {
        "n_images": n,
        "n_subjects": len(canonical),
        "n_multi_image_subjects": len(multi),
        "n_images_in_multi_image_subjects": sum(multi.values()),
        "n_identity_links": int(len(ii)),
        "n_images_with_embedding": detected,
        "largest_subject": largest,
        "largest_subject_fraction": largest / n if n else 0.0,
        "min_cosine": min_cosine,
    }
    log.info(
        "identity grouping: %d images -> %d subjects (%d with more than one image, "
        "largest %d); faces found in %d/%d",
        n, len(canonical), len(multi), largest, detected, n,
    )
    return regrouped, report
