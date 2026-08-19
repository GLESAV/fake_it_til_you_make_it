"""Calibrate a similarity threshold against the corpus it will be used on.

Three times in this study a published or library default threshold turned out to be wrong
for ACNE04, in the same way each time:

- **Perceptual hash, Hamming ≤ 8.** Percolates into a single cluster holding 14.7% of the
  corpus. The pairs it links are different people in the same pose.
- **CLIP ViT-B/32, cosine ≥ 0.95.** Percolates at 0.96. The median pairwise cosine across
  the *whole corpus* is 0.83.
- **SSCD, similarity ≥ 0.5** — the threshold Somepalli et al. used to measure that 1.88%
  of Stable Diffusion generations are near-copies of training images. On ACNE04, 82.6% of
  real images have another real image above it, and the maximum similarity between two
  genuinely distinct real images is 0.976.

The common cause: every one of those thresholds was chosen on a heterogeneous corpus —
web images, ImageNet, LAION — where two arbitrary items share almost nothing. ACNE04 is
1,457 half-face photographs at one angle under similar lighting, so the *null* distribution
of similarity between unrelated items sits far to the right. An absolute threshold carried
across corpora measures a different thing on each.

This module makes that check routine. Given a corpus and an embedder it reports the null,
the threshold that holds a stated false-positive rate, and how the grouping behaves as the
threshold moves — so a threshold is chosen by measurement rather than inherited.

**None of this is specific to acne, faces, or this study.** Any dataset audit that groups
or flags by embedding similarity needs it, and medical imaging corpora are exactly where
it bites: they are homogeneous by construction.

## Per-item or per-pair: the choice is not cosmetic

The false-positive rate can be counted two ways and the right one depends on the question.

**Per item** — "what share of items have *any* false match?" — is right when the audit
asks a question of each item independently. Memorisation is the case: for every generated
image you ask whether *some* training image is a near-copy of it, so one false neighbour
is one false positive. Per-item is also the honest denominator, because per-pair rates
read reassuringly small on any large corpus: 5.3% of ACNE04 pairs exceed SSCD 0.5, which
sounds tolerable right up until you notice it means 82.6% of images have a false match.

**Per pair** is right when most items are *expected* to have a true match, because then
the per-item maximum is dominated by the true positive and measures nothing. Identity
grouping is that case: 84% of ACNE04 images genuinely belong to a subject with more than
one photograph, so "80.8% of items have a neighbour above 0.60" is the correct answer, not
an error rate. Per-pair works there only because same-subject pairs are a negligible share
of all pairs -- roughly 0.1% of the 979,300 pairs among embedded images -- so the
all-pairs distribution is, to a good approximation, the different-subject null.

Getting this backwards produces a threshold of 1.0 and a warning that the identity
threshold "does not transfer", which is how this distinction came to be written down.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from typing import Callable, Sequence

import numpy as np

from ..data.dedup import UnionFind

log = logging.getLogger(__name__)

Embedder = Callable[[Sequence[str]], np.ndarray]


@dataclass
class NullDistribution:
    """Similarity between pairs the audit considers unrelated."""

    n_items: int
    n_pairs: int
    median: float
    p90: float
    p99: float
    p999: float
    maximum: float

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ThresholdCalibration:
    null: NullDistribution
    threshold: float
    #: "per_item" or "per_pair"; see the module docstring on which applies when.
    mode: str
    target_fpr: float
    achieved_fpr: float
    #: What a threshold taken from the literature would have done here, if one was given.
    reference_threshold: float | None = None
    fpr_at_reference: float | None = None
    #: Largest connected component as a share of the corpus, per threshold.
    percolation: list[tuple[float, int, float]] = field(default_factory=list)
    #: The threshold that satisfies BOTH the error-rate target and the percolation cap.
    #: A false-positive rate alone is not sufficient for a grouping task: on ACNE04 a 1%
    #: per-pair rate permits 0.318, which merges 89.5% of the corpus into one cluster.
    #: Every false link is cheap, but transitive closure makes them collectively fatal.
    recommended_threshold: float | None = None
    max_component_fraction: float | None = None

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["null"] = self.null.to_dict()
        return payload

    @property
    def reference_is_usable(self) -> bool:
        """A reference threshold is usable if it holds roughly the target error rate."""
        if self.fpr_at_reference is None:
            return True
        return self.fpr_at_reference <= 2 * self.target_fpr


def _normalise(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)


def _largest_component(sim: np.ndarray, threshold: float) -> int:
    n = len(sim)
    uf = UnionFind(n)
    ii, jj = np.where(np.triu(sim >= threshold, k=1))
    for a, b in zip(ii.tolist(), jj.tolist()):
        uf.union(a, b)
    counts: dict[int, int] = {}
    for i in range(n):
        root = uf.find(i)
        counts[root] = counts.get(root, 0) + 1
    return max(counts.values()) if counts else 0


def calibrate(
    paths: Sequence[str],
    embedder: Embedder,
    target_fpr: float = 0.01,
    mode: str = "per_item",
    known_pairs: set[tuple[int, int]] | None = None,
    reference_threshold: float | None = None,
    percolation_grid: Sequence[float] | None = None,
    max_component_fraction: float | None = 0.05,
) -> ThresholdCalibration:
    """Choose a similarity threshold from `paths` at a stated false-positive rate.

    `known_pairs` are index pairs the audit already knows are genuinely related — exact
    duplicates, or two photographs of one subject. They are excluded from the null, since
    including a true positive in the null inflates the threshold and hides real matches.

    `reference_threshold` is the value a paper or library would have you use. It is not
    applied; it is reported, so the comparison between "what the literature says" and
    "what this corpus supports" appears in the record rather than in a footnote.

    `mode` selects how the rate is counted: "per_item" when each item is asked an
    independent question (memorisation, duplicate detection), "per_pair" when most items
    are expected to have a true match so the per-item maximum measures nothing (identity
    grouping). The module docstring explains why this is not a cosmetic choice.

    `max_component_fraction` adds the second constraint a *grouping* task needs. An error
    rate alone is not enough there, because grouping takes the transitive closure of the
    links: individually rare false links chain, and a threshold with a perfectly acceptable
    pairwise error rate can still merge most of the corpus into one cluster. On ACNE04 a 1%
    per-pair rate permits 0.318, which merges 89.5% of the images. Pass None to skip.
    """
    if mode not in ("per_item", "per_pair"):
        raise ValueError(f"mode must be 'per_item' or 'per_pair', not {mode!r}")
    emb = _normalise(embedder(list(paths)))
    n = len(emb)
    if n < 3:
        raise ValueError("need at least three items to estimate a null distribution")
    sim = emb @ emb.T

    best = sim - np.eye(n, dtype=np.float32) * 2.0
    upper = np.triu(np.ones((n, n), dtype=bool), k=1)
    for a, b in known_pairs or ():
        best[a, b] = -2.0
        best[b, a] = -2.0
        upper[min(a, b), max(a, b)] = False
    per_item_best = best.max(axis=1)
    null_values = sim[upper]

    def rate(threshold: float) -> float:
        if mode == "per_item":
            return float((per_item_best >= threshold).mean())
        return float((null_values >= threshold).mean())

    lo, hi = float(min(per_item_best.min(), null_values.min())), 1.0
    for _ in range(80):
        mid = (lo + hi) / 2
        if rate(mid) > target_fpr:
            lo = mid
        else:
            hi = mid

    null = NullDistribution(
        n_items=n,
        n_pairs=int(null_values.size),
        median=float(np.median(null_values)),
        p90=float(np.percentile(null_values, 90)),
        p99=float(np.percentile(null_values, 99)),
        p999=float(np.percentile(null_values, 99.9)),
        maximum=float(null_values.max()),
    )

    grid = percolation_grid
    if grid is None:
        grid = [float(v) for v in np.linspace(max(null.median, 0.0), 1.0, 9)]
    percolation = []
    for t in grid:
        largest = _largest_component(sim, t)
        percolation.append((float(t), int(largest), float(largest / n)))

    result = ThresholdCalibration(
        null=null,
        threshold=float(hi),
        mode=mode,
        target_fpr=target_fpr,
        achieved_fpr=rate(float(hi)),
        percolation=percolation,
    )
    if max_component_fraction is not None:
        # Search upward from the error-rate threshold for the first value that also keeps
        # the largest connected component under the cap.
        candidates = sorted({float(hi), *(t for t, _, _ in percolation),
                             *(float(v) for v in np.linspace(float(hi), 1.0, 40))})
        result.max_component_fraction = float(max_component_fraction)
        for t in candidates:
            if t < hi:
                continue
            if _largest_component(sim, t) / n <= max_component_fraction:
                result.recommended_threshold = float(t)
                break
        if result.recommended_threshold is None:
            log.warning(
                "no threshold satisfies both a %.1f%% %s error rate and a %.0f%% component "
                "cap; the embedder may not separate this corpus at all",
                100 * target_fpr, mode.replace("_", "-"), 100 * max_component_fraction,
            )
        elif result.recommended_threshold > hi:
            log.info(
                "error rate alone permits %.4f, but that merges more than %.0f%% of the "
                "corpus; recommending %.4f, where grouping is stable",
                hi, 100 * max_component_fraction, result.recommended_threshold,
            )

    if reference_threshold is not None:
        result.reference_threshold = float(reference_threshold)
        result.fpr_at_reference = rate(float(reference_threshold))
        if not result.reference_is_usable:
            log.warning(
                "the reference threshold %.3f gives a %.1f%% %s false-positive rate on this "
                "corpus against a %.1f%% target; it was calibrated somewhere else",
                reference_threshold,
                100 * result.fpr_at_reference,
                mode.replace("_", "-"),
                100 * target_fpr,
            )

    log.info(
        "calibrated threshold %.4f for a %.1f%% %s false-positive rate "
        "(null median %.3f, max %.3f)",
        result.threshold, 100 * target_fpr, mode.replace("_", "-"), null.median, null.maximum,
    )
    return result
