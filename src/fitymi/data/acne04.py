"""ACNE04 loader.

ACNE04 (Wu et al., ICCV 2019) is the benchmark: 1,457 half-face images with Hayashi
severity grades and 18,983 lesion boxes. It is academic-use only and is not
redistributed by this repository. Fetch it from https://github.com/xpwu95/LDL and
point `--root` at the extracted directory.

The distributed archive contains more files than labelled images, so the loader is
strict about the label index and warns when it finds unreferenced files: those are
exactly the near-duplicates that `dedup.py` needs to see.
"""

from __future__ import annotations

import logging
from pathlib import Path

from .records import NUM_CLASSES, Corpus, Record, Source

log = logging.getLogger(__name__)

#: The label files shipped with ACNE04, in the layout of the LDL release.
DEFAULT_LABEL_FILES = ("NNEW_trainval_0.txt", "NNEW_test_0.txt")
DEFAULT_IMAGE_DIR = "JPEGImages"


class Acne04NotFoundError(FileNotFoundError):
    pass


def _parse_label_line(line: str) -> tuple[str, int] | None:
    """Lines are ``<filename> <severity> <lesion_count>``; tolerate extra columns."""
    parts = line.split()
    if len(parts) < 2:
        return None
    filename = parts[0]
    try:
        severity = int(parts[1])
    except ValueError:
        return None
    if not 0 <= severity < NUM_CLASSES:
        return None
    return filename, severity


def load_acne04(
    root: str | Path,
    label_files: tuple[str, ...] = DEFAULT_LABEL_FILES,
    image_dir: str = DEFAULT_IMAGE_DIR,
) -> Corpus:
    """Load ACNE04 into a `Corpus`, ignoring the original train/test partition.

    The shipped partition is discarded on purpose: protocol §3.2 requires that
    splitting happen *after* deduplication, on our own seed, so that the sealed test
    set is defined by us and reproducible from the committed split seed.
    """
    root = Path(root)
    if not root.exists():
        raise Acne04NotFoundError(
            f"{root} does not exist. ACNE04 is not redistributed here; obtain it from "
            "https://github.com/xpwu95/LDL (academic use only) and extract it."
        )

    images = root / image_dir
    if not images.exists():
        candidates = [p for p in root.rglob(image_dir) if p.is_dir()]
        if not candidates:
            raise Acne04NotFoundError(f"no {image_dir}/ directory under {root}")
        images = candidates[0]

    labels: dict[str, int] = {}
    found_any = False
    for name in label_files:
        matches = list(root.rglob(name))
        if not matches:
            log.warning("label file %s not found under %s", name, root)
            continue
        found_any = True
        for line in matches[0].read_text().splitlines():
            parsed = _parse_label_line(line.strip())
            if parsed is None:
                continue
            filename, severity = parsed
            labels[Path(filename).name] = severity

    if not found_any or not labels:
        raise Acne04NotFoundError(
            f"no parseable label files ({', '.join(label_files)}) under {root}"
        )

    records: list[Record] = []
    missing: list[str] = []
    for filename, severity in sorted(labels.items()):
        path = images / filename
        if not path.exists():
            missing.append(filename)
            continue
        records.append(Record(path=str(path), label=severity, source=Source.REAL))

    on_disk = {p.name for p in images.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}}
    unreferenced = on_disk - set(labels)
    if unreferenced:
        log.warning(
            "%d image files on disk are not referenced by any label file; these are "
            "prime near-duplicate candidates and are excluded from the corpus",
            len(unreferenced),
        )
    if missing:
        log.warning("%d labelled images are missing from disk", len(missing))

    corpus = Corpus(records)
    log.info("loaded ACNE04: %s", corpus)
    return corpus
