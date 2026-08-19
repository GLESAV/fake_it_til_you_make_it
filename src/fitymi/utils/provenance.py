"""Run provenance.

Each run writes a record containing the git SHA, the config hash, the dataset hash
and the package versions. Without this, a mixing curve assembled from 250 runs is
not auditable, and the whole point of the protocol is auditability.
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any


def _git(*args: str) -> str | None:
    try:
        out = subprocess.run(
            ["git", *args], capture_output=True, text=True, timeout=10, check=False
        )
        return out.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def git_sha() -> str | None:
    return _git("rev-parse", "HEAD")


def git_dirty() -> bool:
    status = _git("status", "--porcelain")
    return bool(status)


def hash_obj(obj: Any) -> str:
    """Stable hash of a JSON-serialisable object."""
    payload = json.dumps(obj, sort_keys=True, default=str).encode()
    return hashlib.sha256(payload).hexdigest()[:16]


def hash_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def package_versions() -> dict[str, str]:
    versions: dict[str, str] = {"python": platform.python_version()}
    for name in ("numpy", "torch", "torchvision", "sklearn", "diffusers"):
        try:
            mod = __import__(name)
            versions[name] = getattr(mod, "__version__", "unknown")
        except ImportError:
            continue
    return versions


@dataclass
class RunProvenance:
    run_id: str
    config_hash: str
    dataset_hash: str
    git_sha: str | None = field(default_factory=git_sha)
    git_dirty: bool = field(default_factory=git_dirty)
    packages: dict[str, str] = field(default_factory=package_versions)
    platform: str = field(default_factory=platform.platform)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
