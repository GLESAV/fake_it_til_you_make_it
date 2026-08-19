"""Guards against the repository not containing the code that was written.

Every other test in this suite imports `fitymi` from the working tree, so all of
them pass whether or not the files are actually committed. That is exactly how the
entire `src/fitymi/data/` package -- the layer everything else sits on -- got
excluded by a bare `data/` gitignore pattern and shipped as a repository that fails
at the first import.

These tests read git's own view of the tree rather than the filesystem's.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent

#: Directories whose contents are source, not artefacts. Everything here must be
#: committed.
SOURCE_DIRS = ("src", "tests", "configs", "scripts")
SOURCE_SUFFIXES = {".py", ".yaml", ".yml", ".sh", ".tex", ".bib"}


def _git(*args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(REPO), *args], capture_output=True, text=True, check=False
    )
    if result.returncode != 0:
        pytest.skip(f"git unavailable or not a repository: {result.stderr.strip()}")
    return result.stdout


def _is_source(path: Path) -> bool:
    if "__pycache__" in path.parts or ".egg-info" in str(path):
        return False
    return path.suffix in SOURCE_SUFFIXES


def _source_files_on_disk() -> set[str]:
    found: set[str] = set()
    for directory in SOURCE_DIRS:
        root = REPO / directory
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if path.is_file() and _is_source(path):
                found.add(str(path.relative_to(REPO)))
    return found


def test_every_source_file_is_tracked():
    tracked = set(_git("ls-files").splitlines())
    untracked = sorted(_source_files_on_disk() - tracked)
    assert not untracked, (
        "these source files exist on disk but are in no commit, so a fresh clone "
        f"will not have them: {untracked}"
    )


def test_no_source_file_is_gitignored():
    """The specific failure: a bare directory pattern matching a source package."""
    candidates = sorted(_source_files_on_disk())
    if not candidates:
        pytest.skip("no source files found")
    result = subprocess.run(
        ["git", "-C", str(REPO), "check-ignore", "--stdin"],
        input="\n".join(candidates),
        capture_output=True,
        text=True,
        check=False,
    )
    ignored = sorted(line for line in result.stdout.splitlines() if line.strip())
    assert not ignored, f"source files matched by a gitignore rule: {ignored}"


def test_every_package_directory_has_an_init():
    """A missing __init__.py is the other way a package silently fails to import."""
    missing = [
        str(d.relative_to(REPO))
        for d in (REPO / "src" / "fitymi").rglob("*")
        if d.is_dir() and d.name != "__pycache__" and not (d / "__init__.py").exists()
    ]
    assert not missing, f"package directories without __init__.py: {missing}"


def test_gitignore_anchors_its_artefact_directories():
    """`data/` matches at every level; `/data/` matches only the repo root."""
    lines = [
        line.strip()
        for line in (REPO / ".gitignore").read_text().splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    unanchored = [
        line
        for line in lines
        if line.endswith("/")
        and not line.startswith("/")
        and line not in {"__pycache__/", "*.egg-info/", ".pytest_cache/"}
    ]
    assert not unanchored, (
        "these directory patterns are unanchored and will match at every depth, "
        f"which is how src/fitymi/data/ was lost: {unanchored}"
    )
