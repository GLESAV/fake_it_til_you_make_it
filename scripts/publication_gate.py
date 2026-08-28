#!/usr/bin/env python
"""The mechanical half of docs/08_publication_practices.md §6.

Fails loudly on the things a machine can check before a manuscript leaves the repository.
Everything it cannot check is in the human table in that document; passing this script is
necessary, not sufficient.

    python scripts/publication_gate.py                 # all manuscripts
    python scripts/publication_gate.py paper/audit.tex # just one
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BIB = ROOT / "paper" / "refs.bib"

# Markers left in refs.bib notes by the verification passes (docs/VERIFY.md).
UNVERIFIED = re.compile(r"NOT INDEPENDENTLY VERIFIED|UNVERIFIED", re.I)
VERIFIED = re.compile(r"\bVERIFIED\s+\d{4}-\d{2}-\d{2}")
PLACEHOLDER = re.compile(r"\\(RESULT|TODO)\{")
CITE = re.compile(r"\\(?:cite|citep|citet|citealp|citealt|citeauthor|citeyear)\*?"
                  r"(?:\[[^\]]*\])*\{([^}]*)\}")

failures: list[str] = []
warnings: list[str] = []


def rel(path: Path) -> str:
    """Display path, tolerant of targets given relative to the CWD or absolute."""
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def fail(gate: str, msg: str) -> None:
    failures.append(f"[{gate}] {msg}")


def warn(gate: str, msg: str) -> None:
    warnings.append(f"[{gate}] {msg}")


def parse_bib(text: str) -> dict[str, str]:
    """Map citation key -> raw entry body. Brace-counting, not a real bib parser."""
    entries: dict[str, str] = {}
    for m in re.finditer(r"@\w+\s*\{\s*([^,\s]+)\s*,", text):
        key = m.group(1)
        depth, i = 1, m.end()
        while i < len(text) and depth:
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
            i += 1
        entries[key] = text[m.end():i]
    return entries


def check_placeholders(tex: Path, body: str) -> None:
    """R1/§6 gate 1: a manuscript with placeholders is not finished."""
    hits = [(n, ln.strip()) for n, ln in enumerate(body.splitlines(), 1)
            if PLACEHOLDER.search(ln) and not ln.lstrip().startswith("%")]
    for n, ln in hits:
        fail("placeholder", f"{rel(tex)}:{n}: {ln[:90]}")


def check_citations(tex: Path, body: str, entries: dict[str, str]) -> None:
    """R1: every cited key resolves, and every cited entry is verified."""
    cited: dict[str, int] = {}
    for n, ln in enumerate(body.splitlines(), 1):
        if ln.lstrip().startswith("%"):
            continue
        for m in CITE.finditer(ln):
            for key in (k.strip() for k in m.group(1).split(",")):
                if key:
                    cited.setdefault(key, n)

    for key, line in sorted(cited.items()):
        where = f"{rel(tex)}:{line}"
        if key not in entries:
            fail("dangling-cite", f"{where}: \\cite{{{key}}} has no refs.bib entry")
            continue
        note = entries[key]
        if UNVERIFIED.search(note):
            fail("unverified-cite", f"{where}: {key} is cited but marked NOT VERIFIED")
        elif not VERIFIED.search(note):
            warn("unverified-cite", f"{where}: {key} carries no 'VERIFIED <date>' note")


def check_licence() -> None:
    """R14: the README promises a LICENSE; it must exist and name a holder."""
    lic = ROOT / "LICENSE"
    if not lic.exists():
        fail("licence", "README promises LICENSE (MIT) but no such file exists")
        return
    text = lic.read_text()
    # Only the copyright line, never the body: the MIT warranty clause legitimately
    # contains the words "COPYRIGHT HOLDERS", which the first version of this check
    # flagged as a placeholder in a licence that had already been filled in.
    holder = re.search(r"^Copyright \(c\).*$", text, re.M)
    if not holder:
        fail("licence", "LICENSE has no 'Copyright (c) <year> <holder>' line")
    elif re.search(r"<.+>|COPYRIGHT HOLDER|TODO|FIXME", holder.group(0)):
        fail("licence", f"LICENSE copyright line is a placeholder: {holder.group(0)}")


def check_readme_staleness() -> None:
    """R1: the README's verification-status claim must not contradict VERIFY.md."""
    readme = (ROOT / "README.md").read_text()
    verify = (ROOT / "docs" / "VERIFY.md").read_text()
    claims_all_s = re.search(r"all of them are currently\s+`?\[S\]`?", readme)
    n_verified = len(re.findall(r"\*\*\[V\]\*\*", verify))
    if claims_all_s and n_verified:
        fail("stale-readme",
             f"README says every claim is [S], but VERIFY.md records {n_verified} at [V]")
    # The README quotes the count as a number, and a number goes stale silently. It went
    # stale within one verification pass of first being written.
    quoted = re.search(r"promoted\s+\**(\d+)\**\s+sources", readme)
    if quoted and int(quoted.group(1)) != n_verified:
        fail("stale-readme",
             f"README claims {quoted.group(1)} sources at [V]; VERIFY.md records "
             f"{n_verified}")


def check_bib_health(entries: dict[str, str]) -> None:
    """Informational: how much of the bibliography is still unverified overall."""
    unver = sorted(k for k, v in entries.items() if UNVERIFIED.search(v))
    if unver:
        warn("bib-health",
             f"{len(unver)}/{len(entries)} entries still unverified (uncited ones do not "
             f"block): {', '.join(unver)}")


def main() -> int:
    targets = [Path(a).resolve() for a in sys.argv[1:]] or sorted((ROOT / "paper").glob("*.tex"))
    if not BIB.exists():
        print(f"refs.bib not found at {BIB}", file=sys.stderr)
        return 2
    entries = parse_bib(BIB.read_text())

    for tex in targets:
        body = tex.read_text()
        check_placeholders(tex, body)
        check_citations(tex, body, entries)

    check_licence()
    check_readme_staleness()
    check_bib_health(entries)

    print(f"publication gate: {len(targets)} manuscript(s), {len(entries)} bib entries\n")
    for w in warnings:
        print(f"  warn  {w}")
    if warnings:
        print()
    for f in failures:
        print(f"  FAIL  {f}")

    if failures:
        print(f"\n{len(failures)} blocking failure(s). "
              f"See docs/08_publication_practices.md §6.")
        return 1
    print("Mechanical gates pass. The human gates in §6 still apply — and they are the "
          "ones that matter.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
