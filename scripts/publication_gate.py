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
# "NOT VERIFIED" as well as "NOT INDEPENDENTLY VERIFIED": moroianu2025 carried a
# "TITLE AND AUTHOR LIST TRUNCATED AND NOT VERIFIED" marker that the first version
# of this pattern did not match, so the entry passed as a warning. Its title turned
# out to be invented and its author list truncated from eleven names to three.
UNVERIFIED = re.compile(r"NOT (INDEPENDENTLY )?VERIFIED|UNVERIFIED", re.I)
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


def check_crossrefs(tex: Path, body: str) -> None:
    """Every \\ref resolves to a \\label in the same file.

    A dangling cross-reference renders as "??" and does not stop the build, so it survives
    a clean compile and reaches a reader. One did: a section reference added while writing
    up protocol 8.6 pointed at a label this manuscript does not define.
    """
    labels = set(re.findall(r"\\label\{([^}]*)\}", body))
    for n, line in enumerate(body.splitlines(), 1):
        if line.lstrip().startswith("%"):
            continue
        for m in re.finditer(r"\\(?:page)?ref\{([^}]*)\}", line):
            if m.group(1) not in labels:
                fail("dangling-ref",
                     f"{rel(tex)}:{n}: \\ref{{{m.group(1)}}} has no matching \\label "
                     f"-- it renders as ??")


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
    # Table rows only. VERIFY.md also marks individual *claims* [V] inside prose bullets,
    # and counting those made the README report 27 verified sources when there were 21.
    n_verified = sum(1 for line in verify.splitlines()
                     if line.startswith("|") and "**[V]**" in line)
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


def check_verify_agrees_with_bib(entries: dict[str, str]) -> None:
    """R1: a source VERIFY.md calls verified must say so in the entry a manuscript cites.

    The two files had drifted: `fan2024` sat at [V] in VERIFY.md from the first pass while
    its bib entry carried no note, so one source read as verified in one file and
    unverified in the other. Only rows naming a bib key in backticks are checkable, and the
    earlier rows name papers in prose -- so adding the key to a row is what puts that row
    under this check. Rows 1-13 predate the convention and are still unchecked here.
    """
    verify = (ROOT / "docs" / "VERIFY.md").read_text()
    for line in verify.splitlines():
        if not line.startswith("|") or "**[V]**" not in line:
            continue
        for key in re.findall(r"`([A-Za-z][A-Za-z0-9_]*\d{4}[a-z]?)`", line):
            note = entries.get(key)
            if note is None:
                fail("verify-drift", f"VERIFY.md marks {key} [V] but refs.bib has no entry")
            elif not VERIFIED.search(note):
                fail("verify-drift",
                     f"VERIFY.md marks {key} [V] but its refs.bib entry carries no "
                     f"'VERIFIED <date>' note")


def check_notes_do_not_print(entries: dict[str, str]) -> None:
    """R1: verification provenance belongs in `annote`, which plainnat does not print.

    It lived in `note`, which plainnat does print, so every VERIFIED marker was rendering
    into the manuscript's bibliography -- two pages of it by the end of the third
    verification pass. The notes have to stay in refs.bib because this script reads them;
    they must not be in a submitted PDF.
    """
    for key, body in sorted(entries.items()):
        for m in re.finditer(r"^\s*note\s*=\s*\{", body, re.M):
            tail = body[m.end():]
            depth, i = 1, 0
            while i < len(tail) and depth:
                depth += (tail[i] == "{") - (tail[i] == "}")
                i += 1
            value = tail[:i]
            if VERIFIED.search(value) or UNVERIFIED.search(value):
                fail("printing-note",
                     f"{key}: verification provenance is in `note`, which plainnat prints. "
                     f"Move it to `annote`.")


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
        check_crossrefs(tex, body)
        check_citations(tex, body, entries)

    check_licence()
    check_readme_staleness()
    check_verify_agrees_with_bib(entries)
    check_notes_do_not_print(entries)
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
