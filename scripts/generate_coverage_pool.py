#!/usr/bin/env python
"""Generate the coverage pool: balanced across severity and skin tone, from prompts alone.

This is the pool the actual question needs -- train a classifier on synthetic images only,
validate on real ones. Sized to the real training budget (948) so the comparison against
the real-only arm is budget-matched, and balanced across the grid rather than matched to
ACNE04's 35/43/12/9 prior, because that skew is the deficiency the synthetic set exists to
fix (docs/07_coverage_arm.md section 12.2).

Prompt style is the one that measured best, not the one that sounded best: **severity words
at full frame**. Prompting with explicit lesion counts and tight framing raised exact
agreement while destroying ordinal fidelity -- Spearman 0.010 against 0.176 -- because the
predictions collapsed onto the modal class. That was measured, not guessed, and it is why
the obvious-looking improvement is not used here.

    python scripts/generate_coverage_pool.py --n 960 --out data/synthetic/gemini_pool
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from fitymi.generate.gemini import GeminiConfig, generate_many

SEVERITY = {
    0: "a few scattered comedones and one or two small inflammatory papules, mild acne vulgaris",
    1: "numerous inflammatory papules and pustules across the cheek, moderate acne vulgaris",
    2: "dense inflammatory papules and pustules with several nodules, severe acne vulgaris",
    3: "confluent inflammatory nodules and cysts with scarring, very severe acne vulgaris",
}
TONES = [
    "very fair skin (Fitzpatrick I)", "fair skin (Fitzpatrick II)",
    "light olive skin (Fitzpatrick III)", "olive to light brown skin (Fitzpatrick IV)",
    "brown skin (Fitzpatrick V)", "deeply pigmented skin (Fitzpatrick VI)",
]
AGES = ["a teenage", "a young adult"]
SEXES = ["male", "female"]
VIEWS = [
    "photographed at a three-quarter angle from the front",
    "photographed in profile",
    "photographed frontally",
]
LIGHTING = ["even clinical lighting", "diffuse daylight", "soft window light"]


def build_prompts(total: int) -> list[tuple[str, str]]:
    """One prompt per (grade, tone, age, sex) cell, cycled over view and lighting.

    Balanced by construction rather than by sampling, so the pool's coverage is a property
    of the design and not of a random seed.
    """
    per_grade = total // len(SEVERITY)
    by_grade: dict[int, list[tuple[str, str]]] = {g: [] for g in SEVERITY}
    for grade, description in SEVERITY.items():
        for i in range(per_grade):
            tone = TONES[i % len(TONES)]
            age = AGES[(i // len(TONES)) % len(AGES)]
            sex = SEXES[(i // (len(TONES) * len(AGES))) % len(SEXES)]
            view = VIEWS[i % len(VIEWS)]
            light = LIGHTING[(i // len(VIEWS)) % len(LIGHTING)]
            by_grade[grade].append((
                f"g{grade}_i{i:04d}",
                f"A colour clinical dermatology photograph of the face of {age} {sex} person "
                f"with {tone}, showing {description}, {view} under {light}. Realistic medical "
                f"documentation photograph, neutral background, no text or watermark."
            ))

    # Interleave the grades rather than emitting them in blocks. A fifteen-hour run at a
    # rate-limited tier will sometimes be interrupted, and grade-major order means any
    # prefix is missing whole classes -- a pool stopped at 70% would contain no very-severe
    # images at all and be unusable. Interleaved, every prefix is a balanced pool.
    prompts = []
    for i in range(per_grade):
        for grade in SEVERITY:
            if i < len(by_grade[grade]):
                prompts.append(by_grade[grade][i])
    return prompts


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=960)
    ap.add_argument("--out", default="data/synthetic/gemini_pool")
    ap.add_argument("--project", default="watchmen-4d5b1")
    ap.add_argument("--concurrency", type=int, default=2)
    args = ap.parse_args()

    prompts = build_prompts(args.n)
    out = Path(args.out)
    started = time.time()
    manifest = out / "manifest.jsonl"
    out.mkdir(parents=True, exist_ok=True)

    done = failed = 0
    with open(manifest, "a") as fh:
        for record in generate_many(
            prompts, out,
            GeminiConfig(project=args.project, concurrency=args.concurrency),
        ):
            fh.write(json.dumps(record) + "\n")
            fh.flush()
            done += 1
            failed += record["path"] is None
            if done % 25 == 0:
                rate = done / max(time.time() - started, 1) * 3600
                remaining = (len(prompts) - done) / max(rate, 1e-9)
                print(f"{done}/{len(prompts)}  {failed} failed  "
                      f"{rate:.0f}/h  ~{remaining:.1f}h left", flush=True)

    print(f"done: {done - failed}/{len(prompts)} images in "
          f"{(time.time() - started) / 3600:.2f}h")


if __name__ == "__main__":
    main()
