#!/usr/bin/env python
"""Generate the wide-domain pool: acne across every substrate a generator can imagine.

Every prompt in the first pool was a human face, and the compression result (docs/07
section 12.9) says that pool renders four clinical grades as ~1.1 grades of real variation,
centred on moderate. A face is a strong prior. Asked for confluent nodulocystic disease, a
model that has seen millions of portraits has every reason to return a person who still
looks like a person.

This pool removes that constraint. Acne is asked for on faces, on isolated skin patches, on
surgical specimens, in Petri dishes, on silicone teaching models, in textbook plates, on
dermatoscope fields -- the full imagined domain of "skin with this disease on it", most of
which no real dermatology dataset contains and no camera could capture. The hypothesis is
that severity is easier to render at the extremes when nothing has to remain a plausible
photograph of a plausible person.

Two hypotheses, and they point opposite ways:

- **Breadth helps.** Freed from the face prior, the severe end becomes renderable, the
  Scope statistic rises, and the classifier sees genuine variation rather than prototypical
  moderate acne.
- **Breadth hurts.** Validation is on real half-face clinical photographs. Petri dishes are
  further from that domain than faces are, so the transfer gap widens and everything gained
  in fidelity is lost in domain shift.

Both are plausible and the experiment distinguishes them, which is the point of running it.
Substrate is recorded per image so the pool can be sliced afterwards and each substrate
scored on its own.

## The one design rule that matters

**Substrate is randomised independently of severity.** If Petri dishes skewed severe and
faces skewed mild, the pool would carry exactly the provenance confound this project spent
an audit documenting in ACNE04 -- where 87.1% of mild images but 16.3% of very-severe ones
came from one camera. A classifier would then learn substrate and score well for the wrong
reason. Severity and tone are balanced by construction; everything else is drawn from an
independent stream.

    python scripts/generate_wide_domain.py --n 6000 --out data/synthetic/wide_pool
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

from fitymi.generate.gemini import GeminiConfig, generate_many

#: Hayashi-style inflammatory severity, described rather than named, so the model has
#: something to render instead of a word to associate.
SEVERITY = {
    0: "a few scattered comedones and one or two small inflammatory papules",
    1: "numerous inflammatory papules and pustules",
    2: "dense inflammatory papules and pustules with several deep nodules",
    3: "confluent inflammatory nodules and cysts, widespread and severe, with scarring",
}

TONES = [
    "very fair, pale skin (Fitzpatrick I)", "fair skin (Fitzpatrick II)",
    "light olive skin (Fitzpatrick III)", "olive to light brown skin (Fitzpatrick IV)",
    "brown skin (Fitzpatrick V)", "deeply pigmented dark brown skin (Fitzpatrick VI)",
]

#: (template, needs_person). The wide domain: clinical photography at one end, laboratory
#: and didactic depictions at the other. `{s}` is severity, `{t}` tone, `{p}` the person
#: phrase where one applies.
#: Short label per substrate, in the same order, for grouping the ablation afterwards.
SUBSTRATE_LABELS = [
    "face (clinical)", "cheek (clinical)", "face frontal", "forehead", "jawline/chin",
    "back/shoulders", "chest/trunk", "cheek macro",
    "dermatoscope", "macro + scale bar", "flatbed scan",
    "isolated skin swatch", "excised specimen", "pinned section",
    "Petri dish", "culture plate", "bioengineered construct", "specimen jar",
    "silicone model", "wax moulage", "3D render", "textbook plate",
    "anatomical diagram", "prosthetic limb",
]

SUBSTRATES: list[tuple[str, bool]] = [
    # --- human, clinical: the domain validation actually lives in ---
    ("A colour clinical dermatology photograph of the face of {p} with {t}, showing {s}", True),
    ("A clinical photograph of the left cheek of {p} with {t}, showing {s}", True),
    ("A frontal clinical portrait of {p} with {t}, showing {s} across the face", True),
    ("A close-up clinical photograph of the forehead of {p} with {t}, showing {s}", True),
    ("A clinical photograph of the jawline and chin of {p} with {t}, showing {s}", True),
    ("A clinical photograph of the upper back and shoulders of {p} with {t}, showing {s}", True),
    ("A clinical photograph of the chest and upper trunk of {p} with {t}, showing {s}", True),
    ("A macro photograph of a small area of cheek skin of {p} with {t}, showing {s}", True),
    # --- human, non-standard capture ---
    ("A dermatoscopic image, circular field of view, of {t} affected by {s}", False),
    ("A high-magnification macro photograph of {t} with {s}, a scale bar in frame", False),
    ("A flatbed-scanner-style flat image of an area of {t} showing {s}", False),
    # --- skin as material, off the body ---
    ("A rectangular swatch of isolated {t} on a neutral seamless background, showing {s}", False),
    ("An excised specimen of {t} laid on a blue surgical drape, showing {s}", False),
    ("A section of {t} pinned flat on a dissection board, showing {s}", False),
    # --- laboratory framings: skin with no person anywhere near it ---
    ("A circular sample of {t} in a glass Petri dish on a laboratory bench, showing {s}", False),
    ("A skin explant of {t} in a tissue-culture plate under laboratory lighting, showing {s}", False),
    ("A bioengineered skin construct with {t} pigmentation in a culture well, showing {s}", False),
    ("A laboratory specimen jar containing {t} tissue, showing {s}", False),
    # --- models and depictions: skin that was never alive ---
    ("A silicone dermatology training model with {t} colouring, showing {s}", False),
    ("A wax moulage teaching model of {t} from a medical museum, showing {s}", False),
    ("A 3D-rendered synthetic skin surface with {t} colouring, showing {s}", False),
    ("A medical textbook illustration plate of {t} affected by {s}", False),
    ("An anatomical diagram of a patch of {t} showing {s}", False),
    ("A prosthetic limb covered in synthetic {t}, showing {s}", False),
]

AGES = ["a teenage", "a young adult", "an adult", "a middle-aged"]
SEXES = ["male", "female"]
LIGHTING = [
    "under even clinical lighting", "in diffuse daylight", "under a ring light",
    "under harsh directional light", "under overhead fluorescent light",
    "in soft window light", "under a photographic softbox",
]
CAPTURE = [
    "Shot on a DSLR for medical documentation.", "Shot on a smartphone camera.",
    "Studio clinical photography.", "Macro photography with shallow depth of field.",
    "Copy-stand overhead photograph.", "Documentary record photograph.",
]
BACKDROP = [
    "Neutral grey background.", "Plain white background.", "Blue clinical drape background.",
    "Dark background.", "Laboratory bench background.", "Plain background.",
]


def build_prompts(total: int, seed: int = 0) -> list[tuple[str, str, dict]]:
    """Balanced on severity and tone; every other axis drawn independently.

    Returns (name, prompt, metadata). The metadata is what makes the substrate ablation
    possible afterwards, so it is written to the manifest rather than reconstructed from
    the filename later.
    """
    rng = random.Random(seed)
    cells = [(g, t) for g in SEVERITY for t in range(len(TONES))]
    per_cell = max(1, total // len(cells))

    rows: list[tuple[int, int, int, str, dict]] = []
    for g, t_idx in cells:
        for i in range(per_cell):
            # Substrate cycles within a cell and is offset by the cell index, so every
            # severity meets every substrate roughly equally often. Drawing it at random
            # would leave the balance to luck at this sample size.
            s_idx = (i + g * 7 + t_idx * 3) % len(SUBSTRATES)
            template, needs_person = SUBSTRATES[s_idx]
            person = f"{rng.choice(AGES)} {rng.choice(SEXES)} person"
            body = template.format(s=SEVERITY[g], t=TONES[t_idx], p=person)
            prompt = (
                f"{body}, {rng.choice(LIGHTING)}. {rng.choice(CAPTURE)} "
                f"{rng.choice(BACKDROP)} Photorealistic, no text, no watermark, no labels."
            )
            meta = {
                "grade": g, "tone_index": t_idx, "tone": TONES[t_idx],
                "substrate_index": s_idx, "substrate": SUBSTRATE_LABELS[s_idx],
                "needs_person": needs_person,
            }
            rows.append((g, t_idx, s_idx, prompt, meta))

    rng.shuffle(rows)
    # Interleave by grade so any prefix of an interrupted run is still balanced -- the
    # first pool was stopped mid-run and came back 128 mild against 46 of everything else.
    by_grade: dict[int, list] = {g: [] for g in SEVERITY}
    for row in rows:
        by_grade[row[0]].append(row)
    out: list[tuple[str, str, dict]] = []
    counters = {g: 0 for g in SEVERITY}
    for i in range(max(len(v) for v in by_grade.values())):
        for g in SEVERITY:
            if i < len(by_grade[g]):
                _, t_idx, s_idx, prompt, meta = by_grade[g][i]
                name = f"g{g}_s{s_idx:02d}_i{counters[g]:05d}"
                counters[g] += 1
                out.append((name, prompt, meta))
    return out


#: A prompt the model has refused this many times, every time with an empty response and
#: never with a rate limit, is not going to be granted on the next run either. The backend
#: module documents why -- an empty response is a content decision, not a transient fault,
#: and its rate is a property of the prompt (83% for close-up cheek macros, 0% for wax
#: moulages) rather than of the moment. Resuming re-attempts every prompt with no PNG on
#: disk, so without this the run spends its first half-hour of a rate-limited quota
#: re-earning refusals it has already recorded, and those wasted requests are themselves
#: what provokes the next 429.
REFUSAL_ATTEMPTS_BEFORE_SKIP = 2


def persistently_refused(manifest: Path) -> set[str]:
    """Names whose every recorded attempt came back empty, never rate-limited.

    Rate-limited prompts are deliberately NOT skipped: a 429 says nothing about the
    prompt. The refusals stay in the manifest either way, so the per-substrate refusal
    rate that scripts/substrate_fidelity.py reads is unaffected by this.
    """
    if not manifest.exists():
        return set()
    history: dict[str, list[str | None]] = {}
    for line in manifest.read_text().splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if record.get("path"):
            history.pop(record["name"], None)  # it succeeded once; never skip it
            continue
        history.setdefault(record["name"], []).append(record.get("blocked_reason"))
    return {
        name for name, reasons in history.items()
        if len(reasons) >= REFUSAL_ATTEMPTS_BEFORE_SKIP
        and all("no candidates" in (r or "") or "no image part" in (r or "")
                for r in reasons)
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=6000)
    ap.add_argument("--out", default="data/synthetic/wide_pool")
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    assert len(SUBSTRATE_LABELS) == len(SUBSTRATES), "one label per substrate"
    prompts = build_prompts(args.n)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "prompt_meta.json").write_text(
        json.dumps({name: meta for name, _, meta in prompts}, indent=1))

    if args.dry_run:
        print(f"{len(prompts)} prompts across {len(SUBSTRATES)} substrates, "
              f"{len(SEVERITY)} grades, {len(TONES)} tones\n")
        for name, prompt, _ in prompts[:6]:
            print(f"  {name}\n    {prompt}\n")
        from collections import Counter
        print("grade counts   :", dict(sorted(Counter(n[:2] for n, _, _ in prompts).items())))
        sub = Counter(m["substrate_index"] for _, _, m in prompts)
        print(f"substrate spread: min {min(sub.values())}, max {max(sub.values())}, "
              f"{len(sub)} distinct")
        return

    config = GeminiConfig(project="watchmen-4d5b1", concurrency=args.concurrency)
    manifest = out / "manifest.jsonl"
    refused = persistently_refused(manifest)
    if refused:
        print(f"skipping {len(refused)} prompt(s) the model has refused "
              f"{REFUSAL_ATTEMPTS_BEFORE_SKIP}+ times with no rate limit in between",
              flush=True)
    queue = [(n, p) for n, p, _ in prompts if n not in refused]
    started, done, failed = time.time(), 0, 0
    with manifest.open("a") as handle:
        for record in generate_many(queue, out, config):
            record["meta"] = {m_name: m for m_name, _, m in prompts if m_name == record["name"]}
            handle.write(json.dumps(record) + "\n")
            handle.flush()
            done += 1
            failed += record["path"] is None
            if done % 25 == 0:
                rate = done / max(time.time() - started, 1) * 3600
                print(f"{done} done, {failed} failed, {rate:.0f}/hour", flush=True)


if __name__ == "__main__":
    main()
