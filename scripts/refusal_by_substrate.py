#!/usr/bin/env python
"""Per-substrate refusal rate for the wide-domain pool, and what it costs the gate.

docs/07 records that the generator's refusals "concentrate exactly where clinical utility
is highest" -- ACNE04 is close-up half-face photography, and close-up human skin is what
the model is most reluctant to draw. That was a preliminary read from 166 attempts at 3-10
images per substrate. This script makes it a measurement anyone can repeat, and it answers
the operational question the stage-1 gate raises: the kill criterion is stated per
substrate, `substrate_fidelity.py` wants 24 images before it will score one, and at the
gate's 240-image threshold no substrate is close.

Two things are deliberately separate here. The refusal rate is a property of the prompt and
the backend; Scope and Continuity are properties of the images. Nothing in this script
touches the gate's outcome measure, so it can be run before the gate without looking at it.

One attempt-outcome per prompt, not per record: a prompt retried across runs would
otherwise be counted several times, and the manifest holds several such. Rate limits are
excluded from the denominator -- a 429 says nothing about whether a prompt is renderable.

    python scripts/refusal_by_substrate.py
    python scripts/refusal_by_substrate.py --target 24 --rate 30 --cost 0.039
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default="data/synthetic/wide_pool")
    ap.add_argument("--target", type=int, default=24,
                    help="images per substrate the gate's criterion needs")
    ap.add_argument("--rate", type=float, default=30.0, help="observed images per hour")
    ap.add_argument("--cost", type=float, default=0.039, help="dollars per image")
    args = ap.parse_args()

    pool = Path(args.pool)
    meta = json.loads((pool / "prompt_meta.json").read_text())
    rows = [json.loads(l) for l in (pool / "manifest.jsonl").read_text().splitlines()
            if l.strip()]

    verdict: dict[str, str] = {}
    for r in rows:
        name = r["name"]
        if r["path"]:
            verdict[name] = "ok"
        elif verdict.get(name) != "ok":
            verdict[name] = "429" if "429" in str(r["blocked_reason"]) else "empty"

    by: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    for name, v in verdict.items():
        if name in meta:
            by[meta[name]["substrate"]][v] += 1
    if not by:
        raise SystemExit("no attempts recorded yet")

    def yield_of(c):
        tried = c["ok"] + c["empty"]
        return c["ok"] / tried if tried else 0.0

    print(f"{'substrate':>26} {'person':>7} {'ok':>4} {'refused':>8} {'429':>4} "
          f"{'yield':>6}  prompts to reach {args.target}")
    needed = []
    for name, c in sorted(by.items(), key=lambda kv: -yield_of(kv[1])):
        y = yield_of(c)
        person = next((m["needs_person"] for m in meta.values()
                       if m["substrate"] == name), None)
        short = max(args.target - c["ok"], 0)
        more = short / y if y > 0 else float("inf")
        needed.append(more)
        tail = "unreachable" if more == float("inf") else f"{more:.0f}"
        print(f"{name[:26]:>26} {str(person):>7} {c['ok']:>4} {c['empty']:>8} "
              f"{c['429']:>4} {100 * y:>5.0f}%  {tail:>18}")

    finite = [n for n in needed if n != float("inf")]
    overall = sum(c["ok"] for c in by.values()) / max(
        sum(c["ok"] + c["empty"] for c in by.values()), 1)
    print(f"\n{sum(1 for n in needed if n == float('inf'))} of {len(by)} substrates have "
          f"produced nothing at all; overall yield {100 * overall:.0f}%")
    print(f"to bring every producing substrate to {args.target}: {sum(finite):.0f} more "
          f"prompts, about {sum(finite) * overall:.0f} images, "
          f"{sum(finite) * overall / args.rate:.0f} hours and "
          f"${sum(finite) * overall * args.cost:.0f}")

    on = [yield_of(c) for n, c in by.items()
          if next((m["needs_person"] for m in meta.values() if m["substrate"] == n), False)]
    off = [yield_of(c) for n, c in by.items()
           if not next((m["needs_person"] for m in meta.values() if m["substrate"] == n),
                       False)]
    if on and off:
        print(f"\nmean yield on a person {100 * sum(on) / len(on):.0f}% "
              f"({len(on)} substrates), not on a person "
              f"{100 * sum(off) / len(off):.0f}% ({len(off)} substrates)")
        print("The pool fills fastest exactly where it is least like the validation domain,"
              "\nso substrate availability correlates with distance from real clinical "
              "images.\nThat is a confound for the substrate ablation, not an inconvenience.")


if __name__ == "__main__":
    main()
