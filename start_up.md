# START HERE — resume notes

> **Delete this file once the reboot is confirmed good and you've read it.**
> It is a handover note, not project documentation. `STATE.md` holds the durable state;
> everything below is about getting back to where we were. Ask before deleting — if the
> machine was reset again, or someone else picks this up, these notes are the only thing
> that explains why two background jobs are missing.

## One-line status

Two manuscripts compile and are pushed; the classifier question is answered and mostly
negative; **one open experiment** — the wide-domain pool — is 92 images into a 240-image
decision gate that costs $9 to settle and determines whether ~$225 more is worth spending.

## Restart the background jobs (nothing resumes on its own)

```bash
cd /Users/gs/fake_it_til_you_make_it

# Wide-domain generation. Safe to re-run: existing .png files are skipped, so this
# resumes rather than regenerating, and does not double-charge.
nohup .venv/bin/python scripts/generate_wide_domain.py \
    --n 6000 --out data/synthetic/wide_pool --concurrency 2 > /tmp/wide_pool.log 2>&1 &

# Stage-1 gate. Fires automatically at 240 images. Needs no training.
nohup bash -c 'while [ $(ls data/synthetic/wide_pool/*.png 2>/dev/null | wc -l) -lt 240 ];
    do sleep 300; done; .venv/bin/python scripts/substrate_fidelity.py \
    > /tmp/stage1_gate.log 2>&1' > /dev/null 2>&1 &
```

Use `--concurrency 2`. Three gives a 31% failure rate; six fails outright with 429s. Do
**not** restart `generate_coverage_pool.py` — the 644-image face pool is deliberately
frozen because every classifier result is measured against that exact snapshot.

## The immediate next decision

Run `scripts/substrate_fidelity.py` when the wide pool reaches 240.

- **Baseline to beat: Scope 36.3%** (the face-only pool). Real images score 93.1%.
- **Early read at n=92 was 43.4%** — up, but the on-person subgroup scored *higher* than
  the off-person one, which is backwards from the hypothesis and suggests the gain may come
  from prompt diversity rather than from escaping the face prior. Do not trust it; n=24 in
  one arm.
- **Confound to check first:** close-up human-skin prompts are refused 60–83% of the time
  while artificial substrates are refused 0%. Substrate availability therefore correlates
  with proximity to the validation domain, so any substrate ranking must be read against
  refusal rate before it is believed.
- **Kill criterion:** if no substrate group beats 36.3%, the face prior is not what
  compresses severity, the wide-domain premise fails, and the remaining ~$225 should not be
  spent. Say so plainly and stop.

## What is settled (do not re-litigate)

| question | answer |
|---|---|
| Can generated images replace real ones? | **No.** −26.6 pts full budget, −19.8 matched budget and class balance |
| Do they help added across all classes? | **No.** +1.0, sd 2.1 |
| Do they help as pretraining? | **No.** +0.3 vs real, +0.9 vs a compute-matched two-stage control |
| Do they help targeted at scarce classes? | **Unresolved.** +0.0165 over a duplicated-real control, t=1.53, **p=0.159**, 5/10 seeds positive. Needs 22 seeds |
| Why? | Severity is compressed to **36.3% Scope** vs 93.1% for real images |

Full numbers in `results/tenseed_final.json`. The paper's position is that the tail effect
is **underpowered, not disproven**, and that published gains of this size are routinely
asserted from single runs.

## Open items, in priority order

1. The stage-1 gate above.
2. Verify the bib entries still flagged `NOT INDEPENDENTLY VERIFIED` in `paper/refs.bib`.
   Two fabricated author lists were already found and fixed; assume nothing.
3. `paper/main.tex` still has `\RESULT{}` and `\TODO{}` placeholders outside the Results
   section — `grep -n 'RESULT{\|TODO{' paper/main.tex`.
4. Figures are all commented-out `\includegraphics` stubs; no figure has been generated.

## Cost so far

~$29 of generation (~$25 frozen face pool, ~$4 wide pool). The full 6,000-image wide pool
is a further ~$230 and ~95 hours at the measured 63 images/hour.
