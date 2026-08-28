# Resume state — updated 2026-08-28

## How to restart the two background jobs

```bash
cd /Users/gs/fake_it_til_you_make_it

# Auth: Vertex on ADC, project watchmen-4d5b1. No API key needed, but the credentials
# do not survive a reboot cleanly -- check `gcloud auth application-default
# print-access-token` before blaming the generator.

# 1. Wide-domain generation. Resumes automatically: existing .png files are skipped, and
#    since 5767e47 prompts refused twice with no rate limit in between are skipped too.
#    Concurrency 2 is measured -- 3 gives a 31% failure rate, 6 fails outright with 429s.
nohup .venv/bin/python scripts/generate_wide_domain.py \
    --n 6000 --out data/synthetic/wide_pool --concurrency 2 > /tmp/wide_pool.log 2>&1 &

# 2. Stage-1 gate. Fires at 240 images; needs no training, only the saved scorer.
nohup bash -c 'while [ $(ls data/synthetic/wide_pool/*.png 2>/dev/null | wc -l) -lt 240 ];
    do sleep 300; done; .venv/bin/python scripts/substrate_fidelity.py \
    > /tmp/stage1_gate.log 2>&1' > /dev/null 2>&1 &
```

The 10-seed run is finished; `results/tenseed_final.json` is the record.

## Where the two pools stand

| pool | images | status |
|---|---|---|
| `data/synthetic/gemini_pool` | **644** | **frozen** — the face-only pool; all classifier results use it |
| `data/synthetic/wide_pool` | **92** | stalled at 92 since 2026-08-21; see below |

Do not resume the old `generate_coverage_pool.py`; `gemini_pool` is deliberately frozen so
the 10-seed run trains on a fixed snapshot.

**The wide pool is not currently growing.** Restarted 2026-08-28 after a week idle; in the
first hour every completed prompt came back `no candidates`, including substrates that
previously had a 0% empty rate. Those are the 32 prompts the resume re-attempts first, so
it is not yet evidence of a policy change at the backend — but if the pool is still at 92
after the queue clears, the wide-domain arm is blocked on generation and not on analysis,
and the kill criterion in "Open" item 1 should be applied on that basis rather than on a
fidelity result that will never arrive.

## The headline result, as it stands

Ten paired seeds on the frozen pool (`results/tenseed_final.json`):

| arm | n | balanced accuracy |
|---|---|---|
| real (948) | 10 | 0.7347 |
| mixed_tail (generated tail) | 10 | 0.7579 |
| mixed_tail_control (duplicated real tail) | 10 | 0.7414 |

**Paired content effect: +0.0165, sd 0.0339, t = 1.53, p = 0.159, 95% CI [−0.008, +0.041],
5 of 10 seeds positive.** It drifted down as seeds accumulated — +0.0221 at n=2, +0.0224 at
n=6, +0.0156 at n=9, +0.0165 at n=10 — and the interval has spanned zero since n=2 was
abandoned. Power analysis: **22 seeds for 80% power**.

The paper's claim is that this is **underpowered, not disproven**, and that published gains
of this magnitude are routinely asserted from single runs. The classical arm (§14) supplies
an independent second argument: on a fresh cohort of this size the standard error of
balanced accuracy is 4.3 points, and +1.65 sits well below it.

## What is settled

- **Substitution fails**: −26.6 points at full budget, −19.8 budget- and class-matched.
- **All-class augmentation**: +1.0, sd 2.1. Null.
- **Pretraining**: +0.3 over real, +0.9 over a compute-matched two-stage real→real control.
  Null — and that control is the one Moroianu et al.'s +6.5% result omits.
- **Compression mechanism**: Continuity 70.4%, Scope 36.3% against a real-image ceiling of
  88.4% / 93.1%. Worse on both axes than a 2024 LoRA slider.
- **Classical-ML arm (§14)**: the registered prediction — a *larger* substitution deficit
  under a fixed-feature head — is **falsified**. Largest deficit 18.8 points against the
  deep study's 26.6, mean across five heads 8.3.
- **ITA is exposure-confounded**: replicated on 934 SCIN images; a matched-effect-size
  control shows a genuine association retains 60.7% inside a device stratum while ITA
  retains −16.1%.
- **Bibliography**: four defects found in this project's own record — two fabricated author
  lists, one citation to a paper that does not exist, one entry conflating two real papers,
  and one with an invented title and four wrong given names. All fixed; `docs/VERIFY.md`
  carries both verification passes. `make publication-gate` enforces the rule.

## Open, in priority order

1. **Stage-1 gate at 240 wide-domain images** — does moving acne off the face widen Scope
   beyond 36.3%? Kill criterion: if no substrate beats it, the wide-domain premise fails and
   the remaining ~$225 should not be spent. **Blocked on generation, not on analysis.**
2. Read `moroianu2025` and `sagers2023` in full. Both are load-bearing — the manuscript
   makes specific claims about what their designs omit — and neither carries a
   `VERIFIED <date>` note. `docs/VERIFY.md` item 9.
3. Move the `VERIFIED` provenance notes from `note` to `annote` so they stop printing in
   the manuscript's bibliography. They cost `main.pdf` a page. `docs/VERIFY.md` item 10.
4. Run the prototype-effect checks (protocol §8.6) on the classical arm's linear SVM, the
   one head where synthetic beats real. Until then it is not reported as a win.
5. `main.tex` still holds 17 `\RESULT{}` / `\TODO{}` placeholders, which are the whole of
   what `make publication-gate` now blocks on.

## Cost so far

~$25 on the frozen face pool, ~$4 on the wide pool. The full 6,000-image wide pool is a
further ~$230 and ~95 hours at the cheap tier's measured 63 images/hour — a rate the
current run is nowhere near.
