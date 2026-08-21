# Resume state — paused 2026-08-21 for a machine restart

## How to restart the two background jobs

```bash
cd /Users/gs/fake_it_til_you_make_it

# 1. Wide-domain generation. Resumes automatically: existing .png files are skipped.
#    Concurrency 2 is measured -- 3 gives a 31% failure rate, 6 fails outright with 429s.
nohup .venv/bin/python scripts/generate_wide_domain.py \
    --n 6000 --out data/synthetic/wide_pool --concurrency 2 > /tmp/wide_pool.log 2>&1 &

# 2. Stage-1 gate. Fires at 240 images; needs no training, only the saved scorer.
nohup bash -c 'while [ $(ls data/synthetic/wide_pool/*.png 2>/dev/null | wc -l) -lt 240 ];
    do sleep 300; done; .venv/bin/python scripts/substrate_fidelity.py \
    > /tmp/stage1_gate.log 2>&1' > /dev/null 2>&1 &

# 3. Only the last seed of the 10-seed run is missing. To finish it:
nohup .venv/bin/python scripts/train_on_synthetic.py --balance --seeds 9 \
    --arms real mixed_tail mixed_tail_control > /tmp/seed9.log 2>&1 &
```

## Where the two pools stand

| pool | images | status |
|---|---|---|
| `data/synthetic/gemini_pool` | **644** | **frozen** — the face-only pool; all classifier results use it |
| `data/synthetic/wide_pool` | **92** | growing toward the 240-image stage-1 gate, then 6,000 |

Do not resume the old `generate_coverage_pool.py`; `gemini_pool` is deliberately frozen so
the 10-seed run trains on a fixed snapshot.

## The headline result, as it stands

Nine paired seeds on the frozen pool:

| arm | n | balanced accuracy |
|---|---|---|
| real (948) | 9 | 0.7389 ± 0.0202 |
| mixed_tail (generated tail) | 9 | 0.7575 ± 0.0282 |
| mixed_tail_control (duplicated real tail) | 9 | 0.7420 ± 0.0214 |

**Paired content effect: +0.0156, sd 0.0358, t = 1.30, p = 0.229, 95% CI [−0.012, +0.043],
4 of 9 seeds positive.** It has drifted down as seeds accumulated — +0.0221 at n=2,
+0.0224 at n=6, +0.0156 at n=9 — and the interval has always spanned zero since n=2 was
abandoned. Power analysis: **22 seeds for 80% power**. Partial results in
`results/tenseed_partial.json`.

The paper's claim is that this is **underpowered, not disproven**, and that published gains
of this magnitude are routinely asserted from single runs.

## What is settled

- **Substitution fails**: −26.6 points at full budget, −19.8 budget- and class-matched.
- **All-class augmentation**: +1.0, sd 2.1. Null.
- **Pretraining**: +0.3 over real, +0.9 over a compute-matched two-stage real→real control.
  Null — and that control is the one Moroianu et al.'s +6.5% result omits.
- **Compression mechanism**: Continuity 70.4%, Scope 36.3% against a real-image ceiling of
  88.4% / 93.1%. Worse on both axes than a 2024 LoRA slider.
- **ITA is exposure-confounded**: replicated on 934 SCIN images; a matched-effect-size
  control shows a genuine association retains 60.7% inside a device stratum while ITA
  retains −16.1%.
- **Bibliography**: two fabricated author lists and one citation to a non-existent paper
  found and fixed. Both manuscripts compile.

## Open, in priority order

1. **Stage-1 gate at 240 wide-domain images** — does moving acne off the face widen Scope
   beyond 36.3%? Kill criterion: if no substrate beats it, the wide-domain premise fails and
   the remaining $225 should not be spent.
2. Finish seed 9 of the 10-seed run (one arm-triple).
3. Requalify the Zein et al. 97.6% figure in `main.tex` — it is uninterpretable, not a real
   result (generator trained on all real images, test set never described, severity labels
   assigned by the authors sorting generator outputs).
4. Cite and distinguish **ACNEDIT** (Piat et al., DGM4MICCAI 2025) — same dataset, same two
   scarce classes, same intervention; no independent grader, no rebalancing control.
5. Verify the bib entries still flagged `NOT INDEPENDENTLY VERIFIED`.

## Cost so far

~$25 on the frozen face pool, ~$4 on the wide pool. The full 6,000-image wide pool is a
further ~$230 and ~95 hours at the cheap tier's measured 63 images/hour.
