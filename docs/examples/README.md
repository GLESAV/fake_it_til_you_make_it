# Smoke-run artefacts

Output of `make smoke` (`fitymi smoke --quick`, gap=0.6) on 2026-08-19. **Toy data —
these are not results about acne.** They are here as evidence that the pipeline runs
end to end and that the analysis code recovers the effect it is supposed to recover.

## What the run did

480 procedural images, split 312 / 73 / 95 with all four grades present in each
split. Two synthetic pools of 700 images each: `synth_closed` at gap 0.6 and
`synth_open` at gap 0.9, the latter standing in for a generator that never saw the
real data and is therefore further from its distribution. Three substitution arms
(0 / 50 / 100 percent synthetic) plus the real-subset control, three seeds, TinyCNN
from scratch, CPU. 24 runs, ~18 minutes.

## What it found

| arm | synth_closed | synth_open |
|---|---|---|
| 0% synthetic | 0.620 | 0.620 |
| 50% synthetic | 0.422 | 0.356 |
| 100% synthetic | 0.320 | 0.283 |

Balanced accuracy on the sealed test split, mean over 3 seeds. Trend slopes
$-0.299$ (closed, $p=0.065$) and $-0.337$ (open, $p=0.028$).

Three things are working here, and each is a thing the code could plausibly have got
wrong:

1. The curve declines monotonically, so the mixing sampler, the training loop and the
   trend test agree with the mis-specification we injected.
2. `synth_open` sits below `synth_closed` at every fraction, matching the larger gap
   it was generated with. The pipeline is sensitive to gap *magnitude*, not merely to
   the presence of synthetic data.
3. The closed-set arm's slope interval crosses zero at three seeds even though the
   point estimate is large. That is the honest answer for this sample size, and it is
   why the protocol asks for five.

The audit-log excerpt shows the sealed test split being unsealed once per reported
model, with a reason and a run id, which is the mechanism described in protocol §3.3.

## Files

- `smoke_substitution_curve.png` / `.csv` — the figure and its underlying table
- `smoke_h2_trend.json` — trend test output, including per-seed slopes
- `smoke_split_quality.json` — realised vs. requested split sizes, no missing classes
- `smoke_test_unseal_audit_excerpt.jsonl` — first three unsealing records
