# Protocol amendment: the Gemini coverage arm

**Recorded 2026-08-20.** This supersedes the framing of §5 for the primary experiment. The
closed-set Stable Diffusion arm is retained but demoted; see §12.4 below.

## What changed and why

The original question was: *can a wide, deliberately more complete synthetic set — wider
than any real dermatology dataset — train a useful acne classifier, validated on real data?*

The protocol as written asked a different one: *can synthetic images substitute for real
ones at a matched training budget?* Matched budget is a controlled comparison, and it is a
reasonable question, but it is not this one. Worse, the generator it specified was
fine-tuned **on ACNE04**, and a closed-set generator cannot be more complete than its
training set — that is definitional. The symptom was visible and I recorded it neutrally
before noticing what it meant: the fine-tuned pool is overwhelmingly one cohort, because
its 948 training images are 267 people.

## §12.1 The generator

**Gemini 2.5 Flash Image ("nano banana"), prompted, no fine-tuning, no real image ever
shown to it.** Real data is validation only.

Measured: 1,290 output tokens per image, so ~3.9¢ each and ~$39 per thousand. Concurrency
2 — at 6 the cheap tier returned 429 for 12 of 14 requests. Vertex works on
application-default credentials; no API key required.

## §12.2 The pool is balanced by design, not matched to the real prior

This is the substantive change. The mixing protocol allocated synthetic images in
proportion to ACNE04's class histogram (35/43/12/9), so that a mixing arm differed from the
real arm in content rather than in label distribution. That is right for a substitution
study and wrong for this one: **ACNE04's tail scarcity is the deficiency the synthetic set
exists to fix.**

The pool is therefore **balanced across severity and across Fitzpatrick I–VI**, plus age,
sex, view and lighting. ACNE04 holds 129 very-severe images from 81 people; the coverage
arm can hold as many very-severe images as mild ones, across six skin tones.

**Consequence for the comparison, stated plainly:** an arm trained on a balanced synthetic
pool and an arm trained on ACNE04 differ in *two* ways — the images are synthetic, and the
label distribution is different. That confound is deliberate, because removing it would
remove the intervention. It is handled by reporting the real-only arm at both its natural
prior and class-rebalanced, so the reader can see how much of any difference is the
rebalancing alone.

## §12.3 Label fidelity is the risk that decides the arm

Hayashi grades are defined by *counted inflammatory lesions* — 1–5, 6–20, 21–50, >50 on
half the face. Nothing guarantees the model's "severe" coincides with 21–50 lesions. If it
does not, a classifier trained on severity-word prompts has learned a different label
function than the one it is validated against, and every downstream number is
uninterpretable.

Two ways out, tested in order:

1. **Prompt with the count, not the word** — "exactly 30 inflammatory lesions" — which
   makes the label true by construction if the model complies.
2. **Measure and recalibrate**: count lesions on a sample, fit the map from requested
   severity to realised count, and re-band.

**Pre-committed:** if neither produces a stable mapping, the arm is reported as predicting
*"the model's notion of severity"* and not Hayashi severity, and the comparison to ACNE04
accuracy is dropped rather than dressed up.

## §12.4 What the fine-tuned arm is now for

Not deleted — demoted to **the baseline the coverage arm has to beat**. It answers "does
breadth beat domain fidelity?", which is a sharper question than either arm alone:

- **Closed-set (SD, fine-tuned on ACNE04):** maximum domain fidelity, zero breadth beyond
  the real data. Already generated to 2,812 images before being stopped.
- **Coverage (Gemini, prompted):** maximum breadth, no domain fidelity to ACNE04's specific
  capture protocol.

If the coverage arm wins, breadth matters more than matching the target corpus, which is
the useful finding for anyone building a synthetic medical dataset. If the fine-tuned arm
wins, the opposite, and "just prompt a frontier model" is not the answer.

## §12.5 What carries over unchanged

Everything about validation, and it matters more here than it did before. The sealed test
split, subject-disjoint splitting, the memorisation audit, the identity-diversity measure
and the ACNE04 data audit all serve the corrected question directly — because "validate on
real data" is only meaningful if the real data's labels and splits are understood, and
`docs/05_acne04_audit.md` establishes that ACNE04's labels reproduce at 30.1% against a
second expert team and its published folds leak 77.5% of test subjects.

---

## §12.6 First fidelity results

Instrument: ResNet-50 trained on the real subject-disjoint training split, 0.7296 balanced
accuracy on real held-out images. The same scorer is run on every set, and the question is
its agreement with the grade each image was *requested* with.

| set | Spearman | exact | note |
|---|---|---|---|
| **real held-out** | **0.706** | 70.6% | the ceiling |
| Gemini, word-prompted, n=66 | 0.199 | 34.8% | 51 of 66 predicted as one grade |
| Gemini, best crop (70%) | **0.333** | 44.8% | framing recovered, no regeneration |
| SD fine-tuned on ACNE04, n=2,812 | **−0.103** | 32.4% | in-domain, no severity signal |

### What the control establishes

The obvious objection is domain shift: a classifier trained on ACNE04's half-face crops
might fail on studio portraits for reasons unrelated to severity, making the Gemini number
meaningless. **The in-domain control rules that out in the direction that matters.** The
generator fine-tuned *on ACNE04* — same crops, same lighting, same cohort — scores *higher*
on exact agreement (32.4%) and has **no severity control at all**: Spearman −0.103, with
mean predicted grade 1.29 for requested mild against 1.25 for requested moderate. It learned
the look and not the label.

So a prompted model that has never seen the dataset has better severity control than one
fine-tuned on it. That is the single most useful result so far, and it is the opposite of
what the closed-set framing assumed.

### What is still wrong

Gemini's severity signal is **monotone but heavily compressed**: mean predicted grade runs
0.75 → 1.00 → 1.17 → 1.60 across the four requested grades. The direction is right; the
dynamic range is about a third of what it should be.

Framing accounts for part of it. Cropping the portraits towards ACNE04's view lifts Spearman
0.210 → 0.333 and exact 34% → 51%, and over-cropping to 35% destroys it again, which
confirms the mechanism is how much cheek is in frame. But the best crop still leaves a
0.37 gap to the ceiling.

### What is being tested next

Both remedies are prompt changes, chosen by measurement:

1. **Ask for the tight framing directly**, rather than cropping afterwards and losing
   resolution.
2. **Prompt with an explicit lesion count** drawn from inside each Hayashi band, rather than
   a severity word. Hayashi grades *are* counts, so this makes the label true by
   construction if the model complies — and an earlier probe showed it complies at low
   counts and saturates at high ones.

**Pre-committed reading.** If the count-prompted set does not clear roughly 0.5 Spearman,
the arm is reported as predicting the model's notion of severity rather than Hayashi
severity, per §12.3, and the honest conclusion is that this generator produces excellent
*coverage* and unreliable *labels* — which is a useful finding about synthetic medical data
even though it is not the hoped-for one.

