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

---

## §12.7 The coverage claim, measured

The arm's premise is that a prompted generator supplies breadth no real acne dataset has.
That had been asserted from a contact sheet, which is the kind of evidence this project has
repeatedly shown to be unreliable. Measured with individual typology angle (ITA) from CIE
L*a*b* on the central skin region, erythematous pixels dropped so lesions do not drag the
estimate darker, 300 images from each corpus scored identically:

| ITA bin | ACNE04 train | Gemini pool |
|---|---|---|
| very light | 3.7% | 7.0% |
| light | 11.0% | 24.6% |
| intermediate | **40.0%** | 20.3% |
| tan | 29.0% | 17.6% |
| brown | 13.7% | 19.3% |
| **dark** | **2.7%** | **11.2%** |

| | two darkest bins | darkest bin | evenness |
|---|---|---|---|
| ACNE04 | 16.3% | **2.7%** | 0.81 |
| Gemini pool | **30.3%** | **11.2%** | **0.96** |

**The claim holds.** The synthetic pool carries roughly twice the representation in the two
darkest bins and **four times in the darkest**, and fills the tone range far more evenly —
ACNE04's interquartile ITA range is 15.7–37.0 against the pool's 1.2–45.7. ACNE04 is
concentrated in the middle, consistent with a single-cohort clinical collection; the pool is
spread by design, because it was prompted across Fitzpatrick I–VI.

### A metric that inverted the story

The first statistic computed for this was "share outside the lightest bins", and it made
ACNE04 look *broader*: 85.3% against the pool's 68.4%. It is a real number and it is
useless, because a corpus concentrated in the **middle** scores high on it — ACNE04 rates
85% while holding 2.7% dark-skinned images. What the coverage claim is about is
representation at the underserved end and how evenly the range is filled, which is what the
table above reports.

That is the same failure as §12.6's exact-agreement metric, which rose while ordinal fidelity
collapsed. Both times a plausible summary statistic moved the wrong way for a mechanical
reason, and both times the fix was to state what the claim is actually about before choosing
how to measure it.

### What this does not establish

ITA is a proxy for constitutive pigmentation, not a Fitzpatrick or Monk label, and generated
images have no ground truth to check it against. The defensible statement is a comparison
between two distributions measured the same way, not either distribution in absolute terms.
Nor does coverage imply usefulness: §12.6 shows the same pool's severity labels are
unreliable, and a set that is broad and mislabelled is not obviously better than one that is
narrow and correct. **Those are two separate axes and this measures only the first.**

---

## §12.8 First budget-matched result: synthetic is worth less, and the deficit is in the tail

Three arms, all scored on the same real subject-disjoint validation split (218 images).
One seed, 156-image pool — preliminary, and the direction is already clear.

| arm | n train | balanced accuracy | accuracy | QWK |
|---|---|---|---|---|
| synthetic only | 156 | **0.426** | 0.450 | 0.509 |
| **real, budget- and class-matched** | **156** | **0.630** | 0.555 | 0.668 |
| real, full split, natural prior | 948 | 0.730 | 0.706 | 0.783 |

**At matched budget and matched label distribution, a generated image is worth substantially
less than a real one: 0.426 against 0.630, a gap of 20 points.** The third row is there to
show what the budget cut alone costs — going from 948 real images to 156 costs 10 points, and
replacing those 156 with synthetic costs another 20.

### Where the deficit sits is the finding

| arm | mild | moderate | severe | very severe |
|---|---|---|---|---|
| synthetic only | **0.833** | 0.225 | 0.423 | **0.222** |
| real, matched | 0.694 | 0.392 | 0.654 | **0.778** |

**Synthetic *beats* real at mild** (0.833 against 0.694) and collapses at very severe (0.222
against 0.778). That is the same shape the fidelity measurement predicted from an entirely
different direction: the generator renders mild acne convincingly and cannot render severe
disease, so a classifier trained on it learns the easy end and not the hard one.

### Why this is the uncomfortable result

The synthetic set's advertised advantage was coverage of what the real data lacks. ACNE04
lacks two things:

- **demographic breadth** — and the pool genuinely supplies it, with four times the
  darkest-bin representation and a near-uniform tone spread (§12.7);
- **severe cases** — 129 very-severe images from 81 people — **and the pool does not supply
  it at all.**

The gap the synthetic data closes is not the gap that limits the benchmark. The one that
does limit it is precisely where the generator fails, and it fails there for a reason that
is visible in the images rather than mysterious: asked for confluent nodulocystic acne, it
draws scattered papules.

### Standing caveats

One seed, 156 images, validation rather than the sealed test split. The full 960-image pool
will give a learning curve, and if the synthetic curve is still climbing at 960 the deficit
is partly a sample-size artefact rather than a ceiling. That is the one result that would
soften this, and it is worth waiting for — but the per-class pattern would have to change,
not just the aggregate, because a curve that rises while very-severe recall stays at 0.22
does not rescue the claim.


---

## §12.9 The mechanism: monotone but compressed to 41% of the real severity range

The classifier result says generated images are worth less. This says *why*, and the answer
is not the one §12.8 assumed.

The instrument is the real-trained classifier (0.730 balanced accuracy on real held-out
data). It is asked one question of every generated image: what severity does this *look*
like? Run against the requested grade, that gives a request-to-perception matrix.

**Generated images, 270 of them:**

| requested ↓ / perceived → | mild | moderate | severe | very severe |
|---|---|---|---|---|
| mild (n=128) | 0.13 | **0.87** | 0.00 | 0.00 |
| moderate (n=49) | 0.00 | **1.00** | 0.00 | 0.00 |
| severe (n=46) | 0.00 | **0.93** | 0.04 | 0.02 |
| very severe (n=47) | 0.00 | 0.30 | 0.40 | 0.30 |

**The same instrument on real held-out images — the control that decides whether the
compression is the generator's or the scorer's:**

| true ↓ / perceived → | mild | moderate | severe | very severe |
|---|---|---|---|---|
| mild (n=72) | **0.82** | 0.17 | 0.00 | 0.01 |
| moderate (n=102) | 0.17 | **0.64** | 0.14 | 0.06 |
| severe (n=26) | 0.00 | 0.15 | **0.46** | 0.38 |
| very severe (n=18) | 0.00 | 0.00 | 0.00 | **1.00** |

Read the two as mean perceived grade against requested grade:

| | mild | moderate | severe | very severe | span |
|---|---|---|---|---|---|
| real images | 0.21 | 1.09 | 2.23 | 3.00 | **2.79 of 3** |
| generated images | 0.87 | 1.00 | 1.09 | 2.00 | **1.13 of 3** |

**The generator's severity dynamic range is 41% of the real one.** The relationship is
monotone — Spearman 0.593, asking for more severity does reliably produce more — so this is
compression, not noise. But four clinical grades come out as roughly one grade of real
variation, centred on moderate.

### This corrects §12.8's diagnosis

§12.8 read the per-class recalls as "the generator draws convincing mild acne and cannot
draw severe disease". The matrix says something different and more specific: **it fails at
both ends.** 87% of images requested as *mild* are perceived as moderate — it cannot render
nearly-clear skin any more than it can render confluent nodulocystic disease. Everything
regresses to the middle of the scale.

That changes what a fix would look like. If the deficit were only at the severe end, the
move would be better severe prompting, or a different generator for the tail. If the
generator is compressing the whole scale toward its own prior, prompt engineering at one
end will not fix it, and the ceiling is a property of the model rather than of the wording —
consistent with the count-prompting attempt (§4.7), which made severity fidelity *worse*.

### What the control rules out

The obvious objection is that the scorer, trained on real images, simply hedges toward the
prior when shown out-of-distribution inputs, and the compression is the instrument's. The
real-image row rules that out: the same scorer, same weights, same preprocessing, spans
0.21 to 3.00 and calls 100% of real very-severe images very severe. It compresses generated
images and not real ones. Whatever is missing from the severe end of the pool is missing
from the images, not from the measurement.

---

## §12.10 Why rejection sampling does not rescue it

The obvious fix for compression is to over-generate and keep only the images that land where
they were asked to. The request-to-perception matrix gives the pass rates directly, so the
cost of that strategy is arithmetic rather than speculation.

| grade | pass rate | generations needed for 240 | cost | hours |
|---|---|---|---|---|
| mild | 13% | 1,846 | $72 | 29 |
| moderate | 100% | 240 | $9 | 4 |
| severe | **4%** | **6,000** | **$234** | **95** |
| very severe | 30% | 800 | $31 | 13 |
| **total** | | **8,886** | **$347** | **141** |

Against $37 and 15 hours for the unfiltered pool of the same size: **filtering multiplies
cost and wall-clock by 9.3×**, and three quarters of that goes to one class. At 63 images
per hour — the sustained Vertex rate on the cheap tier — the severe class alone is four
days of continuous generation.

For scale: ACNE04 already contains 126 severe and 86 very-severe images, and they cost
nothing.

### The cost is the smaller objection

Money and wall-clock are negotiable; the conceptual problem is not. Filtering on a
real-trained scorer means keeping generated images **in proportion to how much they
resemble ACNE04's severe cases**. The filtered pool is then selected by the real
distribution, which reintroduces exactly the closed-set ceiling the coverage arm exists to
escape — the same ceiling that makes an ACNE04-tuned SD LoRA unable to exceed its training
set. A pool that can only contain what a real-trained model recognises cannot be more
complete than the real data.

That does not make the filter useless — it is the right instrument for *measuring*
fidelity, which is what §12.9 uses it for. It makes it the wrong instrument for
*constructing* the pool, and the distinction is worth keeping straight, because the cost
table above is tempting enough to hide it.

---

## §12.11 Retraction: the per-class classifier story does not survive a second seed

§12.8 read the per-class recalls from a single seed and concluded that synthetic training
"beats real at mild and collapses at very severe". Two seeds at n=180 invert it:

| arm | n | balanced accuracy | mild | moderate | severe | very severe |
|---|---|---|---|---|---|---|
| synthetic | 180 | 0.438 ±0.054 | 0.319 | 0.265 | 0.250 | **0.917** |
| real, matched | 180 | 0.624 ±0.032 | 0.722 | 0.446 | 0.327 | 1.000 |

Very-severe recall on the synthetic arm went from **0.222 to 0.917** between runs, and mild
from 0.833 to 0.319. The direction of the claim reversed completely.

The cause is not mysterious. The validation split holds 72 / 102 / 26 / 18 images:

| class | n | one image is worth | SE of a recall near 0.5 |
|---|---|---|---|
| mild | 72 | 1.4 pts | 5.9 pts |
| moderate | 102 | 1.0 pts | 5.0 pts |
| severe | 26 | 3.8 pts | 9.8 pts |
| very severe | **18** | **5.6 pts** | **11.8 pts** |

Per-class recall on 18 images cannot support a claim about which end of the scale a
generator is good at. It was over-read, and the reading is withdrawn.

### What survives

**The aggregate gap does.** 0.186 with a standard error of 0.044 across seeds — t ≈ 4.2. At
matched budget and matched label distribution a generated image is worth substantially less
than a real one, and that has now held at n=76, 156 and 180, across every seed run.

**The compression result (§12.9) does, and it says the same thing far more reliably.** It is
measured directly on 270 generated images against a control, with no training-seed noise in
it at all: mean perceived grade 0.87 / 1.00 / 1.09 / 2.00 against a real-image span of 0.21
to 3.00. That is the evidence for what the generator can and cannot render. The per-class
recalls were a noisy shadow of it, pointing the same direction on one seed and the opposite
on the next.

### The pattern, again

This is the fourth time in this project that a causal story computed from a small slice
failed on contact with a second measurement, and the audit's §14 already names the shape:
the numbers were real, the arithmetic was right, and the story laid over them was not
supported. The defence that worked here is the same one that worked there — compute the
thing a second way, on a sample large enough to move less than the effect.

---

## §12.12 The exchange rate, and why it is probably optimistic

Six synthetic-only points have now been measured, all scored on the same real
subject-disjoint validation split:

| pool size | balanced accuracy |
|---|---|
| 76 | 0.391 |
| 156 | 0.426 |
| 180 | 0.476 / 0.400 (two seeds) |
| 204 | 0.477 |
| 272 | 0.458 |

A log fit gives **+0.042 balanced accuracy per doubling** of the pool (R² = 0.47 — weak,
and worth stating rather than hiding). Extrapolated:

| target | generated images needed | cost | wall-clock at 63/hr |
|---|---|---|---|
| match 156 real images (0.624) | ~3,500 | $138 | 56 h |
| match 948 real images (0.730) | ~20,000 | $790 | 322 h |

Read as an exchange rate: **roughly 23 generated images per real image**, or about $0.90 of
generation per real-image-equivalent. That is not an absurd number — it is within range of
what annotation alone costs for clinical imagery — which is why it deserves the caveat that
follows rather than a headline.

### The extrapolation is probably wrong, and this project already knows why

A log curve says the pool reaches any target eventually. **The compression result (§12.9)
says it does not.** The generator's perceived-severity range is 1.13 grades against real
data's 2.79, and no quantity of images widens that: every additional very-severe request
returns another image the scorer reads as moderate. Where an axis of the label space is not
rendered at all, more samples along it add nothing, and the curve must flatten short of the
real ceiling rather than continue.

The measured points are consistent with that already. The first three doublings gain
+0.047; the last half-doubling gains +0.020 and the most recent point is *below* the one
before it. Six noisy points cannot distinguish a log curve from a saturating one, but only
one of those two shapes has a mechanism behind it, and it is the one that stops.

So the honest reading of the table above is an **upper bound on what the pool can buy**, not
a plan. The measurement that would settle it is the shape of the curve between 960 and
~4,000 images, which is $115 and roughly 50 hours of generation away, and worth doing only
if the mixed-arm controls come back positive — because if generated images cannot help even
when added to real ones, what they would cost to replace real ones is an academic question.

---

## §12.13 The first result that survived its control

`mixed_tail` — the real training split plus generated images for the two scarce classes
only — beat the real baseline by 3.1 points. That gain had an obvious alternative
explanation: topping up severe and very-severe flattens the class distribution, and
rebalancing is worth something on its own. `mixed_tail_control` adds the same number of
images to the same classes by **duplicating real ones**, so it delivers identical
rebalancing with zero new information.

| arm | n train | balanced accuracy | vs real |
|---|---|---|---|
| real | 948 | 0.7337 ±0.006 | — |
| mixed_tail_control (duplicated real tail) | 1,114 | 0.7422 ±0.004 | +0.85 |
| **mixed_tail (generated tail)** | 1,114 | **0.7643 ±0.007** | **+3.06** |

**Of the 3.1-point gain, 0.9 is rebalancing and 2.2 is the generated images**
(SE 0.0057, t ≈ 3.9). The control took a third of the effect and left two thirds standing.

### The per-class rows say why

| arm | mild | moderate | severe | very severe |
|---|---|---|---|---|
| real | 0.826 | 0.647 | 0.462 | 1.000 |
| mixed_tail_control | 0.889 | 0.657 | **0.423** | 1.000 |
| mixed_tail | 0.861 | 0.696 | **0.500** | 1.000 |

Duplicating real severe images makes severe recall **worse** than not touching it at all —
0.423 against 0.462. That is what adding copies does: no new information, more opportunity
to overfit the handful of severe subjects that exist. The generated images raise it to
0.500 instead. Imperfect images that are *different* beat perfect images that are
*repeated*, which is the entire case for synthetic augmentation stated in one row.

### What this does and does not establish

It establishes that generated acne images carry information a classifier can use, over and
above the rebalancing they incidentally provide, in the one place the real dataset is
thinnest. At 2.2 points on a 0.734 baseline it is a modest effect, not a transformative one,
and it sits alongside the finding that the same images cannot replace real ones at all
(§12.8, a 27-point deficit at matched budget).

It rests on two seeds per arm. Three more are running, and the number to watch is whether
the mixed_tail–control gap holds at 2.2 points or shrinks; the arms are individually stable
(sd 0.004–0.007) which is why two seeds separated them at all, but five is the minimum this
deserves before it goes in a paper.

---

## §12.14 Pretraining does not survive its control; only tail augmentation does

`pretrain` — synthetic first, then fine-tune on real — looked as strong as `mixed_tail` on
seed 0, at 0.7711 against the real baseline's 0.7296. It does not hold. Two-stage training
gives it roughly twice the gradient steps of the single-stage arms, so `real_twostage` runs
the same two stages on real data both times, varying only what stage one saw:

| arm | n train | balanced accuracy | vs real |
|---|---|---|---|
| real | 948 | 0.7337 ±0.006 | — |
| real_twostage (two stages, real both times) | 1,896 | 0.7273 ±0.008 | −0.64 |
| pretrain (synthetic, then real) | 1,380 | 0.7367 ±0.013 | +0.30 |

**+0.3 points over the real baseline and +0.9 over the compute control, against its own
standard deviation of 1.3.** The seed-0 result was noise, and the arm is a null. Worth
noting that `real_twostage` also sits *below* the single-stage baseline: the extra compute
is not merely unhelpful, it costs a little, presumably to overfitting.

### The two arms together are the finding

| use of generated images | effect | survives control? |
|---|---|---|
| replace real images entirely | −27 points | n/a — it is the deficit |
| add across all classes (`mixed`) | +1.0, sd 2.1 | no, indistinguishable from zero |
| pretrain, then fine-tune on real | +0.3, sd 1.3 | **no** — `real_twostage` |
| **add to the scarce classes only** | **+2.2 over control** | **yes** — t ≈ 3.9 |

Three of the four ways to use these images do nothing. The claim that survives is narrow and
specific: **generated images help when they are targeted at the classes the real dataset is
thinnest in, and not otherwise.** Spreading them across the whole label space dilutes them
into noise; using them to teach general features before fine-tuning does nothing a second
pass over the real data would not do.

That specificity is what makes it usable advice rather than an encouraging trend. It also
follows directly from the compression result (§12.9): if the generator renders a narrow band
of severity centred on moderate, then generated images are worth most exactly where real
images are scarcest and worth nothing where real images are already plentiful — which is
what these four rows say.

---

## §12.15 Positioning the compression result: measured with the field's own instrument

Two literature sweeps place §12.9 precisely. The short version: **the finding appears to be
new, the instrument is not, and the correct framing is much sharper than what was written.**

### The metric already has a name, and a lineage

What §12.9 calls "perceived severity range" is the **re-scoring protocol**, introduced by
InterFaceGAN (Shen et al., TPAMI 2020) and named **effectiveness** by Monteiro et al.
(ICLR 2023): apply an *independent* predictor to a generated image and ask whether it
recovers the value that was requested. Two 2025 papers give the exact pair of statistics
used here:

- **Continuity** (CompSlider, arXiv:2509.01028) — the share of ordered pairs where the
  higher request scores higher. A monotonicity rate.
- **Scope** (same) — the gap between the mean scored value at the highest and lowest
  request. A dynamic range.

Recomputing §12.9 in those terms, with the same real-trained scorer:

| | Continuity | Scope |
|---|---|---|
| **prompted severity language (this work)** | **70.4%** | **36.3%** |
| real images, same scorer (ceiling) | 88.4% | 93.1% |
| Concept Sliders, human attributes | 73.4% | 54.4% |
| CompSlider, human attributes | 81.1% | 59.0% |

**Prompted degree language in a 2025 frontier model is worse on both axes than a 2024 LoRA
slider.** That comparison is the single most useful sentence available for this result, and
it is only available because the slider literature had already defined the statistics.

### Someone asserted this finding in 2023 without measuring it

Takezaki et al. (ISBI 2023, arXiv:2302.12482), on ulcerative-colitis severity:

> "the level y′ is not fully reliable as the **absolute** level. However, y′ is still
> reliable as a **relative** level"

They build their entire method on it — MSE loss on real images, a learning-to-rank loss on
generated ones — precisely because they distrust the generated absolute level. That is the
Spearman-preserved / range-compressed dichotomy of §12.9, stated as an engineering premise
three years ago and **never quantified**. The honest framing of this section is therefore
*"we measure what Takezaki et al. assumed"*, not *"we discovered"*.

### The paper to position against, and why its number does not refute this one

Schmidt, Berens & Müller (arXiv:2602.24013, Feb 2026) condition a diffusion model on
diabetic-retinopathy severity and verify it with an independently trained CORAL ordinal
ResNet-50, reporting **QWK 0.79–0.87** between predicted and requested grade. That is the
state of the art for this question, and it is a stronger result than anything here.

**It is also fully compatible with severe range compression.** Quadratic weighting rewards a
predictor that is monotone but compressed, so a high QWK cannot rule out the effect §12.9
measures — and this project's own numbers demonstrate exactly that gap: Spearman 0.593 while
Scope is 36.3%. An agreement scalar and a range ratio are different measurements, and every
grade-conditional generation paper found reports only the former:

| shape of verification | example | blind to compression? |
|---|---|---|
| agreement scalar vs requested grade | fundus ordinal DM, QWK 0.87 | **yes** |
| expert grading accuracy, mean only | DR-GAN, 85.3% (3 ophthalmologists) | **yes** |
| per-grade expert accuracy | knee OA morphing, 62/86/87/90% by KL grade | partly |
| presence/absence, no ordinal axis | RoentMod 89–99%, DermGAN 0.45 vs 0.61 | n/a |

### The sign is novel; the opposite sign is published

Xia et al. (MICCAI 2024) run independent classifiers on counterfactual chest X-rays and find
**attribute amplification** — the requested axis pushed *further* than asked — and trace it
to hard labels in counterfactual training. This work is the **under**-expression counterpart
on an ordinal clinical axis from a prompted foundation model. A novel direction, with an
existing mechanism in the literature to argue against in discussion.

### The acne-specific neighbour a reviewer will name

**ACNEDIT** (Piat et al., DGM4MICCAI 2025) is a GAN that modifies acne severity with
"dynamic intensity tuning", trained on ACNE04, used to rebalance ACNE04's severity
distribution — the same dataset, the same intervention as §12.13. It is evaluated by
downstream segmentation (+7.85% IoU, +8.56% Dice) and **has no independent severity grader
and no check that the requested intensity was achieved.** It must be cited, and the
distinction stated: it does the intervention, this work measures whether the intervention
delivers what it requested.

Also worth noting: the PLOS ONE 2024 acne StyleGAN2 work reports 97.6% synthetic-to-real
transfer with severity classes **sorted by the authors' own manual inspection** — no
dermatologist, no independent grader. Near-ceiling accuracy on synthetic test data is weak
evidence of over-separable prototypical classes rather than of fidelity, and is never framed
that way there.

### Exposure

Nothing found reports the spread of an independently-predicted severity variable for
generated medical images *and* compares it to the same statistic on real images, in any
modality. The 1.13-against-2.79-grade comparison appears to be new. The obvious reviewer
request is to compute Scope on the released fundus ordinal-diffusion model, which is public;
that is worth pre-empting rather than waiting for.

---

## §12.16 The organising fact: results split by generator provenance, not by modality

A second literature sweep produced the framing this whole arm should have started from.

Published results on synthetic training data do **not** divide by imaging modality or by task.
They divide by **whether the generator ever saw the target dataset**:

| regime | examples | synthetic-only vs real | augmentation gains |
|---|---|---|---|
| **closed-set** — generator trained or fine-tuned on the target real split | Khosravi 2024 (CheXpert DDPM), Frid-Adar 2018, Yu 2023, Narahari 2025, Akrout 2023, RoentGen-v2, **ACNEDIT** | at or near parity | reliable |
| **open-set** — foundation generator used zero-shot | Sariyildiz 2023, Fan 2024, He 2023, **this work** | **−7 to −37 points** | null |

Wang et al. (arXiv:2508.09550) show both regimes on the *same* dataset: an exchange rate of
~35:1 closed-set against ~1:1 open-set.

**Our −27 points is exactly what the open-set literature predicts.** It is not a surprising
result and must not be presented as one. It also identifies the obvious reviewer question —
*why not fine-tune the generator?* — which needs an answer in the paper rather than a
footnote. The answer is that a fine-tuned generator cannot exceed its training set, which is
the premise the coverage arm exists to test; but that answer has to be argued, not assumed.

## §12.17 What is confirmatory, what is novel, and what contradicts

### Claim none of this as new

- **Synthetic-only far below real at matched budget.** Our −27 sits inside a published −7 to
  −37. The dermatology precedent is Akrout et al. (DGM4MICCAI 2023): 500 synthetic score
  47.29 against 54.05 for 500 real, and lose to *250* real.
- **Null augmentation when real data is adequate.** Our +1.0 ± 2.1 is Sagers et al. (2023) at
  228 real per class, where every confidence interval spans zero.
- **Gains concentrate in scarce classes.** **Schaudt et al. (Bioengineering 10:1421, 2023)
  state this almost verbatim, with five seeds, in a medical journal.** §12.14's framing of
  that as the finding was wrong; it is the setup.
- **Saturation around 10:1.** Six papers give an exchange rate.

### The genuinely new pieces

1. **The decomposition itself.** +3.1 = +0.9 rebalancing + +2.2 content, measured against a
   duplicate-real-tail control at matched counts. **Sagers et al. (2023) §4.6 built exactly
   this control** — "as a control, we also tested an upsampling strategy of simple
   duplication" — **and put the result in a supplementary figure absent from the released
   PDF, never discussed in the text.** Schaudt's oversampling control collapses a class to
   F1 = 0.0000 and they never subtract. Ktena et al. compare against oversampling on the
   *sensitive attribute*, not class counts. The number has not been published.
2. **The compute-matched two-stage control on the pretraining arm.** Moroianu et al. (2025),
   the flagship synthetic-pretraining result in medical imaging at +6.5%, initialise their
   baselines from ImageNet and their synthetic arm **from scratch** — so content and
   two-stage training are confounded. Our null is not a contradiction of their result; it is
   the control they are missing, run at smaller scale.
3. **Quantified ordinal-severity compression** (§12.15).
4. **First evaluation of Gemini 2.5 Flash Image as a medical training-data generator.**

### Contradictions to address head-on

- **Khosravi et al. (eBioMedicine 104:105174, 2024)** is the real one: synthetic-only matched
  real *exactly* on CheXpert, AUROC 0.783 vs 0.783, p = 0.98. Closed-set DDPM trained on the
  same 72,000 images, a **real** tuning set retained even in the synthetic-only arm, and a
  multi-label detection task rather than an ordinal grading task where the class signal *is*
  lesion count. That last point is our own compression argument, which is a useful internal
  consistency rather than a coincidence.
- **Zein et al. (PLOS ONE 19(4):e0297958, 2024)** report synthetic-only at **97.6%** on real
  acne images — above every real-trained ACNE04 result ever published. Treat as
  uninterpretable rather than contradictory: the StyleGAN2 was trained on all 1,473 real
  images with no generator/test disjointness claimed, test-set size and balance are stated
  nowhere, severity labels were assigned by the authors sorting generator outputs, and the
  three backbones span 21.6 points on identical data. **This project currently cites that
  97.6% figure as a real result. That must be requalified.**
- **Schmidt et al. (2026)** find their classifier scores *better* on generated than real
  images — dominant-class exaggeration, opposite in sign to compression. The two can coexist
  as "prototypical moderate acne", but it is testable from our own data and should be tested.

### The scoop risk that must be cited

**ACNEDIT** (Piat et al., DGM4MICCAI 2025) generates **218 severe and 271 very-severe**
synthetic images to raise ACNE04's two scarce classes from 182 and 129 to 400 each. That is
this project's targeted-tail intervention, on this dataset, on these two classes. It is
evaluated by a user study and segmentation IoU, with no classifier balanced accuracy, no
synthetic-only arm and no rebalancing control, and it is a lesion-compositing GAN rather
than a prompted text-to-image model. The contribution survives — but the paper must cite it
and draw the distinction explicitly rather than be caught by it.

### On the tail result wobbling

If +2.2 does not survive ten seeds, **the paper is stronger.** A clean four-way null —
substitution fails, all-class augmentation null, pretraining null, tail augmentation null
once rebalancing is subtracted — together with a measured mechanism for *why*, is more
coherent and more defensible than a two-point positive. **The decomposition is the
contribution either way; the sign of the content term is a finding, not a requirement.**

---

## §13 The wide-domain pool, delivered in stages

Every prompt in the first pool was a human face, and §12.9 measured what that cost: four
clinical grades rendered as ~1.1 grades of real variation. A face is a strong prior. Asked
for confluent nodulocystic disease, a model trained on millions of portraits has every
reason to return someone who still looks like a person.

The wide-domain pool removes the constraint. Acne is requested on 24 substrates — faces,
cheeks, backs, chests, dermatoscope fields, macro fields with scale bars, isolated skin
swatches, excised specimens on surgical drapes, sections pinned to dissection boards,
**samples in Petri dishes**, tissue-culture plates, bioengineered constructs in culture
wells, specimen jars, silicone training models, wax moulages, 3D renders, textbook plates,
anatomical diagrams, prosthetic limbs — crossed with 4 severities, 6 Fitzpatrick tones, 4
ages, 2 sexes, 7 lighting conditions, 6 capture styles and 6 backdrops.

**Substrate is randomised independently of severity.** Otherwise the pool would carry the
provenance confound the audit documented in ACNE04, where 87.1% of mild images but 16.3% of
very-severe ones came from one camera, and a classifier would learn substrate instead of
disease.

### Staged, with a kill criterion at each gate

The full 6,000 is 95 hours and $234 at the cheap tier's measured 63 images/hour — a
concurrency of 6 was tried and fails with 429s that burn 281 seconds and six attempts per
image. Paying that before knowing whether the premise holds would be a bad trade, so the
pool is delivered in stages against an already-running job whose every prefix is balanced
by construction.

| stage | n | cost | time | question | kill criterion |
|---|---|---|---|---|---|
| **1** | 240 | $9 | 4 h | Does moving off the face widen the severity range? | no substrate beats Scope 36.3% |
| **2** | 650 | $25 | 10 h | Does wide-domain beat face-only at matched budget on real validation? | no gain over the 644-image face pool |
| **3** | 2,500 | $98 | 40 h | Learning curve, per-substrate ablation, tail augmentation at scale | curve flat, or best substrate is just "face" |
| **4** | 6,000 | $234 | 95 h | Full pool | only if 1–3 all pay |

Stage 1 needs **no training at all** — it applies the existing real-trained grade scorer to
generated images and reports Continuity and Scope per substrate. That makes the decisive
measurement the cheapest one, which is the right way round: if the face prior is not what
compresses severity, nothing downstream is worth paying for and the honest move is to say
so at $9 rather than $234.

---

## §13.1 The tail effect is not resolvable, and neither is most of the literature

§12.13 reported +2.2 points for generated tail images over a duplicate-real control, with
t ≈ 3.9 from two seeds. Six paired seeds:

| | value |
|---|---|
| mean paired difference | **+0.0224** |
| standard deviation | 0.0358 |
| t (5 df) | +1.53 |
| **p** | **0.185** |
| 95% CI | **[−0.015, +0.060]** — spans zero |
| seeds positive | **3 of 6** |

**The point estimate barely moved** — +0.0221 at two seeds, +0.0224 at six — while the
uncertainty around it grew by a factor of five. That is exactly what happens when a standard
deviation is estimated from n=2 and then trusted: the t of 3.9 was an artefact of the
variance estimate, not of the effect. Per-seed differences are −0.005, +0.063, −0.011,
−0.008, +0.030, +0.066: half the runs show nothing and half show a large gain.

### How many seeds this would actually take

| power | seeds needed |
|---|---|
| 80% | **22** |
| 90% | 29 |

Even the ten-seed run now finishing lands at t ≈ 2.0, p ≈ 0.08 — still not resolved. An
effect of this size against this variance needs roughly **twenty-two independent training
runs** to detect reliably.

### Which is the finding

**The published literature reports synthetic-augmentation gains of exactly this magnitude —
+1 to +3 balanced-accuracy points — routinely, and usually from a single training run.**
Sagers et al. (2023) at 228 real per class: −2.2 to +2.0, every interval spanning zero.
Noriega Cedeño (2026): +1.02, surviving only after Holm correction. Sundaram & Hulkund
(2021): single run, no error bars. Wang et al. (2024): n=56 test images, no repeats.

Bissoto, Valle & Avila (CVPR-W 2021) already said this out loud after ten runs per cell on
ISIC 2019 — *"performance improvement, when it happened at all, was completely random: the
choice of GAN model or other factors had no explanatory power"* — and listed as their fourth
named flaw in the literature *"ignoring performance fluctuations, e.g., by performing a
single run, or by failing to report the deviation statistics."*

This work supplies the number that critique lacked. **At the seed counts the field actually
uses, a +2-point synthetic-augmentation gain is indistinguishable from noise, and it takes
about twenty-two runs to tell them apart.** That is a stronger and more useful claim than
the +2.2 would have been.

### What is *not* being claimed

The effect is **underpowered, not disproven.** The point estimate is positive and stable
across two independent runs on different pool sizes, and three of six seeds show a large
gain. It may well be real. What cannot be done is to assert it from this evidence — or, by
the same arithmetic, from the evidence most published papers present.

### The decomposition survives regardless

The contribution was never the sign of the content term. It is that **the gain from adding
generated images to a scarce class can be split into a rebalancing component and a content
component, by adding duplicated real images at matched counts** — a control Sagers et al.
built in 2023 (§4.6) and left in an unreleased supplementary figure, Schaudt et al. ran with
a degenerate baseline that collapsed a class to F1 = 0.0000, and Ktena et al. applied to the
sensitive attribute rather than to class counts. The method stands whether the content term
lands at +2.2, at zero, or below.
