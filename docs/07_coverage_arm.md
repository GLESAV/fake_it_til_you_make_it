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
