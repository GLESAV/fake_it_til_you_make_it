# §14 The classical-ML arm

## Why this arm exists

The project's question is "can generated images train a traditional classifier?" Every
classifier result so far answers it for a **supervised deep** classifier: ResNet-50, one
frozen recipe, learning its own representation. The phrase has a second, older reading —
a fixed feature extractor feeding an SVM, a random forest, or a gradient-boosted ensemble —
and that reading has not been tested anywhere in this repository.

It is worth an arm rather than a footnote because of the mechanism in §12.9. The generator
compresses four severity grades into **41% of the real severity range** (Continuity 70.4%,
Scope 36.3%, against a real ceiling of 88.4% / 93.1%). A deep network can partly route
around a compressed input distribution by learning features that magnify whatever variation
survives. A classical head over frozen features cannot — it sees the compression directly.

**Registered prediction, written into `scripts/classical_baseline.py` before the run:** the
substitution deficit should be *larger* here than the deep study's −26.6 points. If it comes
back *smaller*, the compression story is wrong in a way that matters, because the deficit
would then be something the deep model's learned features are creating rather than
overcoming.

Honest sequencing note, per practice R3: a 2-seed probe was run before the 30-seed run
below, so the 30-seed result was not seen blind. The prediction predates both.

## Design

Arms are built by the same construction as `scripts/train_on_synthetic.py` — the two
scripts share `Corpus`, `Record` and the resampling helpers — so the **only** thing that
varies between the deep study and this one is the classifier. Budgets stay matched, the
sealed test split is untouched, and every arm is scored on the same real subject-disjoint
validation split (218 images).

| arm | construction |
|---|---|
| `real` | the real training split, 948 images |
| `synthetic` | the frozen 644-image pool, class-balanced |
| `real_balanced` | real, resampled to the pool's class shape — isolates rebalancing |
| `mixed_tail` | real + generated images for classes 2–3 only |
| `mixed_tail_control` | real + **duplicated real** tail images, same count, same classes |

Five heads, all with `class_weight="balanced"` (the arms differ in class distribution by
design; an unweighted head would report the rebalancing rather than the images): logistic
regression, linear SVM, RBF SVM, random forest, histogram gradient boosting.

### Two feature regimes, never pooled

**`handcrafted`** — 234 dimensions: RGB and HSV histograms (32 bins per channel), Lab colour
moments, uniform LBP at two scales, Sobel gradient statistics. Nothing in it has ever seen a
photograph other than the ones passed in. This is the only regime that is honestly
*classical*, and it is the one the headline uses.

**`embedding`** — the cached 512-d vectors in `data/splits_subject/`. Stronger, but those
weights consumed a large corpus of real photographs, so a "100% synthetic" arm built on them
is not 100% synthetic. This is the same objection the study already makes to
ImageNet-pretrained backbones (README design commitment 4) and it is honoured the same way:
separate tables, no pooling. **The cache currently covers the real corpus only**, so this
regime cannot score synthetic arms until it is extended; the script fails loudly rather than
silently dropping them.

### What this arm cheaply buys

Classical heads fit in milliseconds. The seed counts that are prohibitive for the deep tail
arm — 22 for 80% power, §13.1 — are free here, so the paired content effect can be run at a
sample size the deep study cannot afford. That does **not** resolve the deep arm. It
measures a different classifier on the same data.

## Results

30 seeds, handcrafted features, balanced accuracy on the real subject-disjoint validation
split. Full output in `results/classical_arm.json`.

| arm | logreg | linsvm | rbfsvm | rf | hgb |
|---|---|---|---|---|---|
| real | 0.465 | 0.432 | 0.485 | 0.516 | 0.533 |
| synthetic | 0.429 | **0.468** | 0.297 | 0.370 | 0.453 |
| real_balanced | 0.454 | 0.431 | 0.475 | 0.498 | 0.504 |
| mixed_tail | 0.510 | 0.451 | 0.579 | 0.525 | 0.574 |
| mixed_tail_control | 0.466 | 0.443 | 0.496 | 0.509 | 0.526 |

### The registered prediction failed on its stated terms

The prediction was that the substitution deficit would exceed the deep study's 26.6 points.
It did not, for any head: the largest is 18.8 points (RBF SVM) and the mean across heads is
8.3. **The prediction is recorded as falsified**, in line with practice R3 and with this
project's record of three-for-three failed causal claims (audit §15.1).

A re-normalisation partially rescues it, and is reported here with the caveat that **it was
chosen after seeing the result and is therefore weak evidence, not a defence.** A deficit is
bounded by how far the real baseline sits above chance (0.25 for four balanced classes), and
these baselines are far weaker than the deep study's:

| head | real | synthetic | deficit | headroom above chance | deficit as % of headroom |
|---|---|---|---|---|---|
| logreg | 0.465 | 0.429 | −0.036 | 0.215 | 16.6% |
| linsvm | 0.432 | 0.468 | **+0.036** | 0.182 | **−19.8%** |
| rbfsvm | 0.485 | 0.297 | −0.188 | 0.235 | **79.9%** |
| rf | 0.516 | 0.370 | −0.146 | 0.266 | 54.9% |
| hgb | 0.533 | 0.453 | −0.080 | 0.283 | 28.4% |
| *deep (§12.8)* | *0.734* | *0.468* | *−0.266* | *0.484* | *55.0%* |

Only one head of five exceeds the deep study's relative deficit. That is not the clean
confirmation the prediction asked for, and it is not reported as one.

### Linear SVM inverts, and that is a prototype-effect candidate

The linear SVM is the one head where synthetic **beats** real, +3.6 points on 28 of 30
seeds. It also has the weakest real baseline of the five (0.432, only 0.182 above chance).
The obvious reading is the one protocol §8.6 exists to catch: a generator that narrows
within-class variation produces cleaner prototypes, which a weak linear head finds easier to
separate — a win that does not require the generator to have contributed anything. **Per
README design commitment 4 and protocol §8.6, this is not reported as a win until the
prototype-effect checks are run.** They have not been run.

### The paired content effect is positive on every head — with a large caveat

| head | effect | seed sd | t | p | seeds positive |
|---|---|---|---|---|---|
| logreg | +0.0446 | 0.0085 | +28.8 | <0.001 | 30/30 |
| linsvm | +0.0082 | 0.0133 | +3.4 | 0.002 | 23/30 |
| rbfsvm | +0.0832 | 0.0093 | +48.9 | <0.001 | 30/30 |
| rf | +0.0164 | 0.0221 | +4.1 | <0.001 | 22/30 |
| hgb | +0.0480 | 0.0227 | +11.6 | <0.001 | 29/30 |

Five heads, all positive, all significant, against the control that duplicates real tail
images. Rebalancing alone (`real_balanced` minus `real`) is null-to-negative on every head,
so the effect is not the rebalancing.

**The caveat is the size of those t-statistics, not their sign.** The `real` arm is
deterministic across seeds for four of five heads; only the synthetic draw varies. So the
seed sd measures sensitivity to *which generated images were drawn* — a real and useful
quantity, and the one under a practitioner's control — but it is not the uncertainty a
reader will assume a *p* < 0.001 refers to.

A second route makes the scale clear. The validation split is 218 images with 72/102/26/18
per class; the worst-case standard error of balanced accuracy on a *fresh cohort of the same
size* is **4.3 points**. Four of the five effects are smaller than that; only the RBF SVM's
+8.3 is meaningfully larger. The paired test remains valid for the question it asks — "on
this split, does `mixed_tail` beat its control" — because pairing on identical test items is
what removes split-sampling noise from the comparison. It does not license "this will
replicate on a new cohort of patients."

**The same arithmetic applies to the deep study's tail effect of +1.65 points**, which sits
well below the same 4.3-point floor. That is an independent second argument for the position
§13.1 already takes: underpowered, not disproven.

## Reading limits, stated up front

1. **This does not transfer to the deep result and must not be pooled with it** (practice
   R2). Two classifiers, two questions.
2. **The `real` arm is near-deterministic across seeds.** The seed varies *which* generated
   images are drawn; logistic regression and the SVMs on a fixed real corpus return the same
   fit every time. So the paired contrasts' variance comes almost entirely from the
   synthetic draw, and the paired t-statistics are effectively one-sample tests on that
   draw. This is the right question to ask — it is the variation under a practitioner's
   control — but it is not the same uncertainty the deep study's seed-to-seed spread
   reports, and the two intervals are not comparable.
3. **Five heads is five chances to find something.** Any per-model effect is unadjusted for
   multiplicity. The claim to make is about the pattern across heads, not about whichever
   head reports the smallest *p*.
4. Handcrafted features are weak in absolute terms. Absolute accuracies are not comparable
   to the deep study's; only the *contrasts between arms* are.

## Reproducing

```bash
python scripts/classical_baseline.py --features handcrafted --seeds 30
python scripts/classical_baseline.py --features embedding   --seeds 30   # real arms only
```

Features are cached to `data/features_handcrafted.npz` on first run (~75 s for 1,810
images); subsequent runs are seconds per seed.
