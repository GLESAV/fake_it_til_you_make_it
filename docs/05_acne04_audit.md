# A data-quality audit of ACNE04

**Run 2026-08-19** on the archive distributed from `github.com/xpwu95/LDL`
(`Classification.tar`, 1.13 GB, SHA of the extracted image set recorded in
`data/splits/dedup_report.json`). Every number below is reproducible from the scripts
in this repository against that archive; none of it requires a model, a GPU, or a
judgement call.

We went looking for near-duplicates because our own splitter refused to proceed
(§2). We found something larger, and it bears on every published number on this
benchmark.

---

## Summary

| Finding | Measurement |
|---|---|
| Byte-identical duplicate images | **38 groups, 76 of 1,457 files (5.2%)** |
| Duplicate groups with **conflicting severity labels** | **8 of 38 (21%)** |
| Duplicate groups with **conflicting lesion counts** | **26 of 38 (68%)** |
| Grade agreement between two annotations of the *same image* | **78.9%** (95% Wilson CI **63.7–88.9%**) |
| Exact lesion-count agreement on the same image | **31.6%** |
| Mean train/test leakage in the **dataset's own published 5-fold splits** | **4.25%** of each test set |
| — of which free correct answers (labels agree) | 3.56% |
| — of which forced errors (labels conflict) | 0.68% |
| Files whose `levleN` filename prefix disagrees with the label | **42 of 1,457** |
| **Very-severe images not taken on the dataset's dominant camera** | **83.7%** (against 12.9% for mild) |
| — severity predicted from the file header alone | **sensitivity 0.57 at specificity 0.85** |
| **What image-level splitting is worth** (matched pair, 5 seeds) | **+4.51 balanced-accuracy points** (95% CI +2.33 to +7.04) |
| — on the two majority grades | −1.5 and +0.8 points |
| — on severe and very severe | **+9.9 and +9.0 points** |
| **Distinct individuals in the 1,457 images (ArcFace)** | **~550–750** |
| **The same leak in SCIN, from published metadata alone** | **75.6%** (against ACNE04's 77.5%) |
| **And in HAM10000** | **38.5% overall — but 68.3% on melanoma** |
| — test images sharing a *person* with training, published folds, cosine ≥ 0.85 | **15.9%** (chance rate 0.019%) |
| — the same at the ordinary same-identity operating point (0.60) | **77.5%** |
| **Independent expert re-annotation (ACNE04-v2, n=1,204 shared images)** | |
| — lesions counted per image, v1 vs v2 | **8.4 vs 26.9 (3.2× more)** |
| — Spearman *rho* between the two counts | **0.471** |
| — grade agreement under ACNE04's own labelling function | **30.1%** (QWK 0.296) |
| — grade agreement after optimal monotone rescaling | **58.2%** (QWK 0.480) |
| — share graded severe or worse, v1 vs v2 | **11.7% vs 46.4%** |
| Published accuracy on this benchmark | **83.7–87.3%** |

Two things are the point.

**Reported ACNE04 accuracies sit at or above the dataset's own annotation
self-consistency** (78.9%), measured on images the dataset annotated twice without
realising it.

And more seriously: **the severity label is a property of the annotation team, not of
the photograph.** A second expert team re-annotating the same 1,204 images counts 3.2×
more lesions, ranks the images differently (Spearman *rho* 0.471), and — after being
granted any monotone recalibration it likes, which removes the entire definitional
component — still agrees with the original grade on only 58% of images. An ACNE04
accuracy figure measures agreement with one team's counting convention. §4.

![ACNE04 audit](examples/acne04_audit.png)

---

## 1. The dataset contains 38 exact duplicates

Grouping all 1,457 files by MD5 of the file bytes yields 38 groups of size 2 — 76
files, 5.2% of the corpus. These are not near-duplicates or re-encodes; they are the
same bytes under two filenames, and in most cases under two *different* filename
prefixes (`levle0_120.jpg` and `levle1_486.jpg`).

Decoding to pixels and hashing again yields exactly the same 38 groups, so there are
no additional re-encoded copies hiding behind different compression.

## 2. Generic perceptual and semantic embedders cannot find them safely

Our splitter is group-aware: it refuses to break a duplicate cluster across splits.
That makes it sensitive to over-clustering, and on ACNE04 with the shipped default
(pHash Hamming ≤ 8) it produced a single cluster holding **214 images, 14.7% of the
corpus**, and the guard fired.

Sweeping the threshold shows a percolation transition rather than a plateau:

| pHash Hamming ≤ | linked pairs | groups | largest cluster |
|---|---|---|---|
| 0 | 38 | 1,419 | 2 (0.1%) |
| 2 | 42 | 1,415 | 2 (0.1%) |
| 4 | 62 | 1,397 | 5 (0.3%) |
| 6 | 193 | 1,281 | 31 (2.1%) |
| **8** (shipped default) | **720** | **1,001** | **214 (14.7%)** |
| 10 | 2,567 | 675 | 388 (26.6%) |
| 12 | 7,256 | 480 | 922 (63.3%) |

Visual inspection of sampled pairs at each distance settles what the numbers imply:
at Hamming 0 the pairs are the byte-identical duplicates; at 4 roughly a third are the
same subject and the rest are different people; at 6 and beyond they are overwhelmingly
different people who happen to share the dataset's single pose. **pHash is matching the
ACNE04 capture protocol — half-face at ~70°, dark background, similar lighting — not
duplication.** Every image in this dataset looks like every other image to a perceptual
hash.

CLIP ViT-B/32 fails the same way for the same reason. The median pairwise cosine
across the whole corpus is **0.83**, and percolation starts at 0.96:

| CLIP cosine ≥ | linked pairs | groups | largest cluster |
|---|---|---|---|
| 0.99 | 40 | 1,417 | 2 (0.1%) |
| 0.98 | 58 | 1,400 | 3 (0.2%) |
| 0.97 | 255 | 1,246 | 35 (2.4%) |
| 0.96 | 1,448 | 844 | 405 (27.8%) |
| 0.95 | 6,335 | 525 | 688 (47.2%) |

Calibration point: all 38 byte-identical pairs sit at CLIP cosine **1.0000**, so 0.99
recovers exactly the exact duplicates plus two extra links.

**Consequence for our protocol.** We set `phash_max_hamming: 2` and
`embed_min_cosine: 0.98`, which together give 1,400 groups with a largest cluster of 3
(0.2%) — comfortably inside the 5% guard — and we state the residual risk plainly:
*neither instrument detects same-subject-different-photograph.* We ran that analysis
separately with a face-identity embedding, and it turned out to be the largest finding
in this document -- see §5.

## 3. The published 5-fold splits leak, in every fold

ACNE04 ships fixed splits (`NNEW_trainval_{0..4}.txt` / `NNEW_test_{0..4}.txt`),
1,165 train / 292 test each. Intersecting the 38 duplicate groups against each fold:

| Fold | duplicate groups straddling train/test | leaked test images | free correct | forced errors |
|---|---|---|---|---|
| 0 | 12 | 12 (4.1%) | 7 (2.4%) | 5 (1.7%) |
| 1 | 15 | 15 (5.1%) | 15 (5.1%) | 0 |
| 2 | 12 | 12 (4.1%) | 10 (3.4%) | 2 (0.7%) |
| 3 | 11 | 11 (3.8%) | 11 (3.8%) | 0 |
| 4 | 12 | 12 (4.1%) | 9 (3.1%) | 3 (1.0%) |
| **mean** | | **4.25%** | **3.56%** | **0.68%** |

Across all five folds, 62 distinct images appear in some test split while a
byte-identical copy sits in the corresponding training split.

A model that memorises its training set therefore receives, on average, **3.56% of the
test set for free** and is **guaranteed to fail on another 0.68%** where the duplicate
carries a different label. Deep networks memorise small training sets readily, so the
free fraction is not hypothetical. This does not explain the whole 83.7–87.3% band, but
it is a systematic upward bias of a few points sitting inside every number ever reported
on these splits, and no paper we verified mentions it.

## 4. Two expert teams, the same 1,204 photographs, and a 3.2x difference in lesion count

The duplicates give a within-release reproducibility estimate on 38 images. A second,
independent measurement is available and is far better powered.

The AcneAI team (Gazeau, Nguyen et al., MICCAI 2024) re-annotated ACNE04 and released
**ACNE04-v2**: 1,204 of the original images with **32,443 lesion annotations**, against
the original's 18,983 across 1,457 images. Original filenames are preserved, so the two
annotations can be joined image by image. All 1,204 v2 filenames are present in v1.

| | v1 (Wu et al., ICCV 2019) | v2 (AcneAI, MICCAI 2024) |
|---|---|---|
| lesions per image, mean | 8.4 | **26.9** |
| lesions per image, median | 6 | **19** |
| maximum | 64 | **434** |
| images graded severe or worse | 11.7% | **46.4%** |

**v2 finds 3.2x more lesions on average, and more on 90.1% of the shared images.**
Pearson *r* between the two counts is **0.550**; Spearman *rho* is **0.471**.

### 4.1 Part of this is definitional, and that part is legitimate

The two teams were not counting the same thing, and both are internally reasonable.

Hayashi et al. (*J Dermatol* 35:255-260, 2008) established the grading criterion ACNE04
uses by correlating dermatologists' global severity judgements against lesion counts on
half the face. The bands are 0-5 mild, 6-20 moderate, 21-50 severe, >50 very severe,
and they are defined over **inflammatory eruptions -- papules plus pustules**. The paper
is explicit that global severity **did not correlate with the number of comedones**,
which is why comedones are excluded.

AcneAI states a deliberately different protocol: they annotate "all lesions, either acne
or non-acne lesions that look like acne," including "all acne lesions, even the tiny
ones," with comedones named explicitly among the small lesions marked with a circle.

So a large gap is expected. **The problem is not that the teams disagree; it is that
ACNE04's release specifies a count and a grade without specifying which lesions were
counted**, so a second expert team could not reproduce the first team's numbers even
approximately, and neither the dataset nor any downstream paper we verified states the
lesion definition the labels depend on.

### 4.2 The definitional part does not explain the disagreement

If the difference were purely a matter of counting additional lesion types, the ratio
would be roughly constant and the *rank order* of images by severity would be preserved.
Neither holds.

The per-image ratio has a median of 3.50 but an interquartile range of 2.00-6.64 and a
10th-to-90th-percentile span of **11.5x**. On **8.3% of images v2 found fewer lesions
than v1**. The ratio is also strongly count-dependent:

| v1 count band | n | median v2/v1 ratio |
|---|---|---|
| 1-2 | 345 | **8.00** |
| 2-5 | 310 | 5.50 |
| 5-7 | 264 | 3.00 |
| 7-11 | 360 | 2.65 |
| 11-64 | 272 | **2.00** |

To remove the definitional component entirely, we allow an *arbitrary monotone
rescaling*: re-band the v2 counts by quantile matching so that the v2 grade
distribution is identical to v1's by construction. Any residual disagreement cannot be
attributed to "they counted more lesion types."

| | grade agreement | quadratic weighted kappa |
|---|---|---|
| v2 counts under ACNE04's own Hayashi banding | **30.1%** | 0.296 |
| v2 counts after optimal monotone rescaling | **58.2%** | **0.480** |

After the rescaling, 58.2% of images get the same grade, **38.5% differ by one grade
and 3.2% by two or more**. A weighted kappa of 0.48 is moderate agreement. Restricting
to images with more lesions -- where relative counting noise should be smaller -- does
not help: Spearman *rho* is 0.376 on images with v1 count >= 5 and 0.297 on those with
count >= 20. (Range restriction attenuates *rho*, so these are not directly comparable
to the full-sample 0.471; the point is only that the disagreement does not vanish on the
images where counting should be most stable.)

### 4.3 What this means for the benchmark

Applying ACNE04's own labelling function to a second expert team's counts changes the
grade of **70% of the images** and moves the share of severe-or-worse cases from 11.7%
to 46.4%. Even after granting the second team any monotone recalibration they like,
the two labellings agree on 58%.

A model trained and evaluated on ACNE04 therefore predicts *the grade that Wu et al.'s
counting protocol assigns*, not acne severity in any transferable sense. That is a
perfectly valid object of study, but it is not what the benchmark is used to claim, and
it bounds what an ACNE04 accuracy figure can support:

- **Cross-dataset and clinical-deployment claims are unsupported by ACNE04 accuracy
  alone.** The label does not transfer to another expert team's reading of the same
  photographs.
- **Differences of one to three accuracy points between methods are not interpretable**
  as differences in severity assessment. They are differences in fitting one team's
  counting threshold.
- **The tail is where it is worst.** v1 grades 11.7% of shared images severe or worse;
  v2's counts put 46.4% there. Per-class results on the severe grades, which is where
  the clinical value sits, rest on the least stable part of the labelling.

### 4.4 Limits of this comparison

We compute v2 grades by applying the Hayashi banding to v2's counts. **AcneAI does not
do this** -- they use a continuous 0-100 severity score built from per-lesion severity
and area, and report ICC 0.8 for their own method. Applying Hayashi bands to their
counts is our construction, not theirs, and it is clinically incorrect on its face
precisely because their counts include comedones. That is the intended demonstration:
it shows how much the benchmark's published label depends on an unspecified counting
convention. It is not a criticism of AcneAI's annotations, which are more complete than
v1's by design and by their own account.

We also cannot separate "team A missed lesions" from "team B over-segmented" without a
third annotation. The claim here is symmetric and does not require assigning fault:
**two expert annotations of the same 1,204 photographs do not induce the same severity
ordering**, and the benchmark provides no basis for preferring one.

## 5. ACNE04 is about 600 people, not 1,457 photographs — and every published split ignores that

Section 2 flagged that neither perceptual hashing nor CLIP detects the same subject
photographed twice, and that ACNE04's capture protocol makes repeat subjects plausible.
It does more than that.

We embed every image with ArcFace (InsightFace `buffalo_l`). One practical note first,
because it is a trap: **face detection fails on 91% of ACNE04 raw**. These are close
crops in which the face fills the frame, and RetinaFace expects a face to occupy a
fraction of the image. Padding the border by 60% of the long side raises detection from
**9.3% to 96.1%** (1,400 of 1,457 images). Any pipeline that runs a face detector over
this dataset without padding will conclude, silently and wrongly, that it contains
almost no faces.

### 5.1 The identity structure

Union-find over ArcFace cosine gives, across a wide and stable threshold band:

| cosine ≥ | subjects | images in a multi-image subject | largest subject |
|---|---|---|---|
| 0.75 | 753 | 886 (63%) | 21 |
| 0.70 | 638 | 1,037 (74%) | 21 |
| 0.65 | 577 | 1,132 (81%) | 22 |
| **0.60** | **550** | **1,177 (84%)** | **22** |
| 0.50 | 525 | 1,206 (86%) | 22 |
| 0.45 | 482 | 1,228 | 111 ← percolates |

The structure is stable from 0.75 down to 0.50 and collapses at 0.45, which is the
signature of a real clustering rather than a threshold artefact. **ACNE04's 1,457
images are on the order of 550–750 distinct individuals**, most photographed two or
three times — different angles, sessions and lighting. Visual inspection of the largest
clusters confirms this directly: they are unmistakably one person each.

The 38 byte-identical pairs sit at cosine 1.000, as they must, which calibrates the
instrument.

### 5.2 Leakage, stated as conservatively as we can

Transitive closure can chain, so we report the version that cannot: a test image counts
as leaked only if **it is itself above threshold to some individual training image**.
No clusters, no chaining. Alongside it we give the chance rate — the share of *all*
image pairs in the corpus above that threshold — so the false-positive budget is visible.

| cosine ≥ | chance rate | fold 0 | 1 | 2 | 3 | 4 | **mean** |
|---|---|---|---|---|---|---|---|
| 0.85 | 0.019% | 12.1 | 19.1 | 16.6 | 14.1 | 17.5 | **15.9%** |
| 0.80 | 0.058% | 33.2 | 43.3 | 37.9 | 33.1 | 38.2 | **37.1%** |
| 0.75 | 0.120% | 53.9 | 63.8 | 58.5 | 51.8 | 59.3 | **57.5%** |
| 0.70 | 0.191% | 65.0 | 71.3 | 70.0 | 59.9 | 71.1 | **67.4%** |
| 0.60 | 0.273% | 74.3 | 83.7 | 80.5 | 70.1 | 78.9 | **77.5%** |

At cosine 0.85, where fewer than one pair in five thousand qualifies by chance,
**15.9% of every published ACNE04 test fold is a photograph of somebody who also
appears in the training fold** — roughly 800× the chance rate. At the ordinary
same-identity operating point the figure is 57–78%.

This dwarfs the byte-duplicate leakage of §3. It is not a handful of images: it is most
of the test set.

### 5.2.1 Negative control: is this identity, or shared capture conditions?

The obvious objection to any embedding-based leakage claim is that the model may be
matching the photo shoot rather than the person — same camera, same lighting, same
background. On this corpus that objection has teeth, because 76% of the images share one
resolution and **97.8–99.7% of the linked pairs sit inside a single resolution group**
against a 63.2% chance baseline, with only 4% of identity clusters spanning more than one
resolution.

That pattern has a benign reading (one person, one session, one camera) and a fatal one
(the model is scoring capture conditions). They are separable: restrict the analysis to
the dominant resolution group — 1,107 images, one camera, one setup — and ask whether the
separation survives. If capture conditions drove the similarity, the distribution inside
that group would be shifted high and unimodal.

| within the 1,107-image group | value |
|---|---|
| median pairwise cosine | **0.103** |
| 90th / 99th percentile | 0.220 / 0.342 |
| share of pairs ≥ 0.60 | **0.423%** |
| share of pairs ≥ 0.85 | **0.029%** |
| subjects at cosine 0.60 | 319, largest cluster 22 |

The distribution stays low and the identity tail stays thin and distinct: within one
capture setup, fewer than half a percent of pairs clear 0.60. The clusters are people,
not photo shoots, and the resolution concentration is the benign reading — a subject was
photographed once, in one session, on one camera.

### 5.3 Our first splits leaked too

Our deduplicated image-level splits carry 8.3% / 50.7% / 74.5% at cosine 0.85 / 0.75 /
0.60 — no better than the published folds, because deduplication and identity grouping
answer different questions. Grouping by identity as well (604 subjects, largest 22
images) brings it to **0.0%** at cosine 0.60 and above, and the splits stay well
formed: 948 / 218 / 291 with every class present in every split.

One honest caveat: grouping at threshold *T* guarantees disjointness above *T* only.
At cosine 0.50 the subject-disjoint test split still shows 3.2% — pairs that fall
between 0.50 and 0.60 can straddle the boundary by construction. Dropping the grouping
threshold removes those but coarsens the grouping, and below about 0.45 the clustering
percolates. We take 0.60 and report the residual rather than choosing a threshold that
makes the number zero.

### 5.4 Why this matters more than the duplicate count

Acne severity is graded from lesion counts on *half* the face, so two photographs of one
person from different sides legitimately carry different grades. The leakage is
therefore not simple label copying, which would be easy to dismiss. It is worse in a
subtler way: a model can learn *this individual's skin, complexion, hair, background and
capture session* and carry that to the test set, where the same person appears with a
different lesion count. Nothing about that generalises to a new patient, which is the
only thing an acne grader is for.

Two consequences:

1. **Published ACNE04 accuracies measure within-subject generalisation, not
   between-subject generalisation.** They answer "given other photographs of this
   person, can you grade this one?" — not "given a new patient, can you grade them?"
   Only the second is clinically meaningful.
2. **The size of the effect is measurable, and we measure it.** §7 reports the same
   classifier, the same hyperparameters, the same budget, trained and evaluated on
   image-level splits and on subject-disjoint splits. The difference is the leakage
   bonus, in points.

### 5.5 How many people are in each grade

Per-class claims are bounded by the number of distinct people in a class, not the number
of photographs. Counting subjects per grade (on the 1,400 images that yield a face
embedding, so the image counts here are slightly below the corpus totals):

| grade | images | distinct people | images per person |
|---|---|---|---|
| mild | 509 | 221 | 2.30 |
| moderate | 627 | 300 | 2.09 |
| severe | 166 | **114** | 1.46 |
| very severe | 98 | **81** | 1.21 |

*(identity threshold 0.60; at the conservative 0.75 the subject counts are 280 / 376 /
129 / 91.)*

The correction is real but modest, and smaller in the tail than in the head — severe
cases are rarer, so they are photographed fewer times each. A 20% test split of the
severe grade holds roughly 23 distinct people rather than 33 independent images, and of
the very-severe grade roughly 16 rather than 20. **We flag this rather than lean on it:
the tail is noisy mainly because it is small, not mainly because it is redundant.**

The number worth carrying forward is different. **27% of individuals (149 of 550) have
images under more than one severity grade** — 15% even at the conservative threshold. For
a criterion counted on half the face that is partly legitimate: one cheek can genuinely
be worse than the other. But it means that for a quarter of the people in this dataset,
the severity label is not a property of the person, and a model that identifies the
person cannot infer the grade from that alone. It is the one piece of good news in this
section, because it puts a ceiling on how much the subject leakage of §5.2 can inflate a
score by identity recognition alone.

### 5.6 A property of group-aware splitting worth knowing about

Grouping by identity and then splitting to hit image proportions does **not** split the
people proportionally, and the gap is large:

| split | images | distinct identities | images per identity | share of images | share of identities |
|---|---|---|---|---|---|
| train | 948 | 267 | 3.43 | 65.1% | **48.5%** |
| val | 218 | 129 | 1.59 | 15.0% | **23.5%** |
| test | 291 | 154 | 1.81 | 20.0% | **28.0%** |

The mechanism is mechanical: a group-aware splitter fills the largest split first and
cannot break a group, so high-multiplicity subjects land in training and the smaller
splits end up holding many one-image subjects. Training holds 65% of the photographs but
under half of the people.

**We are recording this rather than correcting it, because its direction is favourable.**
A test set with more distinct people per image is a *better* estimator of
generalisation to a new patient — the images in it are closer to independent. The cost
falls on the training side, where the effective diversity is lower than the image count
suggests, which makes our real-only baseline if anything conservative.

It does matter for one thing downstream. **A closed-set generator fine-tuned on this
training split is learning from 267 distinct people**, and only 57 and 47 of them in the
two tail grades:

| grade | training images | distinct identities | effective identities |
|---|---|---|---|
| mild | 333 | 116 | 93.2 |
| moderate | 403 | 126 | 89.8 |
| severe | 126 | **57** | 46.9 |
| very severe | 86 | **47** | 40.4 |

That is the identity budget the synthetic pool has to work with, and it is the reason
§8.10 of the protocol measures identity diversity of every generated pool. A generator
asked to produce thousands of very-severe faces has 47 people to generalise from.

An alternative splitter that stratifies group assignment by group size would equalise the
subject proportions. We have not adopted it: it would change the frozen splits, and the
current asymmetry costs us nothing we want.

## 6. What the leakage is worth, in points

Two arms differing in exactly one thing: how the splits were drawn. Same ResNet-50, same
ImageNet initialisation, same hyperparameters, same 947/948-image training budget, same
five seeds, same corpus. One splits at the image level after perceptual and CLIP
deduplication — which is what every published ACNE04 result does, and what our own first
attempt did — and carries **74.5% subject leakage**. The other groups by face identity
first and carries **none**.

All numbers are validation; the test split has never been unsealed.

| metric | image-level | subject-disjoint | difference | 95% CI |
|---|---|---|---|---|
| **balanced accuracy** | 0.7920 ± 0.017 | 0.7469 ± 0.016 | **+4.51 pts** | **[+2.33, +7.04]** |
| macro F1 | 0.7515 ± 0.025 | 0.6945 ± 0.027 | +5.69 pts | [+2.37, +9.57] |
| quadratic weighted κ | 0.8487 ± 0.016 | 0.8221 ± 0.018 | +2.66 pts | [+0.46, +5.15] |
| plain accuracy | 0.7644 ± 0.018 | 0.7394 ± 0.025 | +2.49 pts | [−0.80, +5.79] |
| mean absolute error | 0.2511 ± 0.024 | 0.2817 ± 0.030 | −3.05 pts | [−7.63, +0.52] |

Paired by seed, bootstrap intervals over the paired differences.

**Splitting a face dataset at the image level is worth about four and a half points of
balanced accuracy on ACNE04** — a bonus larger than the entire spread between the
published methods competing on this benchmark (83.70%, 84.11%, 86.06%, 87.33%).

### 6.1 The bonus is not spread evenly, and where it lands is the argument

| grade | distinct people in training | image-level recall | subject-disjoint recall | difference |
|---|---|---|---|---|
| mild | 116 | 0.885 | 0.900 | **−1.5 pts** |
| moderate | 126 | 0.657 | 0.649 | **+0.8 pts** |
| severe | **57** | 0.637 | 0.538 | **+9.9 pts** |
| very severe | **47** | 0.990 | 0.900 | **+9.0 pts** |

The bonus is **essentially zero on the two majority grades and about ten points on the two
rare ones** — exactly the grades with the fewest distinct subjects to memorise. That is
what the mechanism predicts: recognising an individual is worth far more when the class
holds 47 people than when it holds 126.

It also matters for reading the aggregate row above. Plain accuracy is dominated by the
majority classes, where there is no bonus, so its interval spans zero. Balanced accuracy
and macro F1 weight the tail equally and show the effect clearly. **A benchmark reported
in plain accuracy will not show this leak even though it is there.**

### 6.2 The confound, and why the pattern survives it

The two arms have different test sets, so this comparison mixes the leakage bonus with
whatever intrinsic difficulty difference exists between those sets. We flag it rather
than hide it, and §6.4 reports the matched within-model analysis — which turns out to
carry a confound of its own, in the opposite direction.

But the per-class pattern is already hard to explain as a test-set artefact. A difficulty
difference between two random subject-disjoint splits has no reason to land specifically
on severe and very severe while being zero on mild and moderate. The mechanism predicts
exactly that shape; a confound would have to reproduce it by coincidence, in the two
classes the mechanism singles out in advance.

### 6.3 What it does not say

This is not a claim that published ACNE04 results are overstated by 4.5 points. Those
results use plain accuracy on the dataset's own folds, not balanced accuracy on ours, and
their leakage rate is 77.5% against our image-level arm's 74.5% — close, but not the same
experiment. The defensible statement is narrower and still substantial: **on this dataset,
with this architecture and budget, image-level splitting is worth +4.5 balanced-accuracy
points, concentrated almost entirely in the two clinically important grades.**


### 6.4 The within-model contrast, and why it is not the headline

The cross-arm comparison above has a confound we flagged: two different test sets. The
obvious remedy is to stay inside one model and one test set and partition the test images
by whether the person in them also appears in that model's training split. We ran it. It
has a confound of its own, and it is worth showing why, because the same trap is waiting
for anyone who repeats this on another patient dataset.

At identity threshold 0.60, on the image-level arm (three seeds, per-image predictions):

| | leaked | clean | difference | 95% CI |
|---|---|---|---|---|
| plain accuracy | 0.796 | 0.682 | +11.45 pts | [+10.47, +12.26] |
| balanced accuracy | 0.855 | 0.712 | +14.36 pts | [+12.13, +16.20] |

Consistent across all three seeds, tight intervals — and **inflated**, by roughly a factor
of three against the cross-arm estimate. The reason is structural:

| | leaked partition | clean partition |
|---|---|---|
| mild | 43.4% | 17.9% |
| moderate | 45.4% | 38.8% |
| severe | 8.6% | 20.9% |
| very severe | 2.6% | 22.4% |
| **severe + very severe** | **11.2%** | **43.3%** |

**Severity is inversely related to how many times a subject was photographed** — 2.30
images per person at mild, 1.21 at very severe (§5.5). A subject photographed once cannot
have a twin in training, so rare-grade images leak far less, and the clean partition ends
up enriched four-fold in the hard classes. A leaked-versus-clean split on this dataset is
therefore also a severity split, and the aggregate difference between them is part leakage
bonus and part class mix.

Holding class fixed is the only comparison the partition permits:

| grade | n leaked | n clean | recall leaked | recall clean | difference |
|---|---|---|---|---|---|
| mild | 66 | 12 | 0.889 | 0.889 | **+0.0** |
| moderate | 69 | 26 | 0.686 | 0.551 | **+13.5** |
| severe | 13 | 14 | 0.846 | 0.429 | **+41.8** *(n too small to quantify)* |
| very severe | 4 | 15 | 1.000 | 0.978 | *(n too small)* |

The two analyses agree on shape and disagree on magnitude, which is what you would expect
if each is biased in a known direction. **Zero bonus on mild, a substantial one on
moderate, and a large one on severe** — the same ordering the cross-arm comparison found,
arrived at by a different route with a different confound.

**We report the cross-arm +4.5 as the headline** because its confound (test-set
difficulty) has no reason to align with class, whereas the within-model confound
provably does. The within-model analysis is reported as corroboration of the mechanism,
not as a second estimate of its size, and the severe-grade number is deliberately left
unquantified: 13 images against 14 is not a measurement.


## 7. The published memorisation threshold does not survive contact with this corpus

This one is about instrumentation rather than about ACNE04, and it generalises to any
memorisation audit run on a homogeneous corpus.

Somepalli et al. (CVPR 2023) measure that **1.88% of random Stable Diffusion v1.4
generations exceed SSCD similarity 0.5** to a training image, and that number is the
reference point every subsequent memorisation claim is compared against. We adopted their
instrument and their threshold deliberately, so our rate would be comparable to theirs.

Then we calibrated it on ACNE04, and the threshold turned out to be unusable here.

**Positive control.** The 38 byte-identical pairs sit at similarity **1.0000**, min,
median and max. The instrument works.

**Negative control — pairs of genuinely distinct real images**, excluding those 38 pairs,
1,060,658 pairs in total:

| | |
|---|---|
| median similarity | 0.287 |
| 99th percentile | 0.566 |
| 99.99th percentile | 0.746 |
| **maximum between two distinct real images** | **0.9763** |
| pairs above 0.5 | **5.30%** |
| **real images with *some* other real image above 0.5** | **82.6%** |

**At the published threshold, 83% of real ACNE04 images are copies of another real
image.** They are not. They are different people photographed the same way.

The reason is not subtle once stated. SSCD's 0.5 was calibrated against LAION, where two
arbitrary web images share essentially nothing — one is a cat, the next is a spreadsheet.
Every ACNE04 image is a half-face photograph at roughly 70° under similar lighting, so the
*null* distribution of similarity is shifted far to the right. An absolute threshold
inherited from one corpus measures something different on the other.

### 7.1 Calibrating to a false-positive rate instead

The fix is to set the threshold from the corpus's own null — the distribution of
similarity between a real image and an unrelated real image — at a stated false-positive
rate:

| target per-image false-positive rate | threshold on ACNE04 |
|---|---|
| 10% | 0.736 |
| 5% | 0.768 |
| **1%** | **0.824** |

We use **1%, giving 0.824**, and report the null distribution alongside every
memorisation rate, because a rate quoted without the null it was measured against is not
interpretable.

### 7.2 What this implies beyond this study

Any memorisation or copy-detection audit that inherits an absolute similarity threshold
from work calibrated on web-scale imagery, and applies it to a medical or face corpus
where all images are the same kind of picture, is reporting a number dominated by its
false-positive rate. On ACNE04 the error is not marginal — 82.6% against a true duplicate
rate of 5.2%.

This cuts both ways for us and we should say so. It means a naive audit would have made
our closed-set generator look catastrophically memorising when it may not be. It also
means the acne prior's anonymity claim (§4.1) — which rests entirely on generated faces
not being copies — cannot be checked with an off-the-shelf threshold either, and was not
checked with any.


## 8. Preliminary: what the closed-set pool looks like at 150 of 4,740 images

Run 2026-08-19 on the first 150 generated images, against the 948-image real training
split the generator was fine-tuned on. **Preliminary by construction** — the pool is 3%
complete and these numbers will be recomputed on the finished pool — but the two gating
questions have clear enough answers to record now.

### The generator did not memorise its training set

| | |
|---|---|
| memorisation threshold, calibrated on the training split at 1% per-image FPR | **0.698** |
| generated images above it | **0.00%** |
| **maximum SSCD similarity to any training image** | **0.5836** |
| what the published threshold (0.5) would report | 12.00% |
| what the published threshold flags among *real* training images | 79.3% |

The closest thing the generator produced to a training image scores **0.584**, against a
calibrated threshold of 0.698 and a maximum of **0.976 between two genuinely distinct real
photographs**. Nothing in the pool is a near-copy in any sense the corpus supports.

The 12.00% figure is what an audit inheriting Somepalli et al.'s 0.5 would have published.
It is a false-positive rate, and §7 explains why: the same threshold calls 79.3% of the
real training images copies of each other.

**This is the measurement the acne prior asserted and never made.** Zein et al.'s stated
motivation is anonymity — that the generated faces are not real patients — and the paper
contains no nearest-neighbour audit of any kind. Ours took twenty minutes on the first 3%
of the pool.

### The generator did not collapse identity

| | pool (150 images) | real training split |
|---|---|---|
| faces detected | 117 / 150 (78%) | 916 / 948 (96.6%) |
| distinct identities | 115 | 267 |
| **identities per image** | **0.767** | **0.282** |
| largest identity's share | 2.6% | 2.4% |
| coverage of the 267 real training identities | **0.0%** | — |

The pool carries **more identity diversity per image than the real data does**, which is
not a surprise once the audit is in hand: the real corpus repeats subjects 2–3 times each
and the generator does not. Identity collapse — the failure §8.10 exists to catch, and the
one that is invisible to FID and to every per-image quality metric — has not happened here.

**Zero coverage of the real training identities** is the second half of the anonymity
question. Not one of the 267 people in the generator's training data has a synthetic
near-neighbour above the identity threshold. The generator learned the domain — clinical
framing, pathology, demographics — without learning the individuals.

### It also produces artefacts at a measurable rate

Face detection succeeds on 78% of generated images against 96.6% of real ones. The missing
22% are the artefacts §4.9 declines to filter: multi-panel composites, extreme crops,
distorted anatomy. That rate is reported as a property of the generator rather than removed,
because a generator producing unusable images at rate *r* is worth less than one that does
not, and the substitution study should say so rather than curating the difference away.


## 9. It is not just ACNE04, and the second case needs no model at all

The strongest objection to §5 is that the subject structure was recovered with a face
embedding, so the finding could be an artefact of that instrument. **SCIN** removes the
objection.

SCIN (Google Research and Stanford Medicine, 10,000+ crowdsourced dermatology images)
ships `image_1_path`, `image_2_path` and `image_3_path` per case. **`case_id` is ground
truth for "same subject."** No embedding, no threshold, no judgement call — and the whole
audit is metadata-only, a few megabytes of CSV with no images downloaded.

| | SCIN | ACNE04 |
|---|---|---|
| images | 10,407 | 1,457 |
| distinct subjects / cases | 5,033 | ~550–750 |
| **images per subject** | **2.07** | **~2.4** |
| subjects with more than one image | 61.3% | 54% |
| **images belonging to a multi-image subject** | **81.3%** | **84%** |
| **test images sharing a subject with training, image-level split** | **75.6%** | **77.5%** |
| how the subject grouping was obtained | published metadata | ArcFace embedding |

The two datasets come from different institutions, different countries, different
collection methods — one a clinical cohort photographed to protocol, the other crowdsourced
from US internet users — and different content. They have **the same structure**, and the
standard splitting practice leaks the same amount in both.

SCIN's version is arguably worse. Its multi-image cases are the *same lesion* at different
distances and angles (2,289 cases carry all three of `AT_DISTANCE`, `AT_AN_ANGLE` and
`CLOSE_UP`), which is a tighter relationship than two photographs of one person's two
cheeks. An image-level split there puts the same lesion, photographed twice, on both sides.

**SCIN also has file-level duplication that case grouping would not fix**: 15 image files
are listed under more than one `case_id`, and one appears under **eleven**. Grouping by
case is necessary and not sufficient; file-level deduplication is needed too.

### 9.1 What this changes about the claim

§5 said ACNE04's published splits leak. This says something larger and more useful:

**Dermatology image datasets routinely contain several images per subject or lesion, and
image-level splitting — the default everywhere — leaks roughly three-quarters of the test
set in both datasets we checked.** One required a face-recognition model to see; the other
is stated in the published schema and nobody appears to have measured the consequence.

**We claim no novelty for the existence of the leak.** A literature check on 2026-08-19
found that patient- and lesion-level partitioning is established practice in a growing part
of the field, that HAM10000's release paper reports its own per-lesion image counts, and
that Abhishek et al. documented the leakage across three datasets and published corrected
partitions. Anyone reading this should assume the general phenomenon is known.

What the cross-dataset comparison adds is narrower: the rate varies by a factor of two
across datasets and tracks redundancy, and ACNE04 — which has no published audit at all —
sits at the top of that range while being the only one whose subject structure is invisible
in metadata.

The check costs nothing. For a dataset that publishes case or subject identifiers it is a
`groupby`; for one that does not, it is an embedding pass. `scripts/audit_scin.py` runs the
first in under a second.


### 9.2 A third dataset, and the leak lands on melanoma

HAM10000 publishes `lesion_id`, so it is a third metadata-only check. Its duplication is
**already known** — Abhishek et al. documented near-duplicate lesions spanning the
train/validation boundary and released corrected partitions for the DermaMNIST derivative.
We are not claiming to have found it. What is measured here is the size and the *shape* of
the consequence.

| | images | per subject/lesion | image-level split leakage |
|---|---|---|---|
| ACNE04 | 1,457 | ~2.4 | **77.5%** |
| SCIN | 10,407 | 2.07 | **75.6%** |
| HAM10000 | 10,015 | 1.34 | **38.5%** |

HAM10000 leaks about half as much, which follows directly from having half as many repeats
per lesion. The mechanism is the same; the magnitude tracks the redundancy.

**But the leak is not spread evenly across diagnoses, and where it concentrates is the
problem:**

| diagnosis | images | images per lesion | test images leaked |
|---|---|---|---|
| **melanoma** | 1,113 | **1.81** | **68.3%** |
| dermatofibroma | 115 | 1.58 | 56.1% |
| basal cell carcinoma | 514 | 1.57 | 55.8% |
| benign keratosis | 1,099 | 1.51 | 51.3% |
| vascular | 142 | 1.45 | 46.6% |
| actinic keratosis | 327 | 1.43 | 46.3% |
| **nevus** | 6,705 | **1.24** | **29.4%** |
| overall | 10,015 | 1.34 | 38.5% |

**Melanoma leaks at 68.3%, more than double the benign majority class at 29.4%** — because
melanomas were photographed more often per lesion, which is entirely reasonable clinical
practice and has nothing to do with anyone's methodology.

Melanoma sensitivity is *the* headline metric on this benchmark. On an image-level split,
roughly two-thirds of the melanoma test images have another photograph of the same lesion
in training, against under a third of the nevi. Whatever a model learns about a specific
lesion helps disproportionately on exactly the class the number is reported for.

This is the same shape found on ACNE04 in §6: the leakage bonus was near zero on the two
majority grades and about ten points on the two rare ones. Two datasets, two different
reasons for the correlation — there, rare grades were photographed *less*, so the *clean*
partition was enriched in them; here, the malignant class is photographed *more*, so the
*leaked* partition is. **The general lesson is that duplication is almost never independent
of class, so a leakage rate quoted as a single number understates its effect on the classes
anyone cares about.**


### 9.3 When the leak is class-dependent, and when it is not

HAM10000's leak varies by a factor of 2.3 across diagnoses (68.3% melanoma, 29.4% nevus).
The obvious next question is whether that is a general property of leakage or a property of
HAM10000, and SCIN answers it — because SCIN's duplication has a different cause.

In HAM10000, how many times a lesion was photographed is **a clinician's decision**:
worrying lesions get more images. In SCIN, it is **a fixed capture protocol** — contributors
were asked for a close-up, one at a distance and one at an angle — so the count reflects
compliance rather than concern.

| SCIN, by condition | images | per case | leaked |
|---|---|---|---|
| Tinea | 212 | 2.28 | 82.3% |
| Eczema | 1,079 | 2.21 | 80.2% |
| Allergic contact dermatitis | 590 | 2.19 | 79.7% |
| Folliculitis | 306 | 2.15 | 78.5% |
| Urticaria | 442 | 2.07 | 75.4% |

The spread is **6.9 points**, against HAM10000's **38.9**. The mechanism predicts the
difference and the data show it, which turns an observation into something falsifiable:

> **Leakage is class-dependent when duplication reflects a clinical judgement, and roughly
> uniform when it reflects a fixed capture protocol.**

ACNE04 fits the same rule from the other side. There the *rare* grades are photographed
*less* — 1.21 images per person at very severe against 2.30 at mild — because severe cases
are rarer, not because anyone chose to photograph them less. The leak is again
class-dependent, and again the direction follows the collection process rather than
anything about the images.

**Practical consequence.** A dataset's leakage rate cannot be assumed uniform, and whether
it is depends on something knowable in advance: whether the number of images per subject
was a clinical decision or a protocol. When it was a clinical decision, expect the leak to
concentrate on the class the benchmark is reported for, because the same clinical concern
that drove the extra photograph drives the reporting.

### 9.4 A null worth reporting: leakage does not vary by skin tone

Leakage compounding an existing fairness problem is a reasonable worry, and SCIN can test
it directly because it publishes Monk skin tone labels.

| Monk tone | cases | images per case | leaked |
|---|---|---|---|
| 1 | 577 | 2.00 | 73.2% |
| 2 | 1,660 | 2.07 | 75.4% |
| 3 | 1,265 | 2.06 | 75.3% |
| 4 | 687 | 2.10 | 76.9% |
| 5 | 361 | 2.05 | 74.9% |
| 6 | 248 | 2.20 | 79.8% |
| 7 | 137 | 2.09 | 75.5% |
| 8 | 57 | 2.14 | 77.2% |

**It does not.** The range is 73.2–79.8% with no monotone trend, and the two extremes sit
on the smallest strata. Contributors across skin tones complied with the capture protocol
equally, so the leak is distributed equally.

This is a null and we report it as one. It is worth stating because the opposite result
would have been a significant fairness finding, and because a reader who has just been told
that leakage concentrates on clinically important classes will reasonably wonder whether it
also concentrates on under-represented groups. In this dataset it does not.


### 9.5 A fourth dataset falsifies the direction, and sharpens the claim

§9.3 predicted that where duplication reflects clinical judgement, the leak concentrates on
the *clinically worrying* class. ISIC 2020 — 33,126 images, a clinical screening collection
carrying both `patient_id` and `lesion_id` — was chosen to test that prediction. **It fails
it.**

| ISIC 2020 | images per group | image-level split leakage |
|---|---|---|
| grouped by **lesion** | 1.01 | **2.0%** |
| grouped by **patient** | 16.11 | **99.9%** |

| at patient level | images | per patient | leaked |
|---|---|---|---|
| **benign** | 32,542 | **15.84** | **99.8%** |
| **malignant** | 584 | **1.36** | **39.3%** |

The class dependence is enormous — a **60-point** spread, larger than HAM10000's 38.9 — but
the direction is **reversed**. Here the *benign* class leaks almost totally, because
screening photographs many nevi per patient while a malignant finding is usually a single
lesion. HAM10000's clinician photographed the worrying lesion repeatedly; ISIC 2020's
photographed everything else.

**So the prediction was wrong and the claim has to be weakened where it was specific and
strengthened where it was vague:**

> Leakage is strongly class-dependent whenever duplication is driven by any clinical
> process, and **the direction is not predictable from first principles**. It depends on
> whether the process re-photographs the lesion of concern or surveys the ones that are not.
> It must be measured per dataset; it cannot be assumed.

That is less satisfying than a rule and more useful than one, because the failure mode it
guards against is assuming a direction and reporting the wrong class as protected.

### 9.6 The grouping level decides whether you see the problem at all

The sharper lesson from ISIC 2020 is methodological. **The same dataset looks clean or
catastrophic depending on which identifier you group by:**

- by lesion: 1.01 images per lesion, **2.0% leakage** — nothing to report
- by patient: 16.11 images per patient, **99.9% leakage** — almost the entire test set

An audit that grouped ISIC 2020 by `lesion_id`, which is exactly the column HAM10000
taught the field to use, would conclude the dataset is fine. It is not; it is one of the
most duplicated datasets we have looked at, at the level that matters.

The reverse holds too. HAM10000 ships no `patient_id`, so its leakage can only be seen at
the lesion level, and ACNE04 ships neither — its structure is invisible in metadata
entirely and takes a face model to recover.

| dataset | grouping available | where the leak is visible |
|---|---|---|
| HAM10000 | `lesion_id` | lesion (38.5%) |
| ISIC 2020 | `lesion_id`, `patient_id` | **patient only** (99.9% vs 2.0%) |
| SCIN | `case_id` | case (75.6%) |
| ACNE04 | none | requires face identity (77.5%) |

**Practical rule:** group at the coarsest level the data supports — patient above lesion
above image — and where no identifier exists, assume the structure is there until measured.
Three of these four datasets would be misread by an audit that used the first identifier it
found.


## 10. The severity label is confounded with where the image came from

Looking at random real images by grade turned up something the numbers had not: the severe
ones carry watermarks, Chinese acne-treatment branding, "Before 治疗前" banners and
pixelated eye regions. The mild ones look like ordinary clinic photographs. That is the
signature of two different sources — a clinic camera and scraped before/after marketing
material — and if severity tracks the source, a classifier can predict it from capture
artefacts without looking at skin.

Image resolution fingerprints the capture device, so this is measurable with no modelling
at all. One resolution, **3112×3456, covers 76% of the corpus** — one device. The rest are
scattered across 154 other resolutions.

| grade | n | on the dominant device | elsewhere |
|---|---|---|---|
| mild | 513 | **87.1%** | 12.9% |
| moderate | 633 | 83.3% | 16.7% |
| severe | 182 | 61.5% | 38.5% |
| **very severe** | **129** | **16.3%** | **83.7%** |

**Five in six very-severe images were not taken on the device that produced three quarters
of the dataset.**

### 10.1 The shortcut, stated as a classifier

A rule that reads only the file header — *"not the dominant resolution, therefore severe"* —
achieves **sensitivity 0.57 at specificity 0.85** for the severe-versus-mild distinction the
benchmark exists to make. It never opens the image.

Seen from the other direction:

| source group | mild | moderate | severe | very severe |
|---|---|---|---|---|
| dominant device (n=1,107) | 40.4% | 47.6% | 10.1% | **1.9%** |
| everything else (n=350) | 18.9% | 30.3% | 20.0% | **30.9%** |

The prior on very-severe is **16× higher** off the dominant device. Any feature that leaks
capture provenance — JPEG quantisation tables, resampling artefacts, colour profile,
watermark pixels, the mosaic blocks over someone's eyes — carries most of that signal, and
a convolutional network is good at exactly this.

### 10.2 Why this ties the rest of the audit together

It explains a result from §6 that had no mechanism attached. The leakage bonus was near zero
on the two majority grades and about ten points on the two rare ones. Both of those grades
are also the ones sourced from marketing material, where a handful of clinics' promotional
sets supply many images — so those grades are simultaneously the most subject-leaky, the most
provenance-confounded, and the smallest. Three separate problems land on the same two
classes, which are the two that matter clinically.

It also reframes §4. Two expert teams disagreeing about severity is easier to understand
when a third of the severe images come from an advertiser with an interest in the "before"
looking bad.

### 10.3 The ablation, which came back against the hypothesis

The section above originally ended by saying an ablation was needed and had not been run.
It has now been run, and **it does not support the concern.**

Every image was re-encoded to 512×512 square at JPEG quality 90 — identical resolution,
aspect ratio and quantisation for every file — and training repeated with the same
subject-disjoint splits, hyperparameters and seed:

| | balanced accuracy | plain accuracy | QWK |
|---|---|---|---|
| original images | 0.7296 | 0.7064 | 0.783 |
| **provenance-normalised** | **0.7391** | 0.7339 | 0.818 |

Accuracy did not drop. It rose by 0.95 points, which is comfortably inside the
seed-to-seed standard deviation of 1.68 points measured in §6, so the honest reading is
**no change**.

That looked like a clean negative result, and it was reported as one. **It is not, and the
next experiment is why.**

### 10.4 The ablation did not actually remove the confound

Watermarks, branding and eye-pixelation survive re-encoding, so the obvious follow-up is
whether capture source is *still* detectable after normalisation. It is, almost perfectly.

A classifier trained to predict source — dominant device or not — **from the normalised
images alone**, on the same subject-disjoint splits, reaches **99.5% accuracy and 99.4%
balanced accuracy** against a 61.0% majority-class baseline. It converges in one epoch.

So the ablation of §10.3 removed resolution, aspect ratio and quantisation and left the
channel wide open. **Its null result therefore says nothing about whether models exploit
provenance** — the "normalised" images were still carrying the source almost losslessly.
The conclusion drawn from it was invalid, and this section retracts it.

The honest state is the one §10.1 and §10.2 describe, with one addition:

- Source correlates strongly with severity — 16× the prior on very-severe off the dominant
  device.
- **Source is 99.4% recoverable from the images**, and survives the obvious normalisation.
- Whether a severity classifier uses that channel is **still unknown**, because no ablation
  yet run actually closes it.

Closing it would mean removing watermarks and depixelating eyes, or restricting the corpus
to one source and accepting that this leaves 21 very-severe images. Neither is cheap, and
neither is done here.

A single seed also cannot distinguish a small effect from none; a difference of a point
either way is noise here.

**What survives.** The confound in the data is real and measurable from file headers in
under a minute: 87.1% of mild images and 16.3% of very-severe ones share one resolution, and
the prior on very-severe is 16× higher off the dominant device. Nothing in the release warns
of it. What is *not* established, and what an earlier draft of this section overstated, is
that any model exploits it.

**Practical rule:** before trusting a medical image benchmark, check whether the label
correlates with capture metadata. Resolution, aspect ratio, JPEG quantisation and EXIF are
all free to read, and a dataset assembled from multiple sources with source-dependent labels
is a shortcut waiting to be found.


## 11. The labels are derived, and the derivation exposes the noise

The label file rows are `<filename> <grade> <lesion_count>`. For all 1,457 records the
grade is exactly the Hayashi banding of the count — 1–5 → 0, 6–20 → 1, 21–50 → 2,
>50 → 3, with **zero** exceptions. The severity label carries no information beyond the
count; all annotation noise originates in counting.

That makes the duplicates a free repeat-annotation experiment. The same image, counted
twice, without the annotators knowing it was the same image:

- **Exact count agreement: 12 of 38 (31.6%).**
- Median |Δcount| = 1, mean 3.0, maximum 16.
- Median *relative* difference 22%.
- **Grade agreement: 30 of 38 (78.9%)**, 95% Wilson CI **63.7–88.9%**.

All 8 grade disagreements are cases where the two counts fall in different Hayashi
bands. Four straddle the 5/6 boundary; four straddle 20/21. The 20/21 cases are not
boundary jitter — they are (9 vs 21), (8 vs 24), (7 vs 22), (9 vs 24), differences of a
factor of 2.4–3. Their filenames form consecutive runs
(`levle1_16/20/22/23` ↔ `levle2_150/151/152/153`), which is the signature of a block of
images ingested twice and counted by different procedures rather than of individual
annotator slips.

**This is the finding with teeth.** ACNE04's own labels reproduce at 78.9% when the
identical photograph is annotated a second time. Published accuracies on the benchmark
are 83.7% (Wu et al.'s baseline), 84.11% (Label Distribution Smoothing), 86.06%
(KIEGLFN) and 87.33%. The upper end of the self-consistency confidence interval is
88.9%, so these results are not *impossible* — but they are being scored against a
label function that is itself unreliable at roughly the rate they are claiming to
improve upon, and the 4.25% leakage pushes in the same direction. Reported
improvements of 0.4 points (83.70 → 84.11) are far inside this noise.

Caveat stated up front: n = 38 is small and the interval is wide. The 78.9% is an
*estimate of annotation reproducibility*, not a hard ceiling, and duplicates may not be
a random sample of the corpus. The correct reading is not "these papers are wrong" but
"this benchmark cannot currently resolve differences of a few points, and nobody has
been reporting that."

## 12. The filename prefix is not a label

**42 of 1,457 files** have a `levleN` prefix that disagrees with their labelled grade
(e.g. `levle1_151.jpg` is labelled grade 0 with 4 lesions). Any pipeline that derives
labels from filenames — a natural shortcut given the naming scheme — gets 2.9% of the
dataset wrong. Our loader reads the label files and ignores the prefix.

---

## 13. What this changes for our study

1. **Dedup config is now `phash_max_hamming: 2`, `embed_min_cosine: 0.98`, CLIP
   embedder enabled.** The shipped defaults were tuned for datasets where perceptual
   hashing has signal; on a single-pose face corpus they do not.
2. **We do not use the published folds.** Protocol §3.2 already required splitting
   after deduplication on our own seed; this audit turns that from hygiene into a
   necessity, since the published folds leak.
3. **The comparison to the 83.7–87.3% band is now heavily qualified.** Our real-only
   arm is trained on de-duplicated, leak-free splits and is scored with balanced
   accuracy. It should be expected to land *below* the published band for three
   compounding reasons — no leakage bonus, a harder metric, and a smaller training set
   after deduplication — and that is not evidence of a broken pipeline.
4. **The annotation-noise estimate becomes a reference line in our results.** If the
   synthetic-substitution curve's decline is smaller than the label noise, we cannot
   claim to have measured it. 78.9% grade reproducibility is the number to beat before
   any effect of a few points is interpretable.
5. **Same-subject leakage remains unmeasured** and is the next thing to run. It is the
   one channel that could still be inflating both our numbers and the literature's.

## 14. What went wrong while doing this, and what that suggests

An audit is a claim about someone else's data made by someone whose own instruments and
inferences are equally fallible. Over the two days that produced this document, several
findings were wrong when first written, and the pattern in *which* ones were wrong is more
useful than any individual correction.

### 15.1 The measurements survived; the explanations did not

| claim | fate |
|---|---|
| 38 byte-identical duplicate groups | held |
| 4.25% fold leakage, every published fold | held |
| ~600 subjects, 77.5% subject leakage | held |
| 30.1% cross-team grade agreement | held |
| 99.4% source detectability | held |
| *"the leak concentrates on the clinically worrying class"* | **falsified** by ISIC 2020 |
| *"models exploit the provenance shortcut"* | **retracted**, ablation was invalid |
| *"count-prompting will fix severity fidelity"* | **falsified**, made it worse |

Everything in the first group is a count or a correlation computed from files. Everything in
the second is a story about *why* — a mechanism, a prediction, a proposed fix. **Three of
three causal claims failed; none of the descriptive ones did.**

That is not a coincidence and it is not modesty. Counting duplicate groups has one degree of
freedom; explaining why duplication correlates with severity has many, and each is an
opportunity to be confidently wrong. A reader deciding how much of this document to trust
should weight it accordingly, and `docs/06_due_diligence.md` is ordered on exactly that principle.

### 15.2 An ablation that does not ablate produces a confident null

The provenance sequence is the sharpest case. The reasoning was: severity correlates with
capture source, so normalise resolution, aspect ratio and quantisation, retrain, and see
whether accuracy drops. It did not drop, and that was reported as evidence that models do
not exploit provenance.

The follow-up showed the normalised images still carry source at **99.4% accuracy**, because
watermarks and eye-pixelation survive re-encoding. The ablation had removed a channel nobody
was necessarily using and left the obvious one open. **A null result from an intervention
that was never verified to intervene is not evidence of anything.**

The check that catches this is cheap and was only run because the first result looked too
clean: *after the ablation, can the thing you claim to have removed still be recovered?*

### 15.3 Instruments need validating against ground truth before use

Two instruments were built here and both failed their first honest test.

The **lesion counter** was meant to measure whether generated severity matched requested
severity. Against ACNE04's published counts it scored Spearman 0.103 untuned and 0.365 after
sweeping 24 settings, against roughly 0.5 for usability — it was measuring skin texture. It
was only ever tested because ACNE04 happens to publish counts.

The **memorisation threshold** was adopted from the literature at SSCD 0.5, the value behind
the field's reference replication rate. On this corpus that flags 82.6% of *real* images as
copies of one another, because the threshold was calibrated on LAION, where two arbitrary
images share nothing. Calibrated on the corpus it becomes 0.824 — and the same audit run at
the published threshold would have reported 12.00% memorisation for a generator whose true
figure is 0.00%.

**A threshold is a property of a corpus, not of an instrument.**

### 15.4 Optimising a proxy is not optimising the thing

Prompting the generator with explicit lesion counts and tighter framing raised exact
agreement from 34.8% to 47.9%, which looked like a clear win, and dropped ordinal fidelity
from Spearman 0.176 to 0.010. The predictions had collapsed onto the modal class, where they
match more often without being more faithful.

Exact agreement was a reasonable-looking metric that moved the wrong way for a
mechanical reason. It is worth reporting two metrics that can disagree, and worth being
suspicious when the easy one improves.

### 15.5 Four latent bugs, all of which produced plausible output

None of these threw an error at the point of failure:

- A bare `data/` line in `.gitignore` silently excluded `src/fitymi/data/`, so the pushed
  repository was missing its core package while every local test passed.
- Run identifiers were content-addressed over everything *except* the data, so two arms
  differing only in how the corpus was split hashed identically and the resume path would
  have skipped the second.
- Image generation ran on CPU while a 40-core GPU sat idle, because the sampler resolved
  `"cuda" if available else "cpu"` — it did not look broken, it looked *slow*, which is the
  attribution that stops the investigation.
- Inverse-frequency class weights clamped absent classes to a count of one, handing them
  the largest weight in the vector; a two-class probe returned exactly 0.0000 accuracy,
  which reads as a broken model rather than a broken loss.

The common feature is that each produced output a reasonable person would accept. The only
reason any surfaced is that a number failed to reconcile with a number obtained another way.

### 15.6 The practical version

- Prefer claims computable from files over claims about mechanisms, and label which is which.
- After any ablation, verify the thing was actually removed.
- Validate every instrument against ground truth on *this* corpus before pointing it at
  anything, and treat published thresholds as corpus-specific.
- Report at least two metrics that can disagree, and distrust improvements in the easy one.
- Reconcile every important number against a second, independent route to it. That is what
  caught all four bugs.


## 15. Publishability

Sections 1, 3, 4 and 5 are a self-contained contribution independent of the synthetic
data question: a benchmark used by a substantial acne-grading literature contains 5.2%
exact duplicates, leaks ~4.25% of every published test split, has 21% grade
disagreement on its own repeated images, and ships 42 misleading filenames. None of it
requires a model. All of it is checkable in minutes by anyone with the archive.

Two things to do before claiming priority: search for an existing ACNE04 audit (we
found none in the 2026-08-19 literature pass, but absence of evidence from one pass is
weak), and run the same-subject analysis in §6.5 so the audit is complete rather than
partial.

## Reproducing

```
make prepare CONFIG=configs/acne04_closed.yaml   # writes data/splits/dedup_report.json
python scripts/audit_acne04.py                   # the tables above
python scripts/make_acne04_audit_figure.py       # docs/examples/acne04_audit.png
```
