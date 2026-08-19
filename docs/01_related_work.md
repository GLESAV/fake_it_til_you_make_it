# Related Work: Can Generated Images Train a Traditional Classifier?

**Status:** literature scan completed 2026-08-19; **full-text verification pass 1 completed 2026-08-19**. See [§0 Provenance](#0-provenance-and-verification-status) for how these claims were sourced and which ones still need full-text verification before they go into an arXiv submission.

---

## 0. Provenance and verification status

This review was assembled inside a sandbox whose egress policy blocks `arxiv.org`,
`openreview.net`, `nature.com`, `link.springer.com`, `sciencedirect.com`,
`ncbi.nlm.nih.gov` and every other publisher host we probed. Only search-engine
results were reachable, so **every numeric claim below is second-hand** — taken from
search summaries and abstracts, not from a full-text read.

Each claim is tagged:

- **[V]** — verified against a primary source (full text read).
- **[S]** — sourced from an abstract or search summary only; plausible but **must be
  re-checked against the PDF before citation**.
- **[?]** — we believe this is true but could not locate a citable source; treat as a
  hypothesis, not a fact.

**Verification pass 1 was run on 2026-08-19** from an unrestricted network. The
design-critical claims — the acne prior, the closest medical analogue, the exchange-rate
methodology, the guidance protocol, the ACNE04 statistics, the memorisation prior and
the retraction — are now **[V]**, and several changed materially on contact with the
primary sources (a wrong author attribution, a cherry-picked headline number, a results
table that contradicts its own abstract, and an accuracy band that is not the metric we
report). `docs/VERIFY.md` lists what is verified, what is still [S], and the two hosts
that refused our fetcher. Do not paste a number from this file into the paper without
checking its tag.

---

## 1. Why this question is not already answered

The headline result of the last three years is easy to state and easy to
over-read: *text-to-image models can produce training data that helps.* The
literature that actually matters here is the literature about **where that stops
being true**. Three separate questions get conflated constantly:

| Question | What it measures | Typical finding |
|---|---|---|
| **Q1. Augmentation** — does adding synthetic to real help? | real + synthetic vs. real | Usually yes, modest gains |
| **Q2. Substitution** — can synthetic *replace* real? | 100% synthetic vs. 100% real, equal budget | Usually no, gap persists |
| **Q3. Exchange rate** — how many synthetic images buy one real image? | performance vs. mixing fraction | Rarely measured |

Almost all dermatology work answers **Q1**. Our study is designed around **Q2 and
Q3**, which is where the interesting and publishable uncertainty lives. A paper
that only shows "adding synthetic acne images helps" would be a fifth restatement
of an established result.

---

## 2. General computer vision

### 2.1 The optimistic wave (2022–2023)

**Sariyildiz, Alahari, Larlus & Kalantidis, "Fake it till you make it: Learning
transferable representations from synthetic ImageNet clones," CVPR 2023**
(arXiv:2212.08420) — the paper this repository is named after. **[V]** — full text read
2026-08-19. Generates ImageNet "clones" with Stable Diffusion using only class names
plus WordNet hypernyms/definitions, trains classifiers from scratch, evaluates on real
images.

**The verified numbers are much harsher than the abstract's framing.** On ImageNet-1K at
matched scale (Table 1; testing on real images throughout):

| Training set | guidance | prompt | IN-Val top-1 |
|---|---|---|---|
| ImageNet-1K (real, PyTorch ResNet-50) | — | — | **76.1%** |
| ImageNet-1K (real, RSB-A1) | — | — | **80.1%** |
| ImageNet-1K-SD (synthetic) | 7.5 | `c, dc` | 26.2% |
| ImageNet-1K-SD (synthetic) | 7.5 | `c, hc inside b` | 30.1% |
| ImageNet-1K-SD (synthetic) | **2.0** | `c, dc` | **42.9%** |

Two things to take from this. First, **the in-domain substitution gap at matched budget
is 33 points** (42.9 vs 76.1), and the paper says so plainly: the synthetic-trained model
"lags behind in all cases when compared to the two models that are trained on the
ImageNet-1K training set." The "closes a large part of the gap" claim is about *transfer*
and about ImageNet-100, not about in-domain ImageNet-1K substitution. Citing this paper
as evidence that synthetic can replace real is citing the abstract, not Table 1 — the
same failure mode as Akrout et al. in §3.

Second, and this is the design-relevant one: **dropping guidance from the 7.5 default to
2.0 is worth 16.7 top-1 points** (26.2 → 42.9), holding everything else fixed. That is
larger than any prompt-engineering effect they report. Combined with Fan et al.'s
independently derived optimum of 2.0 for Stable Diffusion (§2.2), **two papers using
different methods agree the training-utility optimum is ≈2.0 against an aesthetic default
of 7.5.** Our guidance sweep is centred accordingly; a study generating at the default
would be measuring a crippled generator and attributing the result to synthetic data.

Scale helps but does not close it: at 10×, 20× and 50× the real set size (guidance 2)
they reach 72.4 / 72.4 / 73.3 top-1 on ImageNet-**100**.

**He et al., "Is synthetic data from generative models ready for image
recognition?" ICLR 2023** (arXiv:2210.07574). Studies synthetic data in
zero-shot, few-shot and pre-training regimes. Introduces *language enhancement*
(a T5 word-to-sentence model that diversifies prompts) because naive prompting
produces insufficiently diverse images. Conclusion is deliberately two-sided:
"powerfulness and shortcomings." **[S]**

**Azizi, Kornblith, Saharia, Norouzi & Fleet, "Synthetic Data from Diffusion Models
Improves ImageNet Classification," 2023** (arXiv:2304.08466). **[V]** for the headline
metrics and authors; the augmentation delta is still [S]. Fine-tunes Imagen into a
class-conditional generator (FID 1.76 at 256×256, IS 239); augmenting real
ImageNet with its samples improves accuracy over strong ResNet/ViT baselines.
Classification Accuracy Score 64.96 at 256×256, 69.24 at 1024×1024. **[S]**

Note that this is a **Q1** result obtained with a generator far stronger than
anything we will fine-tune, and it *still* frames synthetic data as augmentation.

### 2.2 The corrective wave (2024–2026)

**Fan et al., "Scaling Laws of Synthetic Images for Model Training … for Now,"
2024** (arXiv:2312.04567; Lijie Fan, Kaifeng Chen, Dilip Krishnan, Dina Katabi,
Phillip Isola, Yonglong Tian). **[V]** — full text read 2026-08-19. *Note: the arXiv
page carries no CVPR 2024 acceptance note; verify the venue before citing it as CVPR.*
The most important prior for our design. Findings: prompt strategy, classifier-free
guidance scale, and choice of generator all materially change the scaling exponent;
after tuning, synthetic scales *nearly* as well as real for CLIP but **significantly
underperforms for supervised classifiers**. The stated cause is that off-the-shelf
text-to-image models cannot render certain concepts.

**The directly actionable number: the guidance scale that maximises training utility
is far below the aesthetic default.** They sweep CFG ∈ [1.5, 10.0] for Stable
Diffusion, [1.0, 2.0] for Imagen and [0.1, 1.0] for Muse, and the optima for
supervised classifier training are **SD 2.0**, Imagen 1.5, Muse 0.3. **[V]** Stable
Diffusion's usual default is 7.5. Higher CFG improves text–image alignment and
aesthetics while cutting diversity, and diversity is what training wants. **Our
guidance sweep should be centred on 2.0 and should not extend far above 5.0**; a study
that generated at 7.5 because it is the default would be measuring a handicapped
generator and calling the result a property of synthetic data.

The "…for Now" in the title is the whole argument: the gap is attributed to generator
capability, not to something fundamental about synthetic data.

**Adamkiewicz, Moser, Frolov, Nauen, Raue & Dengel, "When Pretty Isn't Useful:
Investigating Why Modern Text-to-Image Models Fail as Reliable Training Data
Generators," CVPR 2026** (arXiv:2602.19946). **[V]** — the paper is real and the
author list, previously a placeholder, is now resolved. Trains classifiers purely on
synthetic sets from state-of-the-art T2I models released 2022–2025 and finds that
**accuracy on real test data consistently *declines* with newer generators**, which
converge on "a narrow, aesthetic-centric distribution that undermines diversity and
real data distribution coverage." Argues that visual
fidelity and *training utility* have decoupled — newer, prettier models are not
better data generators, because they concentrate mass on a narrow region of the
real manifold. The fidelity–diversity trade-off is named as the mechanism. **[S]**

This is directly relevant to a decision we have to make: we should **not** assume
the newest/best-looking generator is the best generator for this study, and we
should sweep guidance scale rather than fixing it at the aesthetic default.

**Wang et al., "Exploring the Equivalence of Closed-Set Generative and Real Data
Augmentation in Image Classification," 2025** (arXiv:2508.09550; Haowen Wang, Guowei
Zhang, Xiang Zhang, Zeyuan Chen, Haiyang Xu, Dou Hoon Kwark, Zhuowen Tu). **[V]** —
full text read 2026-08-19. The closest existing work to our **Q3**.

Definitions, verified: **closed-set** = the generative model is trained from scratch on
the classification dataset itself, with no external data; **open-set** = the generator
was trained or pre-trained on a large external corpus (e.g. LAION-5B), as with Stable
Diffusion. We adopt this axis directly (see §6). Their closed-set generators are EDM
(CIFAR-10, BloodMNIST) and DiT-XL/2 (ImageNet-100); open-set is SD 1.4/2.0/3.0.

They fit an empirical equivalence of the form
`n_syn/n_base ≃ c₁^(n_base/k) × (c₂^(n_real⁺/n_base) − 1)`, and the **c₂ term is the
exchange rate**: how many synthetic images it takes to match the benefit of one real
image. Verified values: **[V]**

| Dataset | Setting | Exchange rate (c₂) |
|---|---|---|
| CIFAR-10 | closed | 2.53× |
| CIFAR-10 | open | 2.93× |
| **BloodMNIST (medical)** | **closed** | **3.88×** |
| ImageNet-100 | closed | 37.84× |
| ImageNet-100 | open | 1.68× |

Their headline is unambiguous: **"real images are always more advantageous than using
synthetic data."** Closed-set scales better as the base training set grows.

**Two things to be careful about when we position against this work.** First, the
spread from 1.68× to 37.84× means there is no transferable constant — the exchange rate
is a property of the dataset and generator, which is the argument for measuring it on
acne rather than importing a number. Second, and more importantly for our gap
statement: **this is augmentation-equivalence, not substitution.** Their quantity is
"how much synthetic must I *add* to match *adding* n real images," with the real base
held fixed. Ours is "what happens when synthetic *displaces* real at a fixed total
budget." Akrout et al.'s table above shows these two questions have different answers —
the marginal synthetic image is worth ~1 real image when added and <0.5 when
substituted. Wang et al. do not measure the substitution end, and our §4.2 gap
statement stands.

**Karthikeyan, Unger & Eilertsen, "Representation-Conditioned Diffusion Models for
Guided Training Data Generation"** (arXiv:2605.27495). **[V]** Conditioning latent
diffusion on DINOv2/DINOv3/CLIP representations rather than on class labels improves
ImageNet-100 top-1 by **+10.76 pp** over class-conditioned generation, and scaling the
synthetic set lets it **exceed the real-data baseline by 2.0 pp**. *Correction: our
earlier draft attributed the win to "~3× the real set size"; the verified claim is a
+2.0 pp margin under scaling, and the headline number is the +10.76 pp conditioning
effect.* If this replicates it is a strong argument that the bottleneck is
conditioning, not synthesis — and it is the one result in this section pointing the
other way.

### 2.3 The failure mode nobody should ignore

**Shumailov et al., "AI models collapse when trained on recursively generated
data," Nature 2024** (doi:10.1038/s41586-024-07566-y). Recursive training on
model output causes *early collapse* (distribution tails thin out) and *late
collapse* (low-frequency modes vanish permanently). Mitigation requires periodic
injection of fresh real data. **[S]** A critical note (arXiv:2410.12954) disputes
parts of the setup — cite both. **[S]**

For us this is not directly a threat (we generate one generation, not a
recursive chain) but it names the exact quantity our study should measure:
**tail preservation**. In acne, the tail *is* the clinical signal — severe and
nodulocystic presentations are rare and are the cases that matter.

---

## 3. Medical imaging and dermatology

**Sagers, Diao, Groh, Rajpurkar, Adamson & Manrai, "Improving dermatology classifiers
across populations using images generated by large diffusion models," NeurIPS 2022
Workshop on Synthetic Data for Empowering ML Research** **[V]** for authors and venue;
the size of the improvement is still [S].
(arXiv:2211.13352). DALL·E 2 generates photorealistic skin-disease images across
skin types; augmenting Fitzpatrick17k training data improves classification
overall and **especially for underrepresented skin tones**. **[S]**

**Ktena et al., "Generative models improve fairness of medical classifiers under
distribution shifts," Nature Medicine 30:1166–1173, 2024** (arXiv:2304.09218).
The strongest-venue result in this space. Learns augmentations from data with
generative models in a label-efficient way, exploiting abundant *unlabeled* data,
and improves fairness under distribution shift across imaging modalities
including dermatology. **[S]**

**Akrout et al., "Diffusion-based Data Augmentation for Skin Disease
Classification: Impact Across Original Medical Datasets to Fully Synthetic
Images," 2023** (arXiv:2301.04802). **[V]** — full text and results table read
2026-08-19. The closest medical analogue to our Q2, and **its own numbers contradict
its abstract**, which is why reading it mattered.

Setup: Stable Diffusion embeddings learned by textual inversion for six skin
conditions on a proprietary macroscopic dataset; 30,000 images generated per class,
filtered by a non-skin classifier and then by a pretrained ensemble; four balanced
training sets built from the survivors; evaluation on **3,582 real images**.

| Training set | Real | Synthetic | Top-1 | Top-2 | Top-3 |
|---|---|---|---|---|---|
| real-small | 250 | 0 | 53.41% | 73.51% | 83.22% |
| real | 500 | 0 | 54.05% | 73.95% | 84.84% |
| hybrid | 250 | 250 | 54.13% | 73.23% | 85.01% |
| **synthetic** | **0** | **500** | **47.29%** | 70.71% | 84.09% |

Read the top-1 column, which is the only one not saturated:

- **At matched budget (500 vs 500) the fully synthetic arm loses 6.76 points.** This is
  a clean Q2 substitution result and it is negative.
- **500 synthetic images are worth less than 250 real ones** — the synthetic arm is
  6.1 points below real-small, which has half the budget. The exchange rate at the
  fully-synthetic end is worse than 2:1 *against*.
- **But the hybrid arm at 250 real + 250 synthetic matches the 500-real arm** (54.13
  vs 54.05). So the marginal synthetic image is worth roughly one real image when it
  is *added alongside* real data, and worth well under half a real image when it
  *replaces* real data. The exchange rate is not a constant — it collapses as the real
  fraction goes to zero. This is exactly the curvature our sweep is designed to
  measure, and it is the strongest existing evidence that a single-point comparison
  cannot characterise substitution.

The abstract's claim — that the approach "maintains a similar classification accuracy
even when trained on a fully synthetic dataset" — is true only at top-4/top-5, where
all four arms are within 0.2 points of each other because the metric has saturated on
a six-class problem. **Do not cite the abstract's framing; cite Table 4.**

One further observation from their §3.3 that we should treat as a finding rather than
an aside: early stopping triggered **only** on the fully-synthetic arm, and its
validation accuracy peaked at 89% while its real-test top-1 was 47.29%. They read this
as synthetic data "allowing for faster training and convergence." The alternative
reading is the one our §8.6 control formalises: the synthetic set is narrower and more
prototypical, so it is easier to fit and its held-out synthetic validation score is
inflated, while generalisation to real images is worse. A 42-point validation-to-test
gap on the synthetic arm versus a much smaller one elsewhere is the prototype effect
with a number attached.

**Derm-T2IM** (arXiv:2401.05159), **LesionGen** (arXiv:2507.23001),
**SkinDualGen** (arXiv:2507.19970), **DermDiff** (arXiv:2503.17536) — a steady
stream of dermatology-specific text-to-image systems, mostly evaluated as
augmentation on HAM10000/ISIC. Reported hybrid-vs-real gains in the 4–15%
range depending on metric and paper. **[S]** These numbers are not comparable
across papers; the variance is itself evidence that the evaluation protocol in
this subfield is under-standardised.

**Recurring methodological weaknesses across this literature** — these are what
our protocol is designed to avoid:

1. **Test-set contamination.** When the generator is fine-tuned on data that
   overlaps the evaluation split, "synthetic training" is laundered real
   training. Few papers state their generator's training split explicitly.
2. **Budget mismatch.** Real-only baselines are frequently trained on *N* images
   while hybrid arms get *N + M*. That comparison cannot separate "synthetic
   helps" from "more data helps."
3. **Single seed, single split.** Differences of 1–3 points are reported without
   variance estimates, on test sets of a few hundred images where the binomial
   standard error alone is ±2–3 points.
4. **FID as a proxy for utility.** FID measures distributional distance to the
   generator's *training* set and correlates poorly with downstream accuracy;
   the "When Pretty Isn't Useful" result above is precisely this point.

---

## 4. Acne specifically

**Wu et al., "Joint Acne Image Grading and Counting via Label Distribution
Learning," ICCV 2019.** Introduces **ACNE04**: 1,457 facial images, 18,983 lesion
bounding boxes, severity graded on the **Hayashi** criterion into 4 ordinal levels,
shot at roughly 70° from frontal per the Hayashi protocol's half-face requirement.
Distributed via `github.com/xpwu95/LDL`; the README states verbatim: *"This work is
free for academic usage. For other purposes, please contact Xiaoping Wu."* **[V]**

The **Hayashi thresholds are lesion counts**, verified: mild **1–5**, moderate
**6–20**, severe **21–50**, very severe **>50**. **[V]**

**The class distribution is severely imbalanced, and this is the single most
consequential fact about ACNE04 for our study.** Under the standard 80/20 split
(1,165 train / 292 test): **[V]**

| Grade | Lesions | Train | Test | Total | Share |
|---|---|---|---|---|---|
| Mild | 1–5 | 410 | 103 | 513 | 35.2% |
| Moderate | 6–20 | 506 | 127 | 633 | 43.4% |
| Severe | 21–50 | 146 | 36 | 182 | 12.5% |
| Very severe | >50 | 103 | 26 | 129 | 8.9% |

Three consequences we have to design around:

1. **The two clinically load-bearing classes have 182 and 129 images in total.** After
   sealing a test split and carving a validation split, the severe and very-severe
   *training* pools are on the order of 110 and 80 images. A closed-set generator
   fine-tuned per class is being asked to learn a distribution from ~100 examples,
   which is precisely the regime where memorisation, not synthesis, is the likely
   outcome. Our memorisation audit is therefore not a formality — it is the load-
   bearing control for the tail classes.
2. **Tail metrics will be noisy no matter what we do.** With 36 and 26 test images in
   the two tail classes, the binomial standard error on per-class recall is roughly
   ±8 and ±10 points. Any per-class tail claim needs the interval attached, and
   differences smaller than ~15 points on those classes are not interpretable from a
   single test set.
3. **Plain accuracy on this test set is dominated by the two majority classes**
   (78.6% of the data). A model that never predicts "very severe" loses at most 8.9
   points of accuracy. This is why we report **balanced accuracy** as the primary
   metric — and why the comparison in the next paragraph is not apples-to-apples.

**Reported ACNE04 performance.** The published band is **83.7–87.3% plain accuracy**:
Wu et al.'s own baseline is **83.70 ± 1.53%**, Label Distribution Smoothing (ISBI 2024,
arXiv:2403.00268) reports **84.11 ± 1.94%** under pre-defined 5-fold cross-validation,
KIEGLFN reports **86.06%** (precision 85.31%, sensitivity 84.83%, specificity 94.66%),
and an LDL-family framework reports **87.33%**. **[V]** The MICCAI 2024 **AcneAI** work
reports ICC 0.8 via a lesion-segmentation route. **[S]**

**Caveat that our earlier draft got wrong:** these are *plain accuracy on the
imbalanced test set*, not balanced accuracy. Our real-only arm reports balanced
accuracy and will legitimately land well below 84% while being a perfectly healthy
pipeline. The sanity check on our real-only baseline must therefore be run on **plain
accuracy** against this band, with balanced accuracy reported alongside it. Comparing
our balanced accuracy to their plain accuracy and concluding "our pipeline is broken"
would be a self-inflicted wound.

### 4.1 The one existing acne-synthetic result — and why it needs replication

**Zein, Chantaf, Fournier & Nait-Ali, "Generative adversarial networks for anonymous
acneic face dataset generation," PLOS ONE 19(4): e0297958, 2024** (arXiv:2211.04214,
PMC11020863). **[V]** — full text read 2026-08-19.

> **Attribution correction.** Earlier drafts of this review cited this work as
> "Zaghbani / Boukhris et al." That attribution was wrong and was never checked; it
> came from a search summary. The authors are Hazem Zein, Samer Chantaf, Régis
> Fournier and Amine Nait-Ali. `refs.bib` has been corrected.

What the paper actually does: StyleGAN2-ADA is trained per severity level to produce
1024×1024 synthetic acneic faces (mild / moderate / severe), plus a fourth
StyleGAN2-generated **healthy** class. Three CNNs are then trained on synthetic images
only and evaluated on real faces. Anonymity is the stated motivation.

Verified specifics, all of which weaken the headline:

- **The generator's real training set is 1,473 images: 1,073 from ACNE04 plus 400
  scraped from Google Images.** It is not "ACNE04" as our earlier draft implied, and
  the 400 web images have no stated provenance or licence.
- **The 97.6% is the best of three numbers, and the other two are far lower.**
  Figure 9 reports, on the same real test images: InceptionResNetV2 **97.6%**,
  ResNet152V2 **82.7%**, ResNet50V2 **76.03%**. A 21.6-point spread across three
  closely related ImageNet backbones trained on identical data is not a property of
  the data; it is a symptom of a small and/or unstable evaluation. Quoting only the
  97.6% — as we did — is the cherry-pick the paper invites.
- **The size, class balance and source of the real test set are stated nowhere.** The
  Figure 9 caption reads only: *"Confusion matrix for each CNN model tested on unseen
  images."* No n, no per-class counts, no source dataset.
- **There is no explicit statement that the real test images were excluded from the
  1,473 images used to train the StyleGAN2 generators.** The paper says the models
  were evaluated on "unseen real facial acne images"; "unseen" is only ever asserted
  with respect to *classifier* training, which is trivially true because the
  classifier saw nothing but synthetic images. The generator-side disjointness — the
  one that matters — is never claimed.
- **The real test subset "contains only authentic acneic face images."** The
  classifier has four classes and the healthy class is synthetic-only. So a four-class
  model appears to be scored on a test set containing no examples of one of its
  classes. Either the healthy class is absent from the real evaluation, or its real
  images come from an unnamed source. Either way the reported accuracy is not
  comparable to a four-class ACNE04 result.
- Training-set composition for the classifier: 1,387 healthy, 841 mild, 1,173
  moderate, 1,086 severe synthetic images (4,487 total).
- On a *synthetic* held-out split (Table 2) all three backbones score 97.8–98.4%.
  The collapse to 76–83% for two of them on real images is the real/synthetic domain
  gap showing up exactly where our study predicts it.

**Bottom line for our design.** The 97.6% is not evidence that synthetic acne images
can replace real ones. It is a single unreplicated number, from the best of three
models, on an unspecified test set, with no stated generator/test disjointness, for a
task whose class structure differs from ACNE04's. Our study should still target it —
but as a *claim to be characterised*, not a result to be beaten. §8.6's prototype-effect
control applies directly: a StyleGAN2 trained per class on ~100–400 images will produce
narrow, prototypical samples, and a classifier trained on those can score high on the
easy majority of a real test set while learning nothing about the tail.

### 4.1.1 The subject structure supplies a concrete mechanism for the 97.6%

Added after the ACNE04 identity audit (`docs/05_acne04_audit.md` §5), which found the
dataset's 1,457 photographs to be roughly 550–750 individuals.

Zein et al. trained StyleGAN2-ADA per severity level on **1,073 ACNE04 images** plus 400
web images, then evaluated a synthetic-trained classifier on "unseen real acneic face
images." The paper never states where those real test images came from, how many there
were, or whether they were excluded from the generators' training data. The most
economical reading is that they are the ACNE04 remainder.

If so, the identity structure alone predicts heavy contamination. Simulating that split
200 times over the measured identity graph — 1,073 images to the generator, the rest
held out, stratified by severity because the generators were trained per level:

| identity threshold | chance rate | held-out images sharing a subject with the generator's training set |
|---|---|---|
| 0.85 | 0.019% | **15.9%** (95% range 12.8–18.7) |
| 0.80 | 0.058% | 36.7% (32.7–41.0) |
| 0.75 | 0.120% | **56.9%** (52.6–61.2) |
| 0.60 | 0.273% | 77.4% (72.8–82.0) |

Stratifying per severity, as they did, changes nothing: 15.2% and 56.3% at 0.85 and 0.75.

**Why this matters for interpreting their number.** Their generators saw on the order of
100–400 images per severity level. The 1.88% replication rate Somepalli et al. measure
is for Stable Diffusion trained on two billion images; a GAN fitted to a few hundred
faces is a different regime entirely, and memorisation there is the expected outcome
rather than a tail risk. Compose the two facts and a specific mechanism appears:

1. StyleGAN2 memorises individuals from a few-hundred-image training set.
2. Those individuals reappear in the "unseen" real test set, because ACNE04 photographs
   each person several times and any split of it puts most subjects on both sides.
3. A classifier trained on the synthetic images has therefore effectively been trained
   on those individuals, and recognising them at test time is not severity grading.

This is a hypothesis, not a demonstration: the paper does not describe its test set, so
we cannot confirm the split. But it is a hypothesis with a computed prior, it explains an
otherwise extraordinary result — a synthetic-trained model beating every real-trained
ACNE04 result by roughly eleven points — and it requires no assumption of error on the
authors' part beyond the one the field shares, namely splitting a face dataset at the
image level.

It also sharpens what our own study has to do. Replicating 97.6% "under controls" is not
one control but three: a generator trained only on our real *train* split (§6.2), splits
that are subject-disjoint rather than image-disjoint (§8.8 of the protocol), and a
memorisation audit against the generator's own training set (§8.2). Any one of them
absent and the number is uninterpretable.


**The retracted adjacent paper.** Sankar, Chaturvedi, Nayan, Hesamian, Braytee &
Prasad, "Utilizing Generative Adversarial Networks for Acne Dataset Generation in
Dermatology," *BioMedInformatics* 2024, **4**, 1059–1070, was retracted on
**12 August 2024** (notice: doi:10.3390/biomedinformatics4030104). **[V]** The stated
reason is **significant overlap of methodology, data and a figure (Figure 6) with an
earlier preprint by a different authorship group, without acknowledgment or
citation**; the retraction was approved by the Editor-in-Chief and **the authors did
not agree to it**. Given the subject matter and the dates, the "earlier preprint by a
different authorship group" is plausibly arXiv:2211.04214 — the Zein et al. preprint
above — but the notice does not name it and we should not assert this. Practical
consequence: cite the retraction, do not cite the retracted paper, and treat the small
acne-GAN literature as thin enough that one plagiarism case removes a meaningful
fraction of it.

**ACNEDIT** (Springer 2025) does non-destructive acne editing with dynamic intensity
tuning — an *editing* rather than *de novo* generation approach, which is a third arm
worth considering: paste synthetic lesions onto real skin. **[S]**

### 4.2 Gap statement

To the best of this scan, **no published work measures acne classifier
performance as a continuous function of the real:synthetic mixing ratio against a
fixed, untouched real test set, with matched training budgets and variance
estimates.** The nearest neighbours are Wang et al. 2025 (exchange rate, not
acne) and Zein et al. 2024 (acne, fully synthetic, single point, no budget
control, implausible headline). That gap is our paper.

---

## 5. Datasets available for the real arm

| Dataset | n | Labels | Access | Fit |
|---|---|---|---|---|
| **ACNE04** | 1,457 | Hayashi 4-level + 18,983 lesion boxes | GitHub/Baidu/Drive, academic use | **Primary choice.** Ordinal severity, standard benchmark, comparable numbers exist |
| **AcneSCU** | — | acne, higher-res / lesion-level | public per secondary sources **[?]** | Possible second real test set for external validity |
| **Fitzpatrick17k** | 16,577 | 114 conditions + Fitzpatrick scale | public | Acne subset is small; but the *only* option with skin-tone labels |
| **SCIN** | ~10k imgs / 5k volunteers | up to 3 weighted condition labels | public (Google) | Crowdsourced, real-world capture; acne subset useful as OOD test |
| **DDI** | 656 | biopsy-proven, Fitzpatrick I–VI | gated | Gold-standard for skin-tone robustness, little acne |

Known caveat: an audit of Fitzpatrick17k data quality (arXiv:2401.14497) reports
duplicates and label problems — deduplicate before use. **[S]**

---

## 6. What the literature implies for our design

Positions we adopt, each traceable to something above:

1. **Report a curve, not a point.** Mixing fraction ∈ {0, 25, 50, 75, 100}% is the
   independent variable (Q3). Single-point comparisons are what made the acne
   prior unfalsifiable.
2. **Hold the real test set completely out of everything** — classifier training,
   generator training, prompt selection, hyperparameter selection. This is the
   single most common defect in §3/§4.
3. **Match the training budget across arms.** Every arm sees the same *number* of
   training images. Otherwise Q1 contaminates Q2.
4. **Run closed-set and open-set generator arms separately** (Wang et al. 2025).
   Closed-set = generator fine-tuned only on the real train split. Open-set =
   off-the-shelf prompted generator that never saw our real data. These answer
   genuinely different questions and mixing them is why the field's numbers do
   not reconcile.
5. **Sweep guidance scale; do not use the aesthetic default.** Fan et al. 2024 and
   "When Pretty Isn't Useful" both show this is a first-order variable for
   training utility and that it trades against diversity.
6. **Do not report FID as evidence of utility.** Report it as a descriptor
   alongside precision/recall/coverage/density, and report memorisation checks.
7. **Measure the tail.** Per-class metrics on severe/very-severe, not just overall
   accuracy — this is where model collapse and mode-dropping show up, and it is
   the clinically load-bearing part of the label space.
8. **Run a memorisation audit, with SSCD at threshold 0.5.** Somepalli et al.,
   "Diffusion Art or Digital Forgery?" (arXiv:2212.03860, CVPR 2023), measure that
   **1.88% of random Stable Diffusion v1.4 generations have SSCD similarity > 0.5 to a
   training image** — and they state this is a **lower bound**, because their retrieval
   covered only the 12M-image aesthetic split, "which accounts for only 0.6% of total
   training data." **[V]** Carlini et al. (arXiv:2301.13188) extract "over a thousand"
   verbatim training examples from diffusion models; we could not verify a *rate* from
   that paper and should not quote one. **[S]**

   Two design consequences. First, adopt their instrument rather than inventing one:
   **SSCD embeddings, top-1 similarity, 0.5 threshold**, reported as a lower bound.
   Second, 1.88% is the rate for a model trained on ~2 billion images. Our closed-set
   generator is fine-tuned on roughly 1,000 faces, with the tail classes at ~100
   images each — three to four orders of magnitude less data per mode. There is no
   basis for expecting 1.88% to transfer, and every reason to expect substantially
   more copying, concentrated in exactly the severe and very-severe classes. The
   "anonymity" claim in the acne prior depends entirely on this rate and that paper
   reports no such audit. If our memorisation rate on the tail classes is high, the
   synthetic-only arm is a laundered copy of the real training set and the substitution
   question is not being measured at all — this is a study-invalidating condition, not
   a caveat.

---

## 7. Verification status

**Pass 1 complete, 2026-08-19**, from an unrestricted network. `docs/VERIFY.md` holds
the full table. Summary:

**Promoted to [V]:** Zein et al. 2024 (the acne prior — with four material corrections,
§4.1); Akrout et al. 2023 (results table read; contradicts its own abstract, §3);
Wang et al. 2025 (exchange rates 1.68×–37.84×, §2.2); Fan et al. 2024 (CFG optima,
§2.2); ACNE04 statistics, Hayashi thresholds and class distribution (§4); the published
ACNE04 accuracy band and the metric caveat (§4); Somepalli et al. (1.88% at SSCD>0.5,
a lower bound, §6.8); the BioMedInformatics retraction and its reason (§4.1); Ktena et
al. (Nat Med 30:1166–1173); "When Pretty Isn't Useful" and the representation-
conditioned paper (both real, author lists resolved).

**Still [S], in priority order for pass 2:**

1. Sariyildiz et al. CVPR 2023 — prompt ablation and the actual accuracy numbers. Our
   claims about it are currently qualitative only.
2. Azizi et al. 2023 — FID 1.76, IS 239, CAS 64.96/69.24.
3. Shumailov et al. 2024 — Nature volume and pages.
4. Carlini et al. 2023 — an extraction *rate*, if one exists in the paper.
5. He et al. ICLR 2023; Sagers et al. 2022.
6. The "4–15%" hybrid-vs-real range across Derm-T2IM / LesionGen / SkinDualGen /
   DermDiff. This is a range assembled from incomparable protocols and should probably
   be deleted rather than verified.

**Two hosts refused our fetcher** (403): `openaccess.thecvf.com` and `www.mdpi.com`.
Wu et al. ICCV 2019 and the retraction notice were therefore verified through
secondary full texts and search indexing of the notice text respectively. Re-check both
from a browser before submission.

**What changed on contact with the primary sources** — worth stating plainly, because
it is the argument for having done this before writing any of the paper:

- The acne prior's authors were wrong in our draft (not "Zaghbani/Boukhris").
- Its 97.6% is the best of three numbers on the same test set; the others are 82.7%
  and 76.03%.
- Its test set is never described — no size, no class balance, no source — and no
  generator/test disjointness is claimed.
- The closest medical analogue to our Q2 reports, in its own results table, a 6.8-point
  *loss* at matched budget, while its abstract says accuracy is "maintained."
- The ACNE04 accuracy band we planned to sanity-check against is plain accuracy on an
  imbalanced test set, not the balanced accuracy we report.
- The guidance scale that maximises training utility is ~2.0, not the 7.5 default.

---

## Sources

- [Fake it till you make it (CVPR 2023) — arXiv:2212.08420](https://arxiv.org/abs/2212.08420) · [project page](https://europe.naverlabs.com/research/computer-vision/imagenet-sd/)
- [Is synthetic data from generative models ready for image recognition? (ICLR 2023)](https://openreview.net/forum?id=nUmCcZ5RKF)
- [Synthetic Data from Diffusion Models Improves ImageNet Classification — arXiv:2304.08466](https://arxiv.org/abs/2304.08466)
- [Scaling Laws of Synthetic Images for Model Training … for Now (CVPR 2024) — arXiv:2312.04567](https://arxiv.org/abs/2312.04567v1)
- [When Pretty Isn't Useful — arXiv:2602.19946](https://arxiv.org/pdf/2602.19946)
- [Exploring the Equivalence of Closed-Set Generative and Real Data Augmentation — arXiv:2508.09550](https://arxiv.org/abs/2508.09550)
- [Representation-Conditioned Diffusion Models for Guided Training Data Generation — arXiv:2605.27495](https://arxiv.org/pdf/2605.27495)
- [AI models collapse when trained on recursively generated data (Nature 2024)](https://www.nature.com/articles/s41586-024-07566-y) · [critical note — arXiv:2410.12954](https://arxiv.org/abs/2410.12954)
- [Improving dermatology classifiers across populations using images generated by large diffusion models — arXiv:2211.13352](https://arxiv.org/abs/2211.13352) · [OpenReview](https://openreview.net/forum?id=Vzdbjtz6Tys)
- [Generative models improve fairness of medical classifiers under distribution shifts (Nature Medicine 2024)](https://www.nature.com/articles/s41591-024-02838-6) · [arXiv:2304.09218](https://arxiv.org/abs/2304.09218)
- [Diffusion-based Data Augmentation for Skin Disease Classification — arXiv:2301.04802](https://arxiv.org/pdf/2301.04802) · [Springer chapter](https://link.springer.com/chapter/10.1007/978-3-031-53767-7_10)
- [Derm-T2IM — arXiv:2401.05159](https://arxiv.org/abs/2401.05159)
- [LesionGen — arXiv:2507.23001](https://arxiv.org/pdf/2507.23001)
- [SkinDualGen — arXiv:2507.19970](https://arxiv.org/html/2507.19970)
- [DermDiff — arXiv:2503.17536](https://arxiv.org/pdf/2503.17536)
- [Joint Acne Image Grading and Counting via Label Distribution Learning (ICCV 2019)](https://openaccess.thecvf.com/content_ICCV_2019/papers/Wu_Joint_Acne_Image_Grading_and_Counting_via_Label_Distribution_Learning_ICCV_2019_paper.pdf) · [ACNE04 / LDL code](https://github.com/xpwu95/LDL)
- [Improving Acne Image Grading with Label Distribution Smoothing (ISBI 2024) — arXiv:2403.00268](https://arxiv.org/pdf/2403.00268) · [code](https://github.com/openface-io/acne-lds)
- [AcneAI (MICCAI 2024)](https://papers.miccai.org/miccai-2024/042-Paper2216.html)
- [KIEGLFN: A unified acne grading framework](https://www.sciencedirect.com/science/article/abs/pii/S0169260722002930)
- [AI in the Assessment and Grading of Acne Vulgaris: A Systematic Review (MDPI 2025)](https://www.mdpi.com/2075-4426/15/6/238)
- [Generative Adversarial Networks for anonymous Acneic face dataset generation (PLOS ONE 2024)](https://journals.plos.org/plosone/article?id=10.1371%2Fjournal.pone.0297958) · [arXiv:2211.04214](https://arxiv.org/abs/2211.04214) · [PMC11020863](https://pmc.ncbi.nlm.nih.gov/articles/PMC11020863/)
- [RETRACTED: Utilizing GANs for Acne Dataset Generation in Dermatology](https://doi.org/10.3390/biomedinformatics4020059)
- [ACNEDIT (Springer 2025)](https://link.springer.com/chapter/10.1007/978-3-032-05472-2_10)
- [Extracting Training Data from Diffusion Models — arXiv:2301.13188](https://arxiv.org/pdf/2301.13188)
- [Understanding and Mitigating Copying in Diffusion Models — arXiv:2305.20086](https://arxiv.org/pdf/2305.20086)
- [Investigating the Quality of DermaMNIST and Fitzpatrick17k — arXiv:2401.14497](https://arxiv.org/pdf/2401.14497)
- [SCIN: Crowdsourcing Dermatology Images with Google Search Ads — arXiv:2402.18545](https://arxiv.org/pdf/2402.18545)
- [awesome-skin-image-analysis-datasets](https://github.com/sfu-mial/awesome-skin-image-analysis-datasets)
