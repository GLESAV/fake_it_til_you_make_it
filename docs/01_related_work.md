# Related Work: Can Generated Images Train a Traditional Classifier?

**Status:** literature scan completed 2026-08-19. See [§0 Provenance](#0-provenance-and-verification-status) for how these claims were sourced and which ones still need full-text verification before they go into an arXiv submission.

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

At the time of writing, **everything is [S] or [?]**. `docs/VERIFY.md` tracks the
re-verification checklist. Do not paste a number from this file into the paper
without promoting it to [V] first.

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

**Sariyildiz et al., "Fake it till you make it: Learning transferable
representations from synthetic ImageNet clones," CVPR 2023**
(arXiv:2212.08420) — the paper this repository is named after. Generates
ImageNet "clones" with Stable Diffusion using only class names, trains
classifiers from scratch, and finds that with minimal class-agnostic prompt
engineering the synthetic-trained models close **a large part** — explicitly not
all — of the gap to real-data training, while transferring surprisingly well to
downstream tasks. **[S]**

The framing matters for us: *transfer* performance held up better than
*in-domain* performance. A representation can be good without the classifier
being calibrated for the target distribution.

**He et al., "Is synthetic data from generative models ready for image
recognition?" ICLR 2023** (arXiv:2210.07574). Studies synthetic data in
zero-shot, few-shot and pre-training regimes. Introduces *language enhancement*
(a T5 word-to-sentence model that diversifies prompts) because naive prompting
produces insufficiently diverse images. Conclusion is deliberately two-sided:
"powerfulness and shortcomings." **[S]**

**Azizi et al., "Synthetic Data from Diffusion Models Improves ImageNet
Classification," 2023** (arXiv:2304.08466). Fine-tunes Imagen into a
class-conditional generator (FID 1.76 at 256×256, IS 239); augmenting real
ImageNet with its samples improves accuracy over strong ResNet/ViT baselines.
Classification Accuracy Score 64.96 at 256×256, 69.24 at 1024×1024. **[S]**

Note that this is a **Q1** result obtained with a generator far stronger than
anything we will fine-tune, and it *still* frames synthetic data as augmentation.

### 2.2 The corrective wave (2024–2026)

**Fan et al., "Scaling Laws of Synthetic Images for Model Training … for Now,"
CVPR 2024** (arXiv:2312.04567). The most important prior for our design. Studies
how performance scales with synthetic dataset size for both CLIP-style and
supervised classifier training. Findings: prompt strategy, classifier-free
guidance scale, and choice of generator all materially change the scaling
exponent; after tuning, synthetic scales *nearly* as well as real for CLIP but
**significantly underperforms for supervised classifiers**. The stated cause is
that off-the-shelf text-to-image models simply cannot render certain concepts.
**[S]**

The "…for Now" in the title is the whole argument: the gap is attributed to
generator capability, not to something fundamental about synthetic data.

**"When Pretty Isn't Useful: Investigating Why Modern Text-to-Image Models Fail
as Reliable Training Data Generators"** (arXiv:2602.19946). Argues that visual
fidelity and *training utility* have decoupled — newer, prettier models are not
better data generators, because they concentrate mass on a narrow region of the
real manifold. The fidelity–diversity trade-off is named as the mechanism. **[S]**

This is directly relevant to a decision we have to make: we should **not** assume
the newest/best-looking generator is the best generator for this study, and we
should sweep guidance scale rather than fixing it at the aesthetic default.

**Wang et al., "Exploring the Equivalence of Closed-Set Generative and Real Data
Augmentation in Image Classification," 2025** (arXiv:2508.09550). The closest
existing work to our **Q3**. Distinguishes *closed-set* augmentation (generator
trained only on the given training set) from *open-set* (generator saw outside
data), and derives an empirical "equivalent scale" — how many synthetic images
are worth one real image — noting the exchange rate varies with baseline
training-set size. Evaluated on natural **and medical** images. **[S]**

Their closed-set/open-set distinction is the correct axis and we adopt it
directly (see §6).

**Representation-conditioned generation** (arXiv:2605.27495) reports that
classifiers trained on synthetic data can *outperform* real-data baselines once
the synthetic set is scaled to ~3× the real set size, when generation is
conditioned on representations rather than text. **[S]** If this replicates it is
a strong argument that the bottleneck is conditioning, not synthesis.

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

**Sagers et al., "Improving dermatology classifiers across populations using
images generated by large diffusion models," NeurIPS 2022 workshop**
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
Images," 2023** (arXiv:2301.04802). Notable because the title explicitly reaches
the *fully synthetic* regime — the closest medical analogue to our Q2. **[S]**
**Full text is a required read before we finalise the design; our sandbox could
not retrieve it.**

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
Learning," ICCV 2019.** Introduces **ACNE04**: 1,457 facial images (1,513 files
in the distributed archive), 18,983 lesion bounding boxes, severity graded on the
**Hayashi** criterion into 4 ordinal levels (mild / moderate / severe / very
severe), shot at roughly 70° from frontal per the Hayashi protocol's half-face
requirement. Distributed via `github.com/xpwu95/LDL`, "free for academic usage."
**[S]** This is the de-facto benchmark.

**Reported ACNE04 performance** in follow-up work clusters around 85–86%
accuracy for severity grading — e.g. one system reports precision 85.31%,
sensitivity 84.83%, specificity 94.66%, accuracy 86.06%. **[S]** The MICCAI 2024
**AcneAI** work reports ICC 0.8 for severity via a UNet/EfficientNet lesion
segmentation route. **[S]** **Label Distribution Smoothing** (arXiv:2403.00268,
ISBI 2024) is the current strong ordinal baseline. **[S]**

This ~86% figure is the number our real-only ceiling should be compared against.
If our real-only baseline lands far below it, our pipeline is broken, not the
synthetic data.

### 4.1 The one existing acne-synthetic result — and why it needs replication

**Zaghbani / Boukhris et al., "Generative Adversarial Networks for anonymous
Acneic face dataset generation," PLOS ONE 2024** (arXiv:2211.04214,
PMC11020863). Trains StyleGAN2-ADA per severity level to produce 1024×1024
synthetic acneic faces (mild/moderate/severe plus a synthetic healthy class), with
**anonymity** as the stated motivation. Reports a CNN "trained using the generated
synthetic acneic face images and tested using authentic face images" reaching
**97.6% accuracy with InceptionResNetV2**. **[S]**

**We should treat this number as the central target of our study, and we should
expect it not to hold.** Reasons for scepticism, all of which our design controls
for:

- 97.6% on real test images exceeds every *real-trained* ACNE04 severity result
  we found (~86%). A synthetic-trained model beating the real-trained
  state-of-the-art by 11 points is an extraordinary claim.
- Adding a **synthetic healthy class** changes the task. Discriminating
  "GAN-generated clear face" from "real acne face" can be solved by
  generator-artefact detection rather than by dermatological features. If the
  healthy class is synthetic and the acne classes are also synthetic but the
  *test* set is real, the class structure and the real/synthetic axis may be
  entangled.
- The training faces are StyleGAN2 samples fine-tuned from a face prior; the test
  faces are real. No statement of whether generator training data and classifier
  test data were disjoint was retrievable from the abstract.

A separate 2024 paper on GANs for acne dataset generation in *BioMedInformatics*
(doi:10.3390/biomedinformatics4020059) is marked **RETRACTED**. **[S]** We should
find out why before building on anything adjacent to it.

**ACNEDIT** (Springer 2025) does non-destructive acne editing with dynamic
intensity tuning — an *editing* rather than *de novo* generation approach, which
is a third arm worth considering: paste synthetic lesions onto real skin.

### 4.2 Gap statement

To the best of this scan, **no published work measures acne classifier
performance as a continuous function of the real:synthetic mixing ratio against a
fixed, untouched real test set, with matched training budgets and variance
estimates.** The nearest neighbours are Wang et al. 2025 (exchange rate, not
acne) and Zaghbani et al. 2024 (acne, fully synthetic, single point, no budget
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
8. **Run a memorisation audit.** Carlini et al. (arXiv:2301.13188) and Somepalli et
   al. (arXiv:2305.20086) find 0.5–2% of generations are partial training-data
   duplicates, driven by data duplication and text conditioning. **[S]** For a
   fine-tuned-on-1,457-faces generator this rate could be far higher, and the
   "anonymity" claim in the acne prior depends entirely on it. Nearest-neighbour
   retrieval against the generator's training set is mandatory, not optional.

---

## 7. Reading list to promote [S] → [V]

Priority order for the full-text pass (needs an unrestricted network):

1. Zaghbani et al. 2024 (PLOS ONE) — the 97.6% claim. **Highest priority.**
2. Akrout et al. 2023 (arXiv:2301.04802) — fully-synthetic medical regime.
3. Wang et al. 2025 (arXiv:2508.09550) — exchange-rate methodology.
4. Fan et al. 2024 (arXiv:2312.04567) — scaling protocol, guidance sweep.
5. Wu et al. ICCV 2019 — ACNE04 splits and the exact Hayashi mapping.
6. Sariyildiz et al. CVPR 2023 — prompt strategy details.
7. Ktena et al. Nature Medicine 2024 — fairness evaluation protocol.
8. The retracted BioMedInformatics acne-GAN paper — retraction notice.

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
