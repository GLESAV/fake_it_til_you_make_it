# Citation verification checklist

The literature review and `paper/refs.bib` were originally assembled in a sandbox with
no access to publisher hosts (`docs/03_environment.md`). **No claim may enter the
manuscript until it is verified here.**

Tags used in `docs/01_related_work.md`:

- **[V]** verified against the primary source (full text read)
- **[S]** from an abstract or search summary only
- **[?]** believed true, no citable source located

## Status

Verification pass 1 run 2026-08-19 on an unrestricted network (local M4 Max session).

| # | Source | Why it matters | Status | Verified by / date |
|---|---|---|---|---|
| 1 | **Zein** et al. 2024, PLOS ONE 19(4):e0297958 (arXiv:2211.04214), `zein2024` | The 97.6% synthetic-trained / real-tested acne claim. | **[V]** — with major corrections, see §4.1 | local session, 2026-08-19 |
| 2 | Akrout et al. 2023 (arXiv:2301.04802), `akrout2023` | Closest medical analogue to our fully-synthetic arm. | **[V]** — full results table read | local session, 2026-08-19 |
| 3 | Wang et al. 2025 (arXiv:2508.09550), `wang2025` | Exchange-rate methodology, closed-/open-set distinction. | **[V]** | local session, 2026-08-19 |
| 4 | Fan et al. 2024 (arXiv:2312.04567), `fan2024` | Guidance-scale and scaling protocol. | **[V]** — CFG values confirmed in pass 1; the CVPR 2024 acceptance it could not confirm is now confirmed, doi `10.1109/CVPR52733.2024.00705`, pp. 7382–7392 | local session, 2026-08-19; venue 2026-08-28 |
| 5 | Wu et al. 2019, ICCV, `wu2019` | ACNE04 splits, Hayashi mapping, image count. | **[V]** — via secondary full texts; thecvf.com returns 403 to our fetcher, see note below | local session, 2026-08-19 |
| 6 | Sariyildiz et al. 2023, CVPR, `sariyildiz2023` | Prompt strategy; how much of the gap closed. | **[V]** — Table 1 read; in-domain gap is 33 points and guidance 7.5→2.0 is worth 16.7 | local session, 2026-08-19 |
| 7 | Ktena et al. 2024, Nature Medicine, `ktena2024` | Fairness protocol; volume/pages. | **[V]** — Nat Med **30**, 1166–1173 (2024) confirmed | local session, 2026-08-19 |
| 8 | Retracted acne-GAN paper (doi:10.3390/biomedinformatics4020059), `sankar2024retracted` | Whether anything adjacent is safe to cite. | **[V]** — reason obtained, see §4.1 | local session, 2026-08-19 |
| 9 | Published ACNE04 severity baselines | Our real-only arm should land near this. | **[V]** — but the band is 83.7–87.3% and it is **plain accuracy on an imbalanced test set**, not balanced accuracy | local session, 2026-08-19 |
| 10 | Carlini et al. 2023 / Somepalli et al. 2023 | The replication prior our memorisation audit compares against. | **[V] for Somepalli** (1.88%, lower bound); **[S] for Carlini** (rate not extracted) | local session, 2026-08-19 |
| 11 | Shumailov et al. 2024, Nature, `shumailov2024` | Volume and page numbers. | **[V]** — Nature **631**, 755–759 (2024) confirmed | local session, 2026-08-19 |
| 12 | "When Pretty Isn't Useful" (arXiv:2602.19946), `prettynotuseful2026` | Author list was entirely unverified. | **[V]** — paper is real, authors resolved, CVPR 2026 | local session, 2026-08-19 |
| 13 | Representation-conditioned generation (arXiv:2605.27495), `repcond2026` | Claimed synthetic > real at 3× scale. | **[V]** — paper is real, authors resolved; claimed margin is **+2.0 pp**, not "3× scale" as previously written | local session, 2026-08-19 |

### Verification pass 2 — run 2026-08-28, the six `refs.bib` entries left unverified

These six carried `NOT INDEPENDENTLY VERIFIED` notes after pass 1. Four of them were
cited from `main.tex` and were therefore blocking `make publication-gate`. **Two of the
six were wrong**, which is the third and fourth bibliography defect this project has
found in its own record.

| # | Entry | Status | Checked against |
|---|---|---|---|
| 14 | `tschandl2018` | **[V]** — exact match | Crossref record for `10.1038/sdata.2018.161` |
| 15 | `howard2021` | **[V]** — metadata; the three rho values remain **[S]** | arXiv 2106.11240 abstract record + Crossref for `10.1109/TBIOM.2021.3123550` (TBIOM **3**(4):550–560) |
| 16 | `schaudt2023` | **[V]** — after correcting a wrong title | full text at PMC10741143 |
| 17 | `shen2020` | **[V]** — metadata; the protocol description remains **[S]** | arXiv 2005.09635 abstract record + Crossref for `10.1109/TPAMI.2020.3034267` (TPAMI **44**(4):2004–2018, 2022) |
| 18 | `monteiro2023` | **[V]** — metadata and venue; the axiom description remains **[S]** | arXiv 2303.01274 abstract record, whose journal-reference field names ICLR 2023 |
| 19 | `acnedit2025` | **[V]** — after correcting a wrong title and four wrong given names | Crossref + DBLP `conf/miccai/PiatGNAGNN25` + Semantic Scholar + the authors' release README |

#### 16. `schaudt2023` — the entry conflated two real papers by overlapping authors

The title read *"Augmentation Strategies for an Imbalanced Learning Problem on a Novel
COVID-19 Severity Dataset"*. That paper exists — Scientific Reports **13** (2023),
`10.1038/s41598-023-45532-2` — but it is about classical augmentation and random
oversampling, not generative models, and its author list differs (it adds Hafner and
Riedel and drops Späte). The DOI, journal, volume, issue, page and the seven authors in
the entry all belong to a *different* paper by the overlapping group: *"A Critical
Assessment of Generative Models for Synthetic Data Augmentation on Limited Pneumonia
X-ray Data"*, Bioengineering **10**(12):1421.

The full text of the pneumonia paper is what this project actually draws on, and both
claims check out in it:

- **"Five seeds."** §3.4: *"All classification model trainings have been repeated 5 times
  with a different seed."* **[V]**
- **"An oversampling control that collapses one class to F1 = 0.0000."** Table 9: the
  random-oversampling reference arm scores precision `0.0000 ± 0.0000` and recall
  `0.0000 ± 0.0000` on the bacterial class; the text says it *"completely misses the
  bacterial cases."* The generative arms are compared against it without subtraction.
  **[V]**
- Their own stated limitation, worth carrying: generator training time *"prohibited the
  use of cross-validation"*, so their numbers are one split.

`docs/07_coverage_arm.md` cited it correctly as "Bioengineering 10:1421" throughout; only
the bib title was wrong.

#### 19. `acnedit2025` — title invented, four of seven given names wrong

Springer is paywalled and no preprint exists, so pass 1 built the entry from a search
summary. Crossref, DBLP and Semantic Scholar agree with each other and disagree with what
was in the file:

| field | was | is |
|---|---|---|
| title | *Controllable Acne Severity Editing for Dataset Rebalancing* | *Acne Creation and Non-Destructive Editing with Dynamic Intensity Tuning Using Deep Learning on Facial Images for Dermatological Application* |
| authors | Guillaume Piat, Léa Gazeau, **Thanh** Nguyen, **Adel** Ajem, **Alexandre** Gilibert, Zung Nguyen, Hang Nguyen | **Gauthier** Piat, Léa Gazeau, **Thang** Nguyen, **Marwan** Ajem, **Pierre** Gilibert, Zung Nguyen, Hang Nguyen |
| pages | — | 99–108 |

Volume 16128 and the DGM4MICCAI 2025 venue were right.

The authors' release at `github.com/AIpourlapeau/ACNEDIT` is not the paper, but it is a
primary source written by them, and it verifies everything `main.tex` says about their
intervention: severe 182 + **218** generated = 400, very-severe 129 + **271** generated =
400, mild 513 and moderate 633 untouched. Two things it adds:

- **Their four class counts sum to 1,457** — an independent corroboration of the archive
  count in `docs/05_acne04_audit.md`, which had to correct a "1,513 files" figure that
  itself came from an unverified search summary.
- **Every generated image is an edit of a real ACNE04 face**: a GAN lesion overlay refined
  by a LaMa inpainting model trained on ACNE04, applied to user photographs. That is a
  sharper statement of the closed-set point `main.tex` already makes, and it is now made
  in those terms.

**Still [S], and cited nowhere:** the 59.2% user-study figure and the +7.85% IoU / +8.56%
Dice downstream results. They are held here rather than in `refs.bib` so that no entry in
the bibliography carries an unread number. Read the chapter before either is quoted.

### Verification pass 3 — run 2026-08-28, the two load-bearing citations

`main.tex` makes a specific claim about what each of these two designs *omits*. Both were
read in full on arXiv. **A fifth bibliography defect surfaced.**

| # | Entry | Status | Checked against |
|---|---|---|---|
| 20 | `moroianu2025` | **[V]** — after correcting an invented title and a truncated author list | arXiv HTML full text of 2508.16783 |
| 21 | `sagers2023` | **[V]** — every claim verbatim | arXiv HTML full text of 2308.12453 |

#### 20. `moroianu2025` — title invented, eleven authors recorded as three

| field | was | is |
|---|---|---|
| title | *Scaling Synthetic Data for Chest Radiograph Classification* | *Improving Performance, Robustness, and Fairness of Radiographic AI Models with Finely-Controllable Synthetic Data* |
| authors | **Sofia** Moroianu, Christian Bluethgen, Pierre Chambon | **Stefania L.** Moroianu, Christian Bluethgen, Pierre Chambon, Mehdi Cherti, Jean-Benoit Delbrouck, Magdalini Paschali, Brandon Price, Judy Gichoya, Jenia Jitsev, Curtis P. Langlotz, Akshay S. Chaudhari |

The entry's own note said the title and authors were "TRUNCATED AND NOT VERIFIED", but
`make publication-gate` did not block on it: its pattern matched `NOT INDEPENDENTLY
VERIFIED` and `UNVERIFIED` and not the bare `NOT VERIFIED` this entry happened to use. The
pattern is now widened. **A gate that misses the marker it exists to catch is worse than no
gate**, because the entry reads as having passed one.

The substance survives intact, and is stronger than what the manuscript claimed:

- **The gain.** Abstract: *"synthetic pretraining led to a 6.5% accuracy increase in the
  performance of downstream classification models, compared to a modest 2.7% increase when
  naively combining real and synthetic data."* A synthetic-only arm gains 2.1%. **[V]**
- **The confound.** The manuscript inferred the initialisation mismatch; the paper states
  it. Figure 6's caption, on its four scenarios: *"In scenarios (i)–(iii) models were
  initialized using ImageNet pretrained weights; in scenario (iv) the model was trained
  from scratch"* — (iv) being the synthetic-pretrained arm. `main.tex` now quotes this
  rather than characterising it. **[V]**
- **Units, which nobody had checked.** The 6.5% is a *relative* increase in macro-average
  multi-label AUROC averaged over five evaluation datasets, not percentage points:
  in-distribution on MIMIC-test it is 0.772 → 0.798. Our pretraining nulls (+0.3, +0.9) are
  differences in balanced accuracy in points. The two are not on the same scale and
  `main.tex` now says so. This is the same class of error as item 9 in pass 1, where a
  published ACNE04 band turned out to be plain accuracy rather than balanced accuracy.

#### 21. `sagers2023` — verified, including the missing supplementary figure

Title and all nine authors match. Every claim the manuscript draws is present:

- §4.6, *"Class imbalance experiments"*: *"As a control, we also tested an upsampling
  strategy of simple duplication for the rare disease images."* Its result is reported only
  as *"(Figure S-2)"*. **[V]**
- **The figure really is absent.** The released manuscript carries Figures 1–5 and no
  supplementary section, while citing Figures S-1, S-2 and S-4. arXiv holds a single
  version (v1, 2023-08-23) with no journal reference, and Crossref has no published record,
  so there is no other release in which it appears. **[V]**
- Saturation: *"improvements saturated at synthetic-to-real data ratios above 10:1."* **[V]**
- At 228 real images per class with transforms, accuracy changes by 2.0% (95% CI −1.2 to
  5.3) for inpainting, −0.4% (−3.0 to 2.0) for in-then-outpainting and −2.2% (−5.5 to 1.1)
  for text-to-image, *"with none meeting significance thresholds after adjusting for
  multiple testing."* Every interval spans zero. **[V]**

#### 8. The retraction, verified at source rather than through search indexing

Pass 1 marked this `[V]` but could only reach it through search indexing of the notice
text, because mdpi.com refuses our fetcher. Crossref holds both records and settles the
formal facts: the article is BioMedInformatics **4**(2) 1059–1070, 9 April 2024, the
notice is **4**(3) 1901, 12 August 2024, and the notice's `update-to` field formally
records it as a retraction of the article. Both six-author lists match the bib entry —
with one wrinkle worth keeping, that Crossref gives the article's fourth author as
"Mohammad Hesamian" and the notice's as "Mohammad Hesam Hesamian"; the entry follows the
notice. The *reason* for retraction is still from search indexing: Europe PMC does not
hold the notice either, so that one sentence remains the only unread part of the entry.

### Note on host access

Two hosts refused our fetcher during this pass and their claims lean on secondary
sources. Re-check them from a browser before submission:

- `openaccess.thecvf.com` — 403. Wu et al. ICCV 2019 (item 5) was verified through the
  LDS paper (arXiv:2403.00268) and secondary tabulations, not the ICCV PDF itself.
- `www.mdpi.com` — 403. The retraction notice (item 8) was verified through search
  indexing of the notice text, not the notice page. In pass 2 the same 403 blocked
  `schaudt2023`; PMC serves the identical full text at PMC10741143 and was used instead.
- `link.springer.com` — JavaScript challenge, no content. `acnedit2025` (item 19) was
  resolved through three metadata registries and the authors' own release instead, and
  its evaluation figures remain unread.

Everything else was read in full text or in the arXiv HTML rendering.

## Still outstanding

1. Azizi et al. 2023 — FID 1.76 / IS 239 / CAS 64.96 & 69.24 all still [S].
3. He et al. ICLR 2023 — still [S].

5. Carlini et al. 2023 — the extraction rate, as opposed to the raw count of >1,000
   extracted examples, is still [S].
7. The "4–15% hybrid-vs-real gain" range attributed to the dermatology-specific
   generators (Derm-T2IM, LesionGen, SkinDualGen, DermDiff) is still [S] and is a
   range assembled across papers with incomparable protocols. Either verify each or
   delete the range and describe the variance qualitatively.

8. `acnedit2025` — the 59.2% user-study figure and the +7.85% IoU / +8.56% Dice
   results. Paywalled; see item 19. Not cited anywhere, and must not be until read.
9. Four entries cited from `main.tex` still carry no `VERIFIED <date>` note and are
   reported as warnings by `make publication-gate`: `bissoto2021`, `fan2024`,
   `schmidt2026`, `takezaki2023`. None is load-bearing in the way `moroianu2025` and
   `sagers2023` were — those two were read in pass 3 — but `fan2024` supplies the
   guidance-scale protocol and should be next.

10. ~~**The verification notes print.**~~ **Done 2026-08-28.** `plainnat` renders `note`,
    so every `VERIFIED <date>` provenance note was appearing in the manuscript's
    bibliography — two pages of `main.pdf` by the end of pass 3. All 40 entries moved to
    `annote`, which BibTeX carries and `plainnat` does not print; `main.pdf` is 13 → 11
    pages and `audit.pdf` 15 → 14. The gate reads whole entries, so it is unaffected, and
    a new `printing-note` rule now fails on any `note` field containing a verification
    marker.

11. ~~**Rows 1–13 name papers in prose, not bib keys.**~~ **Done 2026-08-28.** Eleven of
    the thirteen now carry their `refs.bib` key and are cross-checked by
    `make publication-gate`. Two cannot be: row 9 is a band assembled from four separate
    entries (`lds2024`, `kieglfn2022`, `ffpll2025`, `acneai2024`) rather than one source,
    and row 10 is `[V]` for Somepalli and `[S]` for Carlini, so it is not a verified row.
    Wiring row 8 up required verifying it properly first — see below.

## Procedure

1. Read the full text, not the abstract.
2. Confirm every number quoted in `01_related_work.md`, and the venue, year, volume
   and page numbers in `refs.bib`.
3. Change the tag to **[V]** and fill in the verifier and date above.
4. If a claim does not survive, remove it from the review rather than softening it.

Entries in `refs.bib` carrying `AUTHOR LIST UNVERIFIED` or similar notes are
placeholders. The note must be deleted only when the entry has actually been checked.
