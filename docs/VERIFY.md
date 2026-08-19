# Citation verification checklist

The literature review and `paper/refs.bib` were assembled in a sandbox with no
access to publisher hosts (`docs/03_environment.md`). Nothing in them has been read
in full text. **No claim may enter the manuscript until it is verified here.**

Tags used in `docs/01_related_work.md`:

- **[V]** verified against the primary source (full text read)
- **[S]** from an abstract or search summary only
- **[?]** believed true, no citable source located

## Status

| # | Source | Why it matters | Status | Verified by / date |
|---|---|---|---|---|
| 1 | Zaghbani et al. 2024, PLOS ONE (arXiv:2211.04214) | The 97.6% synthetic-trained / real-tested acne claim. Our study exists largely to replicate it. Need: exact test set, whether generator training data and test data were disjoint, and what the synthetic healthy class contained. | [S] | — |
| 2 | Akrout et al. 2023 (arXiv:2301.04802) | Closest medical analogue to our fully-synthetic arm. Need the actual fully-synthetic numbers. | [S] | — |
| 3 | Wang et al. 2025 (arXiv:2508.09550) | Exchange-rate methodology and the closed-/open-set distinction we adopt. | [S] | — |
| 4 | Fan et al. 2024, CVPR (arXiv:2312.04567) | Guidance-scale and scaling protocol. Need the exact sweep and the supervised-classifier gap size. | [S] | — |
| 5 | Wu et al. 2019, ICCV | ACNE04 splits, the Hayashi grade mapping, and the true image count (1,457 vs 1,513 files). | [S] | — |
| 6 | Sariyildiz et al. 2023, CVPR | Prompt strategy; how much of the gap actually closed. | [S] | — |
| 7 | Ktena et al. 2024, Nature Medicine | Fairness evaluation protocol. Volume/pages need checking. | [S] | — |
| 8 | Retracted acne-GAN paper (doi:10.3390/biomedinformatics4020059) | Read the retraction notice. Determines whether anything adjacent is safe to cite. | [S] | — |
| 9 | Published ACNE04 severity baselines (~86% accuracy) | Our real-only arm should land near this. If it does not, our pipeline is broken, not the synthetic data. | [S] | — |
| 10 | Carlini et al. 2023 / Somepalli et al. 2023 | The 0.5–2% replication prior our memorisation audit is compared against. | [S] | — |
| 11 | Shumailov et al. 2024, Nature | Volume and page numbers. Also read arXiv:2410.12954, the critical note. | [S] | — |
| 12 | "When Pretty Isn't Useful" (arXiv:2602.19946) | Author list entirely unverified. Do not cite until resolved. | [S] | — |

## Procedure

1. Read the full text, not the abstract.
2. Confirm every number quoted in `01_related_work.md`, and the venue, year, volume
   and page numbers in `refs.bib`.
3. Change the tag to **[V]** and fill in the verifier and date above.
4. If a claim does not survive, remove it from the review rather than softening it.

Entries in `refs.bib` carrying `AUTHOR LIST UNVERIFIED` or similar notes are
placeholders. The note must be deleted only when the entry has actually been checked.
