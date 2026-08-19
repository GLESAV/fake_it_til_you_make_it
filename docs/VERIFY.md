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
| 1 | **Zein** et al. 2024, PLOS ONE 19(4):e0297958 (arXiv:2211.04214) | The 97.6% synthetic-trained / real-tested acne claim. | **[V]** — with major corrections, see §4.1 | local session, 2026-08-19 |
| 2 | Akrout et al. 2023 (arXiv:2301.04802) | Closest medical analogue to our fully-synthetic arm. | **[V]** — full results table read | local session, 2026-08-19 |
| 3 | Wang et al. 2025 (arXiv:2508.09550) | Exchange-rate methodology, closed-/open-set distinction. | **[V]** | local session, 2026-08-19 |
| 4 | Fan et al. 2024 (arXiv:2312.04567) | Guidance-scale and scaling protocol. | **[V]** — CFG values confirmed; **CVPR 2024 acceptance NOT confirmed on the arXiv page** | local session, 2026-08-19 |
| 5 | Wu et al. 2019, ICCV | ACNE04 splits, Hayashi mapping, image count. | **[V]** — via secondary full texts; thecvf.com returns 403 to our fetcher, see note below | local session, 2026-08-19 |
| 6 | Sariyildiz et al. 2023, CVPR | Prompt strategy; how much of the gap closed. | **[S]** — abstract only; numbers still unverified | — |
| 7 | Ktena et al. 2024, Nature Medicine | Fairness protocol; volume/pages. | **[V]** — Nat Med **30**, 1166–1173 (2024) confirmed | local session, 2026-08-19 |
| 8 | Retracted acne-GAN paper (doi:10.3390/biomedinformatics4020059) | Whether anything adjacent is safe to cite. | **[V]** — reason obtained, see §4.1 | local session, 2026-08-19 |
| 9 | Published ACNE04 severity baselines | Our real-only arm should land near this. | **[V]** — but the band is 83.7–87.3% and it is **plain accuracy on an imbalanced test set**, not balanced accuracy | local session, 2026-08-19 |
| 10 | Carlini et al. 2023 / Somepalli et al. 2023 | The replication prior our memorisation audit compares against. | **[V] for Somepalli** (1.88%, lower bound); **[S] for Carlini** (rate not extracted) | local session, 2026-08-19 |
| 11 | Shumailov et al. 2024, Nature | Volume and page numbers. | **[S]** | — |
| 12 | "When Pretty Isn't Useful" (arXiv:2602.19946) | Author list was entirely unverified. | **[V]** — paper is real, authors resolved, CVPR 2026 | local session, 2026-08-19 |
| 13 | Representation-conditioned generation (arXiv:2605.27495) | Claimed synthetic > real at 3× scale. | **[V]** — paper is real, authors resolved; claimed margin is **+2.0 pp**, not "3× scale" as previously written | local session, 2026-08-19 |

### Note on host access

Two hosts refused our fetcher during this pass and their claims lean on secondary
sources. Re-check them from a browser before submission:

- `openaccess.thecvf.com` — 403. Wu et al. ICCV 2019 (item 5) was verified through the
  LDS paper (arXiv:2403.00268) and secondary tabulations, not the ICCV PDF itself.
- `www.mdpi.com` — 403. The retraction notice (item 8) was verified through search
  indexing of the notice text, not the notice page.

Everything else was read in full text or in the arXiv HTML rendering.

## Still outstanding

1. Sariyildiz et al. CVPR 2023 — the actual accuracy numbers and prompt ablation.
   Currently the review only makes a qualitative claim, which is safe but weak.
2. Azizi et al. 2023 — FID 1.76 / IS 239 / CAS 64.96 & 69.24 all still [S].
3. He et al. ICLR 2023 — still [S].
4. Sagers et al. 2022 — still [S].
5. Shumailov et al. 2024 — volume/pages still [S].
6. Carlini et al. 2023 — the extraction rate, as opposed to the raw count of >1,000
   extracted examples, is still [S].
7. The "4–15% hybrid-vs-real gain" range attributed to the dermatology-specific
   generators (Derm-T2IM, LesionGen, SkinDualGen, DermDiff) is still [S] and is a
   range assembled across papers with incomparable protocols. Either verify each or
   delete the range and describe the variance qualitatively.

## Procedure

1. Read the full text, not the abstract.
2. Confirm every number quoted in `01_related_work.md`, and the venue, year, volume
   and page numbers in `refs.bib`.
3. Change the tag to **[V]** and fill in the verifier and date above.
4. If a claim does not survive, remove it from the review rather than softening it.

Entries in `refs.bib` carrying `AUTHOR LIST UNVERIFIED` or similar notes are
placeholders. The note must be deleted only when the entry has actually been checked.
