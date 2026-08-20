# How to check whether any of this is true

Written for someone who did not do the work and has no reason to trust it. The claims
differ enormously in how much they depend on my code, so they are sorted by that. **Start
at tier 1** — if it fails, nothing above it matters.

A note on why this document exists in the form it does: over the session that produced this
work, four separate times reading the implementation contradicted what the surrounding code
or my own writeup implied. Two of those would have shipped as results. That is a reason for
calibrated scepticism about anything here that has not been independently reproduced, and
it is also why the tiers below are ordered by *independence from me* rather than by
importance.

---

## Tier 1 — checkable in five minutes, with none of my code

These need only a public URL and pandas. If they reproduce, the load-bearing cross-dataset
claim is real regardless of anything else in this repository.

### HAM10000: does the leak land on melanoma?

```bash
python scripts/standalone_verify_ham10000.py
```

Twenty lines, no repo imports, no images downloaded. Expect roughly:

```
10015 images, 7470 lesions, 1.34 images per lesion
   mel:  68.3%      <- melanoma
   ALL:  38.7%
    nv:  29.6%      <- nevus
```

**If melanoma does not come out around twice the nevus rate, the central generalisation is
wrong.** This is the single highest-leverage check in the document.

### SCIN: are there several images per case?

```bash
curl -sL -o scin.csv https://storage.googleapis.com/dx-scin-public-data/dataset/scin_cases.csv
python - <<'EOF'
import pandas as pd
d = pd.read_csv("scin.csv")
cols = [c for c in d.columns if c.startswith("image_") and c.endswith("_path")]
n = d[cols].notna().sum(axis=1)
print(f"{n.sum()} images / {len(d)} cases = {n.sum()/len(d):.2f} per case")
print(f"{100*(n>1).mean():.1f}% of cases have more than one image")
EOF
```

Expect 2.07 images per case, 61.3% multi-image. The schema itself is the evidence; the
leakage rate follows arithmetically.

---

## Tier 2 — my scripts, but on data you fetch yourself (about an hour)

Get ACNE04 from `github.com/xpwu95/LDL` (academic use only), extract to `data/acne04/`,
then:

```bash
python scripts/audit_acne04.py                  # duplicates, fold leakage, label noise
python scripts/compare_acne04_versions.py --v2 <acne04v2>/Acne04-v2_annotations.json
```

**What to check rather than take on trust:**

- The 38 duplicate groups are an MD5 `groupby`. Verify a few by hand: `levle0_120.jpg` and
  `levle1_486.jpg` should be byte-identical and carry different grades. `cmp` them.
- The fold leakage intersects those groups with the shipped `NNEW_*.txt` files. Grep two
  filenames from a reported straddling pair and confirm one is in trainval and the other in
  test for that fold.
- The v1-versus-v2 comparison needs the public ACNE04-v2 JSON. The quantile-matching step
  is the one to scrutinise — it is my construction, not theirs, and §4.4 of
  `docs/05_acne04_audit.md` says so. If you disagree with it, the raw counts are right
  there and the Spearman *ρ* of 0.471 does not depend on it.

---

## Tier 3 — depends on my instrument, so look at the pictures

The claim that ACNE04's 1,457 images are ~600 people rests on ArcFace clustering. I ran
three controls, and a sceptic should check the third by eye rather than trusting the first
two:

1. **Chance rate.** At cosine 0.85 only 0.019% of all image pairs qualify, against a 15.9%
   hit rate — roughly 800× chance.
2. **Capture-condition control.** Restricted to one camera and one setup (1,107 images),
   the median pairwise cosine is 0.103 and 0.42% of pairs clear 0.60 — so the clusters are
   not photo shoots.
3. **Just look.** `scripts/audit_acne04_subjects.py` plus the cluster montages. Each row
   should obviously be one person. If the rows look like different people to you, say so —
   that is the whole finding.

**Where I would push back on myself:** the identity threshold of 0.60 was chosen by
stability sweep and eye, then retrospectively endorsed by the calibration procedure
(minimum 0.458). That is not independent confirmation — the same person chose both. The
conservative figure, 15.9% at cosine 0.85, is the one I would defend, and it does not
depend on the threshold choice.

---

## Tier 4 — depends on my whole pipeline, so trust it least

**The +4.5 balanced-accuracy points.** This requires my splitter, my training loop, my
metrics, and my statistics to all be right. It is the number I would most want replicated
independently.

Arguments that it is not an artefact, in the order I find them convincing:

- The per-class pattern (−1.5, +0.8, +9.9, +9.0) matches what the mechanism predicts in
  advance, and a test-set difficulty confound has no reason to land on exactly the two
  classes with the fewest subjects.
- A second analysis with an *opposite* confound (within-model, §6.4) finds the same
  ordering.
- The two arms reproduce bit-identically across reruns, so it is not run-to-run noise.

Arguments for scepticism:

- Five seeds. The interval is [+2.33, +7.04], which is wide.
- Validation, not test — the test split is still sealed, so this is not a final number.
- Both arms came out of one codebase that I also wrote and repeatedly found bugs in.

---

## What I would check about novelty, not just correctness

I ran one literature pass. That is thin.

- HAM10000's duplication is **already published** (Abhishek et al., *Nature Scientific
  Data* 12:196, 2025), and the writeup credits them. What is claimed as new is the
  class-conditional shape and the cross-dataset comparison. **Verify that nobody has
  published the class-conditional breakdown** — search "HAM10000 lesion-level leakage
  melanoma" and check the CleanPatrick and DermaMNIST-C papers properly.
- I found no ACNE04 audit. Absence of evidence from one pass is weak. Ask someone in
  dermatology ML.
- The threshold-calibration argument feels like something the copy-detection literature
  should already know. I did not find it stated this way, but I would not bet heavily.

---

## The cheapest external check

Send the artifact and the three standalone scripts to one person who works on dermatology
ML and one who works on dataset auditing. The specific question worth asking is not "is
this interesting" but:

> *Is it already standard practice in your area to split these datasets by patient or
> lesion rather than by image? If so, who does it, and does the ACNE04 literature?*

If the answer is "everyone already does this", the contribution shrinks to the ACNE04
specifics and the calibration argument. If it is "no, and I had not thought about it",
the three-dataset result stands.

---

## Things I know are unverified

- Two `AUTHOR LIST UNVERIFIED` notes remain in `paper/refs.bib` (LDS, KIEGLFN); their
  numbers are verified, their author lists are not.
- `openaccess.thecvf.com` and `www.mdpi.com` refused the fetcher, so Wu et al. ICCV 2019
  and the BioMedInformatics retraction notice were verified through secondary sources.
- No LaTeX toolchain on this machine, so neither manuscript has been compiled.
- The substitution study has produced **no results at all**. Everything above is the audit.
