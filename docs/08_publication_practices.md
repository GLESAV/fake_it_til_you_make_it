# Publication practices

Binding, in the same sense that `docs/02_study_design.md` is binding. Nothing leaves this
repository — preprint, submission, blog post, or dataset release — until every gate in §6
passes and the sign-off block at the bottom is filled in.

## Why this document exists

The project's success measure is a published paper that accrues citations. That makes
citation a liability as well as an asset: a paper that is cited is a paper that is read
closely, by people with an interest in finding it wrong. This repository's central asset
is that it has repeatedly caught itself — three retracted causal claims, four latent bugs,
two fabricated author lists, one citation to a paper that does not exist. That record is
worth more than any single result, and exactly one avoidable error in a submitted
manuscript destroys it.

Every resolution below is tied to something that actually happened here. None of them are
hypothetical, and none are imported from a generic checklist.

---

## 1. Integrity of the record

### R1 — No citation enters a manuscript unread

Every `refs.bib` entry cited from a manuscript must carry a `VERIFIED <date>` note naming
what was read: full text, not abstract, not search summary. Entries at `[S]` or `[?]` in
`docs/01_related_work.md` may be discussed in prose but may not be cited as support.

> **Incident.** `148d422` — two author lists in `refs.bib` were fabricated outright and one
> citation pointed to a paper that does not exist. `docs/01_related_work.md:389` records a
> third: an attribution to "Zaghbani / Boukhris et al." that "was wrong and was never
> checked." Six entries still carry `NOT INDEPENDENTLY VERIFIED` as of 2026-08-25.

**Check:** `make publication-gate` fails on any `UNVERIFIED` / `NOT INDEPENDENTLY VERIFIED`
marker in an entry that a manuscript cites, and on any `\cite` key with no `refs.bib` entry.

### R2 — Descriptive and causal claims are labelled as such

Claims computable from files (counts, correlations, rates) and claims about mechanism
(explanations, predictions, proposed fixes) are held to different standards and must be
distinguishable by a reader. Mechanistic claims carry their falsification history.

> **Incident.** Audit §15.1: of five descriptive claims, five held. Of three causal claims,
> three failed — "the leak concentrates on the clinically worrying class" (falsified by
> ISIC 2020), "models exploit the provenance shortcut" (retracted, the ablation was
> invalid), "count-prompting will fix severity fidelity" (falsified, made it worse).

### R3 — Retractions stay in the record, including in the paper

When a claim is withdrawn, the withdrawal and its cause stay visible. They do not get
quietly edited out on the way to submission.

> **Incident.** `7a863cd`, `975aa37`, `d6495bf`, `95072f9`, `797c8f8`. The repo already does
> this in its docs and commit messages. The resolution is that it survives the transition
> into `main.tex` and `audit.tex`, where the temptation to tidy is strongest.

---

## 2. Integrity of measurement

### R4 — Every ablation must be shown to have ablated

After removing a channel, demonstrate it can no longer be recovered. A null from an
unverified intervention is not evidence.

> **Incident.** Audit §15.2. Resolution, aspect ratio and quantisation were normalised to
> remove capture provenance; accuracy did not drop; that was written up as evidence models
> do not exploit provenance. The normalised images still carried source at **99.4%
> accuracy** — watermarks and eye-pixelation survive re-encoding. The ablation removed a
> channel nobody was using and left the obvious one open.

### R5 — Instruments are validated on this corpus before use; published thresholds are recalibrated

A threshold is a property of a corpus, not of an instrument.

> **Incident.** Audit §15.3, twice. The lesion counter scored Spearman 0.103 untuned, 0.365
> after sweeping 24 settings, against ~0.5 for usability — it was measuring skin texture,
> and was only ever tested because ACNE04 happens to publish counts. The SSCD memorisation
> threshold of 0.5, taken from the literature, flags **82.6% of real images** as copies of
> each other on this corpus; the same audit at the published threshold would have reported
> 12.00% memorisation for a generator whose true figure is 0.00%.

### R6 — Report at least two metrics that can disagree, and distrust the easy one improving

> **Incident.** Audit §15.4. Count-prompting raised exact grade agreement 34.8% → 47.9%
> while ordinal fidelity collapsed, Spearman 0.176 → 0.010. The predictions had piled onto
> the modal class, matching more often without being more faithful.

### R7 — Every load-bearing number is reconciled by a second, independent route

> **Incident.** Audit §15.5. Four latent bugs, none of which raised an error and all of
> which produced plausible output: a `.gitignore` pattern that silently dropped
> `src/fitymi/data/` from the pushed repo; run IDs content-addressed over everything except
> the data; generation running on CPU beside an idle 40-core GPU; class weights handing
> absent classes the largest weight. The only reason any surfaced is that a number failed
> to reconcile against a number obtained another way.

### R8 — No effect is reported as a win without its control and its power

An arm that beats the baseline is not a result until (a) it beats a control that isolates
the intervention from its confounds, and (b) the comparison's power is stated.

> **Incident.** `ca195ce`, `8a96aa8`. The tail effect read +2.2 points, then +2.21 at n=2,
> +2.24 at n=6, +1.65 at n=10 — t=1.53, p=0.159, 5/10 seeds positive, CI spanning zero
> since n=2. It needs 22 seeds for 80% power. It is reported as **underpowered, not
> disproven**, and the same standard is applied to the literature it is compared against.

---

## 3. Integrity of claim-staking

### R9 — Prior-art check before any novelty claim, repeated at submission

One literature pass is weak evidence of absence. Any "first" or "novel" claim is re-checked
immediately before submission, and again before camera-ready.

> **Incident.** `0233889` — the ITA calibration-standard method turned out to be scooped
> after the framing had already been built on it. `3a8c5e9` — ACNEDIT (Piat et al.,
> DGM4MICCAI 2025) uses the same dataset, the same two scarce classes and the same
> intervention. Audit §15 says it of itself: "absence of evidence from one pass is weak."

### R10 — "Preregistered" is claimed only alongside the amendment list

`docs/02_study_design.md` was fixed before any experiment ran, which is a genuine strength.
It has also been amended at least three times (§12: mandatory subject-disjoint splitting,
label-robustness control, identity-diversity measurement). Claiming the first without
disclosing the second is misleading, and is the kind of thing a reviewer checks.

---

## 4. Publication-facing obligations

These are the gates the repository does **not** currently satisfy. They are not research
hygiene; they are conditions of submission.

### R11 — AI-assistance is disclosed, and AI is not listed as an author

This work was produced substantially by an autonomous AI agent: the codebase, the
literature review, the audit, the analysis and the manuscript drafts. Essentially every
venue in scope now requires this to be disclosed and prohibits AI authorship — ICMJE,
Nature, Science, PLOS, Elsevier, and the major ML conferences all have standing policy.
Non-disclosure discovered post-publication is a retraction-class event, which under a
citation metric is the worst available outcome.

**Action:** draft the disclosure paragraph now, not at submission. It states what was
AI-generated, what a human verified, and who takes responsibility for the contents. A
human author must be in a position to defend every claim — which is what §6's sign-off is
for.

### R12 — Human-subjects posture is determined and stated, not assumed

ACNE04 is ~1,457 photographs of ~600 identifiable people with a visible medical condition.
Secondary analysis of a public dataset is usually exempt, but "usually exempt" is a
determination someone has to actually make and record. Clinical and derm venues will ask.

**Action:** obtain and record the exemption determination (or its local equivalent) before
submission. `main.tex` §Ethics currently covers data redistribution and synthetic-image
sensitivity but says nothing about the real subjects.

### R13 — Responsible disclosure to dataset authors before the audit is public

The audit is a criticism of named researchers' published work: their folds leak, their
labels agree with an independent re-annotation on 30.1% of images, their filenames
mislead. Standard practice is to notify the authors, share the findings, and offer right
of reply before publication.

This is also the self-interested move. An audit the original authors have seen and not
disputed is far more citable than one they first read in public and contest.

**Action:** contact the ACNE04 authors (Wu et al., `github.com/xpwu95/LDL`) with the audit
and a stated response window before any preprint goes up.

### R14 — Licensing and redistribution are settled before release

- ACNE04 is academic-use only and is not redistributed. Already honoured; keep it.
- `README.md` promises `LICENSE` (MIT) and **no such file exists**. Fixed in this pass, but
  the copyright holder line needs a real name.
- The generated pools were produced through a commercial image API. Whether the images may
  be released, and under what terms, is a question about that provider's terms of service
  and has not been answered anywhere in this repo.
- `data/synthetic/*/manifest.jsonl` records model, prompt, attempts and cost per image.
  Keep it; it is what makes a release defensible.

### R15 — Correction policy after publication

If a published claim fails later, the correction is issued by us, promptly, and linked from
the repository. Written down now, while it costs nothing to agree to.

---

## 5. What is deliberately not a practice here

Optimising for citations and optimising for correctness are not the same objective, and
where they conflict this document resolves in favour of correctness. Specifically: results
are not reframed to look more positive than they are, nulls are reported as nulls, and the
underpowered tail effect is not upgraded to a finding because a finding would cite better.

---

## 6. The pre-publication gate

Run before any preprint, submission, or release. Mechanical checks first:

```bash
make publication-gate
```

That covers: no `\RESULT{}` / `\TODO{}` placeholders; every `\cite` key resolves; every
`\ref` resolves to a `\label`; no cited entry carries an unverified marker; every source
`docs/VERIFY.md` marks `[V]` says so in its `refs.bib` entry too; verification provenance
sits in `annote` rather than the `note` field `plainnat` prints; `LICENSE` exists and names
a holder; the README's verification-status claim and its count of verified sources are not
stale.

Five of those eight rules were added after the thing they check had already gone wrong
here — a pattern that missed the marker it existed to catch, a README count that went
stale within one verification pass, a source verified in one file and not the other, two
pages of provenance notes rendering into the manuscript's bibliography, and a dangling
cross-reference that compiled cleanly and printed as `??`. None of them is hypothetical
either. The rule that follows: **when a check misses something, the fix is the check and
not just the instance.**

Everything below needs a person:

| # | Gate | Resolution | Status |
|---|---|---|---|
| 1 | Mechanical checks pass | R1 | `make publication-gate` |
| 2 | Every cited work read in full text by a human | R1 | ☐ |
| 3 | Descriptive vs causal claims labelled in the manuscript | R2 | ☐ |
| 4 | Retracted claims still visible in the paper | R3 | ☐ |
| 5 | Every ablation shown to have ablated | R4 | ☐ |
| 6 | Every instrument validated on this corpus | R5 | ☐ |
| 7 | Every headline number reconciled by a second route | R7 | ☐ |
| 8 | Every reported effect has its control and its power | R8 | ☐ |
| 9 | Prior-art pass repeated within 30 days of submission | R9 | ☐ |
| 10 | Preregistration claim lists its amendments | R10 | ☐ |
| 11 | AI-assistance disclosure drafted and venue-checked | R11 | ☐ |
| 12 | Human-subjects determination obtained and recorded | R12 | ☐ |
| 13 | ACNE04 authors notified, response window elapsed | R13 | ☐ |
| 14 | Licensing and image-release terms settled | R14 | ☐ |
| 15 | Correction policy published in the repo | R15 | ☐ |

**Sign-off.** A named human states: *I have read the full text of every work this
manuscript cites, I can defend every number in it, and I take responsibility for its
contents.*

    Signed: ____________________   Date: __________   Manuscript: __________

Unsigned means unpublished. That is the whole point of the document.
