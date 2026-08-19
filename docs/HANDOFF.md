# Handoff: continuing on local hardware

Paste the block below into Claude Code on the Mac, in an empty directory. It is
written to be self-contained — a fresh session has none of the context from the
remote one.

---

```
We're continuing a research project on whether generated images can substitute for
real ones when training a traditional classifier, using acne severity grading as the
testbed. The research phase and the full codebase are already done and pushed. Your
job is to run the actual experiments on this machine.

Start here:

  git clone https://github.com/GLESAV/fake_it_til_you_make_it
  cd fake_it_til_you_make_it
  git checkout claude/generated-images-acne-classifier-ckwuar

Read these first, in order, before touching anything:
  - README.md
  - docs/02_study_design.md   <- the pre-registered protocol. It is binding.
  - docs/04_running_locally.md <- compute budget for this machine (M4 Max, 128 GB)
  - docs/03_environment.md    <- what the remote sandbox could and couldn't do

The question, precisely: train a fixed classifier on mixtures of real and generated
acne images at 0/25/50/75/100 percent synthetic, holding the TOTAL number of training
images constant, and evaluate every arm on the same sealed real test set. Then measure
the exchange rate: how many synthetic images buy one real image.

Work through these in order. Stop and tell me at each numbered boundary.

1. SET UP AND VERIFY THE PIPELINE.
   python3.11 -m venv .venv && source .venv/bin/activate
   pip install -U pip && pip install -e '.[dev]'
   make test      # 73 tests, no network, ~10s
   make smoke     # end-to-end on procedural data, CPU, ~18 min
   The smoke must PASS: the substitution curve has to slope down and the trend test
   has to detect it. Compare against docs/examples/, which holds the reference run.
   Also run: make smoke-null   # gap=0, the curve must come out FLAT
   If either fails, stop and debug before going near real data.

2. VERIFY THE LITERATURE.
   This is the highest-value thing you can do that the remote session could not: it
   had arxiv, nature, springer, sciencedirect and PMC all blocked, so every claim in
   docs/01_related_work.md came from search summaries and is tagged [S]. You have
   open network. Work docs/VERIFY.md top to bottom, read the actual PDFs, and promote
   claims to [V] or delete them. Priority order is already in that file. Item 1 is
   the Zaghbani PLOS ONE paper reporting 97.6% training on synthetic and testing on
   real — I want to know exactly what their test set was and whether the generator's
   training data was disjoint from it. Also fix the placeholder author lists in
   paper/refs.bib that are marked UNVERIFIED.

3. GET ACNE04.
   From https://github.com/xpwu95/LDL (academic use only; Google Drive or Baidu).
   Extract to data/acne04/. Do not commit the images.
   Then: make prepare CONFIG=configs/acne04_closed.yaml
   READ data/splits/dedup_report.json BEFORE CONTINUING. Check
   largest_cluster_fraction and over_clustered. The archive ships more files (1,513)
   than labelled images (1,457) and the images are half-face crops, so near-duplicates
   are likely. If dedup over-clusters, tune phash_max_hamming and add a CLIP embedder
   rather than accepting a degenerate split — the code will refuse a starved split but
   it cannot pick the thresholds for you. The dedup report is a paper artefact.

4. BUILD THE GENERATORS.
   Closed-set: bash scripts/finetune_closed_set.sh   (LoRA on the real TRAIN split
   only; the no-leakage precondition is enforced in code and will refuse otherwise).
   Then the guidance sweep — protocol §4.4, selected on VALIDATION only.
   Use SD 1.5 at 512px, not SDXL: see docs/04_running_locally.md for why, and shrink
   the pools to ~3,000 images per generator before you generate anything. At ~4 s per
   image the committed 20,000 default is a 22-hour job and you do not need it.

5. RUN THE CONTROLS BEFORE THE SWEEP.
   make controls CONFIG=configs/acne04_closed.yaml
   Two numbers matter. The real-vs-synthetic discriminability AUC bounds how much of
   any mixing effect could be artefact-driven. The memorisation rate tells us whether
   the closed-set generator is just reproducing its training set — if it is, the
   synthetic-only arm is a laundered copy of the real data and the study measures
   nothing. Swap the placeholder pixel embedder for CLIP and SSCD; you have network
   now. Tell me both numbers before running the sweep.

6. RUN THE SWEEP.
   make sweep CONFIG=configs/acne04_closed.yaml
   make sweep CONFIG=configs/acne04_open.yaml
   Validation only at this stage. Runs resume automatically, so closing the lid costs
   at most the run in flight. Then make analyse.

7. FINAL EVALUATION — ONCE, AND ONLY WHEN EVERYTHING ELSE IS FROZEN.
   bash scripts/run_final_eval.sh

Rules that are not negotiable, because the whole point of this study is that the
existing literature doesn't hold to them:

- The real test split is sealed. It is not used for training, early stopping,
  generator training, prompt selection, guidance selection, hyperparameter search, or
  deciding when to stop. Every unsealing is logged to runs/test_unseal_audit.jsonl and
  that log goes in the paper. Keep it short and late.
- Training budgets stay matched across arms. Never "just add" synthetic images to an
  arm to make it look better — that turns a substitution result into an augmentation
  result, which is the mistake most of the prior work makes.
- Closed-set and open-set results are never pooled. Nor are scratch and imagenet
  initialisations: a "100% synthetic" arm on a pretrained backbone has still consumed
  ~1.3M real photographs.
- Hyperparameters are tuned once, on the p=0 arm, then frozen. No per-arm tuning.
- If an arm BEATS the real baseline, do not report it as a win until you have run the
  prototype-effect checks in protocol §8.6. A generator that narrows within-class
  variation makes cleaner prototypes, which are easier to learn from, and that can
  look like a win without the generator having contributed anything. We hit exactly
  this while building the simulator, and it is a plausible explanation for the acne
  prior's 97.6%.

Work on branch claude/generated-images-acne-classifier-ckwuar and push regularly.
The open draft PR is #1. Don't reorganise the paper around whichever way the results
land — the structure in paper/main.tex was fixed before any experiment ran, on
purpose.

Tell me what you find at each stage rather than running the whole thing through.
```
