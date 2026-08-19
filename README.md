# fake_it_til_you_make_it

**Can generated images train a traditional classifier? A controlled study on acne
severity grading.**

The name is a nod to [Sariyildiz et al., CVPR 2023](https://arxiv.org/abs/2212.08420).
The question they asked of ImageNet, we ask of a small, imbalanced, clinically
meaningful dataset — where the answer is much less obvious and much more consequential.

## The question

Most published work shows that adding synthetic images to real ones *helps*. That is a
weak claim and it is well established. We are after the two harder ones:

- **Substitution** — can synthetic images *replace* real ones, at a matched training budget?
- **Exchange rate** — how many synthetic images buy one real image, and does that number
  blow up as you scale?

Concretely: train a fixed classifier on mixtures of real and generated acne images at
**0% / 25% / 50% / 75% / 100% synthetic**, holding the total number of training images
constant, and evaluate every arm on the *same sealed real test set*.

## Status

| | |
|---|---|
| Literature review | ✅ `docs/01_related_work.md` |
| Study protocol | ✅ `docs/02_study_design.md` |
| Environment audit | ✅ `docs/03_environment.md` |
| Codebase | ✅ `src/fitymi/`, verified end-to-end on procedural data |
| Experiments | ⛔ needs a GPU host + ACNE04 (see `docs/03_environment.md`) |
| Paper | 🚧 scaffold in `paper/`, results placeholders unfilled |

## What we found in the literature

Short version, with the long version and citations in `docs/01_related_work.md`:

- General vision: synthetic-only training closes *part* of the gap to real data but
  **still underperforms for supervised classifiers**, and the gap is attributed to
  generator capability and to a fidelity–diversity trade-off — prettier generators are
  not better data generators.
- Dermatology: consistent gains reported for synthetic **augmentation**, especially for
  underrepresented skin tones. Almost nothing on **substitution**.
- Acne: exactly one prior fully-synthetic result we could find — a StyleGAN2 study
  reporting **97.6%** accuracy training on synthetic and testing on real, which is
  ~11 points *above* the best real-trained ACNE04 result we could find. We think that
  number needs replication under budget control, and replicating it is a large part of
  the point of this repo.
- Nobody has published the mixing curve. That is the gap.

## Design commitments

The four that most of this literature gets wrong, and that we bind ourselves to:

1. **Sealed test set** — the real test split touches nothing: not classifier training,
   not generator training, not prompt or hyperparameter selection. Enforced in code.
2. **Matched budgets** — every arm trains on the same number of images, so "synthetic
   helps" cannot be confused with "more data helps."
3. **Closed-set and open-set generators reported separately** — a generator fine-tuned on
   our real training data and one that has never seen it answer different questions.
4. **Pretraining is a variable, not an assumption** — a "100% synthetic" arm on an
   ImageNet-pretrained backbone has seen 1.3M real images. We report from-scratch and
   pretrained separately.

## Running it

```bash
make install          # venv + package
make test             # 60 unit tests, CPU, ~8s
make smoke            # end-to-end pipeline check on procedural data, CPU, ~20 min
make smoke-null       # the null: gap=0, mixing curve must come out flat
```

`make smoke` runs the *same code paths* as the real study — dedup, group-stratified
splitting, sealed-test enforcement, budget-matched mixing, training, evaluation,
statistics, figures — against a procedural generative process whose synthetic arm is
mis-specified by a tunable `gap`. With `gap>0` the mixing curve must slope down and
the trend test must detect it; with `gap=0` the two processes are identical and the
slope interval must contain zero. If either fails, the analysis code isn't fit to
point at real data.

The mis-specification models **concept error** — the generator's rendered severity
drifts toward the corpus mean, so an image labelled "very severe" may depict a
moderate case. That matters: a first version only narrowed the within-grade
distributions, and synthetic-trained models came out *better*, because cleaner
prototypes of each class are easier to learn from. A gap that doesn't corrupt the
label-image relationship isn't the gap this study is about.

The real study needs a GPU host and ACNE04:

```bash
make prepare CONFIG=configs/acne04_closed.yaml   # dedup, split, seal the test set
make finetune                                     # closed-set generator (GPU)
make generate                                     # sample the synthetic pool (GPU)
make sweep                                        # mixing sweep, validation only
make controls                                     # §8 controls
make final                                        # unseals the test set. Once.
make analyse
```

## Layout

```
docs/          literature review, protocol, environment audit, citation checklist
src/fitymi/
  data/        records, ACNE04 loader, dedup, splits, mixing, toy simulator
  generate/    prompts, closed-set fine-tuning, sampling
  train/       backbones and the fixed training recipe
  eval/        metrics and statistics
  controls/    discriminability probe, memorisation audit, skin-tone stratification
  analysis/    aggregation and figures
configs/       one YAML per experimental arm
scripts/       GPU-host driver scripts
tests/         60 tests, no GPU or network required
paper/         arXiv manuscript (LaTeX)
```

## A note on the literature review

Every claim in `docs/01_related_work.md` carries a `[V]`/`[S]`/`[?]` verification tag,
and **all of them are currently `[S]`** — sourced from search summaries, because the
authoring environment blocked every publisher host. `docs/VERIFY.md` is the checklist
for promoting them. Nothing goes into the manuscript at `[S]`.

## Licence

Code: MIT (see `LICENSE`). Data: ACNE04 is academic-use only and is **not** redistributed
here; see `docs/02_study_design.md` §11.
