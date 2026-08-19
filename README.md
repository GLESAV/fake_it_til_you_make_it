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
| Codebase | 🚧 next |
| Experiments | ⛔ needs a GPU host (see `docs/03_environment.md`) |
| Paper | ⛔ |

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

## Layout

```
docs/     literature review, protocol, environment audit
paper/    arXiv manuscript (LaTeX)
```

## Licence

Code: MIT (see `LICENSE`). Data: ACNE04 is academic-use only and is **not** redistributed
here; see `docs/02_study_design.md` §11.
