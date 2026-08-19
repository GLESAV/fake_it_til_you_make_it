# Running the study on Apple Silicon

Written for an **M4 Max MacBook Pro, 128 GB unified memory**. Everything below is
an estimate to plan against, not a measurement — the first thing to do on the machine
is time one generator step and one classifier epoch and replace these numbers.

## Why this machine is a reasonable host

128 GB of unified memory is the unusual part. The entire study — ACNE04 at 1,457
images, both synthetic pools, and a ResNet-50 — fits in memory with room to spare, so
nothing is I/O bound and no checkpoint juggling is needed. The GPU is slower than an
A100 but it is not competing with anyone for it, and this study's total is measured in
days of wall clock rather than weeks.

The real constraint is **generation throughput**, not training.

## Rough budget

| Stage | Estimate | Note |
|---|---|---|
| LoRA fine-tune, SD 1.5, 512px, 4k steps | 2–4 h | one-off per configuration |
| Generation, SD 1.5, 512px, 50 steps | ~3–5 s/image | the bottleneck |
| Classifier run, ResNet-50, 224px, ~950 train images | 10–20 min | with early stopping |

At 4 s/image, a 20,000-image pool is **~22 hours**. Two pools is nearly two days of
pure generation. That is the number to design around.

### Recommended cuts

1. **Use SD 1.5 at 512px, not SDXL at 1024px.** SDXL is roughly 4–6× slower per image
   here, and the literature (`docs/01_related_work.md` §2.2) gives positive reason to
   think the prettier model is not the better *data* generator. Run SDXL later as an
   ablation on a small pool if you want the comparison.
2. **Shrink the pools.** The substitution sweep never needs more than
   *N* = |train| ≈ 950 synthetic images per arm. A pool of **3,000 per generator**
   gives comfortable per-class headroom and takes ~3.5 h instead of 22.
   Set `generator.n_images: 3000` in `configs/acne04_closed.yaml` and
   `configs/acne04_open.yaml`.
3. **Cap the additive sweep at k=8.** k=16 alone needs 15,200 synthetic images. Drop
   `16` from `multipliers` in `configs/arms/additive_exchange_rate.yaml` and generate
   a **10,000**-image pool for that config specifically (~11 h, run it overnight). If
   the exchange rate is unbounded by k=8, k=16 would not have changed the conclusion;
   if it is not, you will see where the curve is heading and can extend.
4. **Consider LCM-LoRA or a 4–8 step schedule** for the *open-set* arm only. It cuts
   generation by roughly 6×. Do not use it for the closed-set arm without checking it
   against the 50-step pool on validation — few-step distillation reduces diversity,
   which is exactly the variable under study.

### Run-count budget

| Sweep | Runs | Wall clock at 15 min/run |
|---|---|---|
| Substitution, both generators, 5 seeds, `init=imagenet` | 45 | ~11 h |
| Real-subset controls, 5 seeds | 15 | ~4 h |
| Substitution, both generators, 5 seeds, `init=scratch` | 45 | ~15 h (longer schedule) |
| Additive (4 budgets × 5 multipliers × 2 generators × 3 seeds) | 114 | ~28 h |

The p=0 arm is shared across generators and trained once, not once per generator.

**Suggested order:** substitution `init=imagenet` first (that is the headline figure
and the practitioner-relevant setting), then the controls, then `init=scratch`, then
additive. Each stage is independently publishable-adjacent, and if you stop early you
still have a coherent result.

## Setup

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -U pip
pip install -e '.[dev]'          # torch/torchvision wheels are universal2, MPS works
pip install -e '.[generate]'     # diffusers, transformers, accelerate, peft
```

Confirm the accelerator is picked up:

```bash
python -c "import torch; print(torch.backends.mps.is_available())"   # True
python -c "from fitymi.train.loop import TrainConfig; print(TrainConfig().resolve_device())"
```

`device: auto` resolves CUDA → MPS → CPU, so the committed configs need no change.

## Apple-Silicon specifics

- **Mixed precision is off on MPS, by design.** `TrainConfig.amp` only takes effect on
  CUDA. MPS autocast exists but is uneven across ops and silently changes numerics,
  and this study compares arms to each other — a numerics difference that varies by
  op is not a trade worth making. Expect the reported per-run times accordingly.
- **DataLoader workers.** Start at `num_workers: 8`. macOS uses spawn rather than
  fork, so workers are more expensive to start; `persistent_workers` is already on. If
  you see stalls, drop to 4, and to 0 for debugging.
- **`PYTORCH_ENABLE_MPS_FALLBACK=1`** is worth exporting for the diffusers steps. Some
  ops still lack MPS kernels and will otherwise hard-fail rather than fall back to CPU.
- **Thermals.** Sustained multi-hour GPU load will throttle a laptop. Plug in, and
  prefer long unattended overnight blocks over interleaving with interactive work —
  otherwise per-run times drift and the wall-clock estimates above stop meaning
  anything.
- **Resume is built in.** `run_arm` skips any run whose record already exists, so
  closing the lid or killing a sweep costs at most the run in flight. Just re-run the
  same command.

## The one thing not to do

Do not run `--final-eval` until every generator, prompt and hyperparameter decision is
frozen. The test split is sealed in code and every unsealing is logged to
`runs/test_unseal_audit.jsonl`, which ships with the paper. That log is only worth
anything if it is short and late.
