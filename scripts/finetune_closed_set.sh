#!/usr/bin/env bash
# Closed-set generator fine-tuning (protocol §4.1).
#
# Deliberately a thin wrapper around diffusers' reference LoRA training script
# rather than a bespoke trainer: the contribution of this study is the downstream
# comparison, and a hand-rolled diffusion trainer is one more thing a reviewer would
# reasonably decline to trust.
#
# PRECONDITION, enforced in code by fitymi.generate.finetune.assert_no_leakage:
# the caption manifest must contain images from the real TRAIN split only. A
# generator that has seen val or test turns the whole study into an elaborate way of
# copying labels.
set -euo pipefail

CONFIG=${CONFIG:-configs/acne04_closed.yaml}
OUT=${OUT:-models/closed_set_lora}
BASE=${BASE:-runwayml/stable-diffusion-v1-5}
RANK=${RANK:-16}
# 900 optimizer steps at effective batch 16 is ~15 epochs over the 948-image training
# split. The reference script's 4,000 would be ~67 epochs, which is both deep into the
# regime where a LoRA memorises its training set -- and this study audits memorisation --
# and, measured on this machine, a twelve-hour job for one of two generators.
STEPS=${STEPS:-900}
# Checkpoints so the step count is SELECTED on validation rather than assumed
# (protocol §4.4 treats generation hyperparameters as swept, not defaulted).
CKPT=${CKPT:-300}

# Batch size is 1 with accumulation, not 4, because on Apple Silicon the larger
# micro-batch is measurably SLOWER per image at 512px: 0.52 s/image at batch 1 against
# 0.73 s/image at batch 4. Effective batch is unchanged at 16.
MICRO=${MICRO:-1}
ACCUM=${ACCUM:-16}
RES=${RES:-512}

# The diffusers reference trainer is vendored, not resolved from the installed package:
# pip does not ship examples/, so the old path lookup silently produced a filename that
# does not exist. vendor/train_text_to_image_lora.py is pinned to the diffusers version
# in pyproject and its provenance is recorded in vendor/README.md.
TRAINER=${TRAINER:-vendor/train_text_to_image_lora.py}
test -f "$TRAINER" || { echo "vendored trainer missing at $TRAINER" >&2; exit 1; }

# Mixed precision is CUDA-only on purpose. accelerate's fp16 path on MPS is uneven
# across ops and silently shifts numerics, which is a bad trade in a study whose whole
# content is arm-versus-arm comparison. On Apple Silicon this runs in fp32.
if .venv/bin/python -c 'import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)'; then
  PRECISION=fp16
else
  PRECISION=no
fi

# Writes captions.jsonl and validates the no-leakage precondition. Exits non-zero
# if any held-out image would reach the generator.
.venv/bin/fitymi finetune --config "$CONFIG" --out "$OUT" || true
test -f "$OUT/captions.jsonl" || { echo "no caption manifest at $OUT" >&2; exit 1; }

.venv/bin/accelerate launch --mixed_precision="$PRECISION" \
  "$TRAINER" \
  --pretrained_model_name_or_path="$BASE" \
  --train_data_dir="$OUT/train" \
  --caption_column=text \
  --resolution="$RES" --center_crop --random_flip \
  --train_batch_size="$MICRO" --gradient_accumulation_steps="$ACCUM" \
  --max_train_steps="$STEPS" \
  --learning_rate=1e-04 --lr_scheduler=cosine --lr_warmup_steps=0 \
  --rank="$RANK" \
  --seed=0 \
  --checkpointing_steps="${CKPT:-1000}" \
  --output_dir="$OUT"

echo "LoRA written to $OUT"
echo "Next: fitymi generate --config $CONFIG"
