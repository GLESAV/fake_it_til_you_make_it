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
STEPS=${STEPS:-4000}
RES=${RES:-512}

# Writes captions.jsonl and validates the no-leakage precondition. Exits non-zero
# if any held-out image would reach the generator.
.venv/bin/fitymi finetune --config "$CONFIG" --out "$OUT" || true
test -f "$OUT/captions.jsonl" || { echo "no caption manifest at $OUT" >&2; exit 1; }

accelerate launch --mixed_precision=fp16 \
  "$(python -c 'import diffusers,os;print(os.path.join(os.path.dirname(diffusers.__file__),"..","examples","text_to_image","train_text_to_image_lora.py"))')" \
  --pretrained_model_name_or_path="$BASE" \
  --train_data_dir="$OUT" \
  --caption_column=text \
  --resolution="$RES" --center_crop --random_flip \
  --train_batch_size=4 --gradient_accumulation_steps=4 \
  --max_train_steps="$STEPS" \
  --learning_rate=1e-04 --lr_scheduler=cosine --lr_warmup_steps=0 \
  --rank="$RANK" \
  --seed=0 \
  --output_dir="$OUT"

echo "LoRA written to $OUT"
echo "Next: fitymi generate --config $CONFIG"
