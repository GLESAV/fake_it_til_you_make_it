# Vendored third-party code

## `train_text_to_image_lora.py`

The diffusers reference LoRA trainer, taken verbatim from

    https://github.com/huggingface/diffusers/blob/v0.39.0/examples/text_to_image/train_text_to_image_lora.py

at tag **v0.39.0**, matching the `diffusers` version this project installs. Apache-2.0,
copyright The HuggingFace Inc. team; the licence header is intact in the file.

It is vendored rather than resolved from the installed package because pip does not ship
`examples/`, so `scripts/finetune_closed_set.sh` was pointing at a path that does not
exist on any install. It is vendored rather than reimplemented because a hand-rolled
diffusion trainer is one more thing a reviewer would reasonably decline to trust, and the
contribution of this study is the downstream comparison rather than the fine-tuning
recipe.

**Unmodified.** If it ever needs changing, patch it in a separate commit that says what
changed and why, so the diff against upstream stays readable.

To re-fetch or bump:

    curl -sL -o vendor/train_text_to_image_lora.py \
      https://raw.githubusercontent.com/huggingface/diffusers/v<VERSION>/examples/text_to_image/train_text_to_image_lora.py
