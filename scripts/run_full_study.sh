#!/usr/bin/env bash
# The whole study, in the order the protocol requires. Needs a GPU host.
#
# Note the ordering: every generator, prompt and hyperparameter decision is made
# against VALIDATION, and the test split is unsealed exactly once at the end. Running
# these steps out of order is how a sealed test set stops being sealed.
set -euo pipefail

CLOSED=configs/acne04_closed.yaml
OPEN=configs/acne04_open.yaml
SCRATCH=configs/acne04_scratch.yaml
ADDITIVE=configs/arms/additive_exchange_rate.yaml

echo "== 1. splits (once; every later step reads them) =="
.venv/bin/fitymi prepare --config "$CLOSED"

echo "== 2. closed-set generator =="
CONFIG=$CLOSED OUT=models/closed_set_lora bash scripts/finetune_closed_set.sh

echo "== 3. guidance-scale selection, on validation only =="
for g in 1.5 3.0 5.0 7.5 10.0; do
  echo "-- guidance $g"
  # Each value writes its own pool; pick the winner on validation balanced accuracy.
  sed "s/^  guidance_scale: .*/  guidance_scale: $g/; s|^  pool_dir: .*|  pool_dir: data/synthetic/closed_g$g|" \
    configs/arms/guidance_sweep.yaml > "/tmp/guidance_$g.yaml"
  .venv/bin/fitymi generate --config "/tmp/guidance_$g.yaml"
  .venv/bin/fitymi sweep    --config "/tmp/guidance_$g.yaml"
done
.venv/bin/fitymi analyse --runs runs/guidance_sweep --out results/guidance_sweep
echo "Pick the winning guidance scale from results/guidance_sweep, then set it in"
echo "$CLOSED before continuing. Stopping here on purpose."
