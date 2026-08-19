#!/usr/bin/env bash
# Final evaluation. Unseals the real test split; every use is appended to the audit
# log that ships with the paper (protocol §3.3).
#
# Run this ONCE, after every generator, prompt and hyperparameter decision is frozen.
set -euo pipefail

for CONFIG in configs/acne04_closed.yaml configs/acne04_open.yaml configs/acne04_scratch.yaml \
              configs/arms/additive_exchange_rate.yaml; do
  echo "== final evaluation: $CONFIG =="
  .venv/bin/fitymi sweep --config "$CONFIG" --final-eval
done

echo "== controls (protocol §8) =="
.venv/bin/fitymi controls --config configs/acne04_closed.yaml
.venv/bin/fitymi controls --config configs/acne04_open.yaml

echo "== analysis =="
.venv/bin/fitymi analyse --runs runs --out results/final

echo
echo "Unseal audit log:"
cat runs/test_unseal_audit.jsonl | wc -l
echo "entries. This number belongs in the paper."
