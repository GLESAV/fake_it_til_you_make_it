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

echo "== 0. data audit (no GPU; run it before trusting any split) =="
.venv/bin/python scripts/audit_acne04.py
.venv/bin/python scripts/audit_acne04_subjects.py --splits data/splits_subject

echo "== 1. splits (once; every later step reads them) =="
# Protocol 8.8: subject-disjoint, not merely deduplicated. ACNE04 is ~600 people, and an
# image-level split puts 77% of the test set's faces in training too.
.venv/bin/fitymi prepare --config "$CLOSED"
python - <<'CHECK'
import json, sys
from pathlib import Path
report = Path("data/splits/subject_report.json")
if not report.exists():
    sys.exit("no subject_report.json: data.subject_grouping is off. Protocol 8.8 requires it.")
r = json.loads(report.read_text())
print(f"  {r['n_images']} images -> {r['n_subjects']} subjects, largest {r['largest_subject']}")
CHECK

echo "== 2. closed-set generator =="
CONFIG=$CLOSED OUT=models/closed_set_lora bash scripts/finetune_closed_set.sh

echo "== 3. guidance-scale selection, on validation only =="
# Grid centred on 2.0, not on the 7.5 aesthetic default: Sariyildiz et al. Table 1 and
# Fan et al. independently put the training-utility optimum there, and the old grid put
# four of five points on the wrong side of it. See configs/arms/guidance_sweep.yaml.
for g in 1.0 1.5 2.0 3.0 5.0; do
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
