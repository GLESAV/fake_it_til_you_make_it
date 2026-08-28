# Study pipeline. Targets that need a GPU say so.
PY      ?= .venv/bin/python
FITYMI  ?= .venv/bin/fitymi
CONFIG  ?= configs/acne04_closed.yaml
RUNS    ?= runs/acne04
RESULTS ?= results/acne04

.PHONY: help venv install test smoke lint prepare finetune generate sweep final controls analyse paper publication-gate clean

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

venv:  ## create the virtualenv
	python3 -m venv .venv && .venv/bin/pip install -U pip

install: venv  ## install the package (add [generate] on the GPU host)
	.venv/bin/pip install -e '.[dev]'

install-gpu:  ## install with the generation extras (needs HuggingFace access)
	.venv/bin/pip install -e '.[dev,generate]'

test:  ## unit tests, CPU, seconds
	$(PY) -m pytest tests/ -q

smoke:  ## end-to-end pipeline check on procedural data, CPU, ~20 min
	$(FITYMI) smoke --out runs/smoke --quick

smoke-full:  ## the same with 5 seeds and 5 mixing fractions, CPU, ~1 h
	$(FITYMI) smoke --out runs/smoke_full

smoke-null:  ## null check: gap=0 makes the processes identical, curve must be flat
	$(FITYMI) smoke --out runs/smoke_null --quick --gap 0

# ---------------------------------------------------------------- real study
# Everything below needs a GPU host and the ACNE04 dataset. See
# docs/03_environment.md for why it cannot run in the authoring sandbox.

prepare:  ## load ACNE04, deduplicate, split, seal the test set
	$(FITYMI) prepare --config $(CONFIG)

finetune:  ## fine-tune the closed-set generator on the real train split (GPU)
	$(FITYMI) finetune --config $(CONFIG) --out models/closed_set_lora

generate:  ## sample the synthetic pool (GPU)
	$(FITYMI) generate --config $(CONFIG)

sweep:  ## the mixing sweep, evaluated on validation only
	$(FITYMI) sweep --config $(CONFIG)

final:  ## FINAL evaluation. Unseals the real test split; every use is logged.
	@echo "This unseals the sealed test set (protocol §3.3). Ctrl-C within 5s to abort."
	@sleep 5
	$(FITYMI) sweep --config $(CONFIG) --final-eval

controls:  ## discriminability probe and memorisation audit (protocol §8)
	$(FITYMI) controls --config $(CONFIG)

analyse:  ## tables and figures from run records
	$(FITYMI) analyse --runs $(RUNS) --out $(RESULTS)

paper:  ## build the manuscript
	cd paper && latexmk -pdf main.tex

publication-gate:  ## pre-publication checks (docs/08_publication_practices.md §6)
	$(PY) scripts/publication_gate.py

clean:
	rm -rf runs/smoke runs/smoke_full .pytest_cache
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
