# Environment constraints observed in the authoring sandbox

Recorded 2026-08-19 so that nobody re-discovers these the hard way.

## Hardware
- 4 vCPU, 15 GB RAM, ~30 GB writable disk.
- **No GPU** (`nvidia-smi` absent).

Consequence: diffusion fine-tuning and generation cannot run here. The codebase is
written to run on a GPU host; a CPU `--smoke` path exists purely to exercise the
pipeline end-to-end on a procedurally generated toy dataset.

## Network egress

The session routes HTTPS through a policy-enforcing proxy. Probed hosts:

| Host | Reachable |
|---|---|
| `pypi.org`, `files.pythonhosted.org` | ✅ (direct, in `noProxy`) |
| `api.github.com`, `raw.githubusercontent.com` | ✅ |
| `storage.googleapis.com` | ✅ |
| `huggingface.co` | ❌ blocked |
| `arxiv.org`, `openreview.net`, `nature.com`, `link.springer.com`, `sciencedirect.com`, `ncbi.nlm.nih.gov`, `zenodo.org`, `isic-archive.com`, `paperswithcode.com`, `semanticscholar.org` | ❌ blocked |

Consequences:

1. **No model weights.** HuggingFace is blocked, so no Stable Diffusion, no CLIP, no
   pretrained backbones can be fetched here.
2. **No datasets.** ACNE04 and every alternative live on blocked hosts.
3. **No full-text papers.** The literature review in `01_related_work.md` is built from
   search-engine summaries only, which is why every claim in it carries a
   [V]/[S]/[?] verification tag. Promoting them to [V] requires an unrestricted network.

## What this means for the plan

The work splits cleanly:

- **Here (done / doable):** literature review, protocol, codebase, unit tests, smoke-test
  runs on synthetic toy data, analysis and plotting code, paper scaffold.
- **On a host with an accelerator and open network (not doable here):** dataset
  acquisition, generator fine-tuning, image generation, the classifier runs, the real
  numbers. See `docs/04_running_locally.md` for doing this on Apple Silicon, where
  generation throughput rather than training is the binding constraint.

The repo is therefore structured so that the second half is `make` targets driven by
committed YAML configs, not ad-hoc notebooks.
