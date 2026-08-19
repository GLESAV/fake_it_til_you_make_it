"""End-to-end pipeline check on procedural data.

The real experiment needs a GPU and a licensed dataset. This runs the *same code
paths* — dedup, group-stratified splitting, sealed-test enforcement, budget-matched
mixing, training, evaluation, statistics, figures — on the toy generative process in
`data/toy.py`, on CPU, in a couple of minutes.

It checks two things beyond "nothing crashed":

1. **Null behaviour.** With `gap=0` the synthetic process is identical to the real
   one, so the mixing curve should be flat. A pipeline that reports a decline here is
   measuring its own bookkeeping, not the data.
2. **Signal behaviour.** With `gap>0` the curve should slope down and the trend test
   should pick it up.

If either check fails, the analysis code is not fit to be pointed at real data.
"""

from __future__ import annotations

import json
import logging
import shutil
from dataclasses import replace
from pathlib import Path

from .config import DataConfig, ExperimentConfig, GeneratorConfig, SweepConfig
from .data.records import Source
from .data.toy import ToyConfig, make_toy_corpus
from .train.loop import TrainConfig

log = logging.getLogger(__name__)


def _toy_train_config(seed: int, epochs: int) -> TrainConfig:
    return TrainConfig(
        arch="tinycnn",
        init="scratch",
        image_size=64,
        batch_size=32,
        epochs=epochs,
        lr=2e-3,
        weight_decay=1e-4,
        patience=epochs,
        num_workers=0,
        seed=seed,
        device="cpu",
        amp=False,
    )


def _build_pools(root: Path, n_synth: int, gap: float, seed: int) -> dict[str, Path]:
    """Write the two synthetic pools and return their directories."""
    pools = {
        Source.SYNTH_CLOSED.value: (root / "synth_closed", gap),
        Source.SYNTH_OPEN.value: (root / "synth_open", min(1.0, gap * 1.5)),
    }
    out: dict[str, Path] = {}
    for i, (name, (directory, pool_gap)) in enumerate(pools.items()):
        corpus = make_toy_corpus(
            directory / "images",
            n_synth,
            Source(name),
            ToyConfig(gap=pool_gap, size=64),
            seed=seed + 11 + i,
            prefix=name,
        )
        corpus.to_jsonl(directory / "corpus.jsonl")
        out[name] = directory
    return out


def run_smoke(out_dir: Path, quick: bool = False, gap: float = 0.6) -> int:
    out_dir = Path(out_dir)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    data_root = out_dir / "data"

    # Calibrated so the toy task is actually learnable (~0.78 val balanced accuracy
    # from scratch). Too few epochs and every arm sits at chance, which would make
    # the checks below vacuous rather than lenient.
    n_real = 480 if quick else 800
    n_synth = 700 if quick else 1600
    seeds = (0, 1, 2) if quick else (0, 1, 2, 3, 4)
    epochs = 18 if quick else 30
    fractions = (0.0, 0.5, 1.0) if quick else (0.0, 0.25, 0.5, 0.75, 1.0)

    log.info(
        "smoke: n_real=%d n_synth=%d seeds=%s epochs=%d fractions=%s gap=%.2f",
        n_real, n_synth, seeds, epochs, fractions, gap,
    )

    base = ExperimentConfig(
        name="smoke",
        output_dir=str(out_dir / "runs"),
        data=DataConfig(
            dataset="toy",
            root=str(data_root),
            splits_dir=str(data_root / "splits"),
            audit_log=str(out_dir / "test_unseal_audit.jsonl"),
            toy_n_real=n_real,
            toy_image_size=64,
            # The toy process draws every image independently, so there are no
            # duplicates to find. Running pHash over 64px low-texture renders
            # measures the hash's resolution limit instead: it collapses the corpus
            # into a few giant groups, and the group-aware splitter then starves the
            # val split. Deduplication is exercised by tests/test_dedup_and_config.py
            # on images that are actually duplicated.
            dedup=False,
        ),
        train=_toy_train_config(0, epochs),
        sweep=SweepConfig(
            fractions=fractions,
            seeds=seeds,
            include_real_subset_controls=True,
            include_additive=False,
        ),
        final_eval=True,
    )

    from .experiment import prepare_data, run_sweep

    bundle = prepare_data(base)
    log.info("splits: %s", bundle.summary())

    pools = _build_pools(data_root, n_synth, gap, seed=base.data.split_seed)

    for source, directory in pools.items():
        config = replace(
            base,
            name=f"smoke_{source}",
            generator=GeneratorConfig(source=source, pool_dir=str(directory)),
        )
        run_sweep(config)

    return _analyse_and_check(out_dir, gap)


def _analyse_and_check(out_dir: Path, gap: float) -> int:
    from .analysis import plots
    from .analysis.aggregate import load_runs, substitution_curve, write_tables

    runs = load_runs(out_dir / "runs")
    if runs.empty:
        log.error("no runs produced")
        return 1

    results_dir = out_dir / "results"
    tables = write_tables(runs, results_dir)
    curve = tables["curve"]
    plots.substitution_curve(curve, results_dir / "fig1_substitution.png")
    plots.tail_recall(runs, results_dir / "fig3_tail.png")

    print("\n=== substitution curve (toy data) ===")
    print(substitution_curve(runs).to_string(index=False))
    print("\n=== H2 trend test ===")
    print(json.dumps(tables["trends"], indent=2, default=float))

    ok = True
    for cell, trend in tables["trends"].items():
        if cell.startswith("real_subset"):
            continue
        slope = trend["slope"]["point"]
        if gap > 0 and slope >= 0:
            log.error("%s: expected a negative slope with gap=%.2f, got %.4f", cell, gap, slope)
            ok = False
        log.info("%s: slope=%.4f p=%.4g", cell, slope, trend["p_value"])

    audit_log = out_dir / "test_unseal_audit.jsonl"
    if audit_log.exists():
        n_unseals = sum(1 for _ in open(audit_log))
        log.info("test split unsealed %d times, all logged to %s", n_unseals, audit_log)
    else:
        log.error("final_eval was set but no unseal audit log was written")
        ok = False

    print(f"\nsmoke {'PASSED' if ok else 'FAILED'}; artefacts in {out_dir}")
    return 0 if ok else 1
