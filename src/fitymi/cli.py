"""Command line interface.

    fitymi prepare   --config configs/acne04_closed.yaml
    fitymi generate  --config configs/acne04_closed.yaml
    fitymi sweep     --config configs/acne04_closed.yaml
    fitymi controls  --config configs/acne04_closed.yaml
    fitymi analyse   --runs runs/acne04_closed --out results/acne04_closed
    fitymi smoke     --out runs/smoke

`--final-eval` is the only way to touch the sealed test split, and every use is
appended to the unseal audit log (protocol §3.3).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from .config import ExperimentConfig


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def _load(args) -> ExperimentConfig:
    config = ExperimentConfig.load(args.config)
    if getattr(args, "final_eval", False):
        config.final_eval = True
    if getattr(args, "output_dir", None):
        config.output_dir = args.output_dir
    return config


def cmd_prepare(args) -> int:
    from .experiment import prepare_data

    config = _load(args)
    bundle = prepare_data(config)
    print(json.dumps(bundle.summary(), indent=2))
    return 0


def cmd_generate(args) -> int:
    from .data.splits import SplitBundle
    from .generate.sample import SampleConfig, generate

    config = _load(args)
    bundle = SplitBundle.load(config.data.splits_dir, audit_log=config.data.audit_log)
    sample_config = SampleConfig(
        model_path=config.generator.model_path,
        lora_path=config.generator.lora_path,
        regime=config.generator.regime,
        guidance_scale=config.generator.guidance_scale,
        steps=config.generator.steps,
        batch_size=config.generator.batch_size,
        seed=config.train.seed,
        source=config.generator.source,
    )
    pool = generate(
        reference=bundle.train,
        total=config.generator.n_images,
        output_dir=config.generator.pool_dir,
        config=sample_config,
    )
    print(f"generated {len(pool)} images into {config.generator.pool_dir}")
    return 0


def cmd_finetune(args) -> int:
    from .data.splits import SplitBundle
    from .generate.finetune import FinetuneConfig, finetune

    config = _load(args)
    bundle = SplitBundle.load(config.data.splits_dir, audit_log=config.data.audit_log)
    out = finetune(
        train=bundle.train,
        output_dir=args.out,
        config=FinetuneConfig(base_model=config.generator.model_path),
        held_out=(bundle.val,),
    )
    print(f"generator written to {out}")
    return 0


def cmd_sweep(args) -> int:
    from .experiment import run_sweep

    config = _load(args)
    records = run_sweep(config)
    print(f"completed {len(records)} runs; records in {config.output_dir}")
    return 0


def cmd_controls(args) -> int:
    from .controls.discriminability import run_probe
    from .controls.memorization import audit, pixel_embedder
    from .data.splits import SplitBundle
    from .experiment import load_synthetic_pool

    config = _load(args)
    bundle = SplitBundle.load(config.data.splits_dir, audit_log=config.data.audit_log)
    pool = load_synthetic_pool(config)

    out_dir = Path(config.output_dir) / "controls"
    out_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, dict] = {}

    probe = run_probe(bundle.train, pool, config.train)
    results["discriminability"] = probe.to_dict()
    print(f"real-vs-synthetic AUC = {probe.auc:.4f} -- {probe.interpretation}")

    memo = audit(pool, bundle.train, pixel_embedder())
    results["memorization"] = memo.to_dict()
    print(f"replication rate = {memo.replication_rate:.4f} -- {memo.note}")
    if not memo.release_recommended:
        print("NOTE: this synthetic pool should not be released as-is.", file=sys.stderr)

    with open(out_dir / "controls.json", "w") as fh:
        json.dump(results, fh, indent=2, default=float)
    return 0


def cmd_calibrate(args) -> int:
    """Report what threshold this corpus supports, next to what the literature says.

    Runs before any audit that groups or flags by similarity, because every default we
    inherited turned out to be wrong here -- see docs/05_acne04_audit.md §2 and §7.
    """
    import json

    from .controls.calibration import calibrate
    from .data.acne04 import load_acne04

    corpus = load_acne04(args.root)
    paths = [r.path for r in corpus]

    if args.embedder == "sscd":
        from .controls.embedders import SSCD_THRESHOLD, sscd_embedder

        embedder, reference = sscd_embedder(device=args.device), SSCD_THRESHOLD
        mode = "per_item"
    elif args.embedder == "clip":
        from .controls.embedders import clip_embedder

        embedder, reference = clip_embedder(device=args.device), 0.95
        mode = "per_item"
    elif args.embedder == "face":
        from .controls.embedders import face_embedder

        # Identity is the per-pair case: 84% of ACNE04 images genuinely belong to a
        # multi-image subject, so a per-item rate measures the true positive rather than
        # the error. See controls/calibration.py.
        embedder, reference = face_embedder(device=args.device), 0.60
        mode = "per_pair"
    else:
        raise ValueError(f"unknown embedder {args.embedder!r}")

    result = calibrate(paths, embedder, target_fpr=args.fpr, mode=mode,
                       reference_threshold=reference)

    print(f"\n== {args.embedder} on {len(paths)} images ==")
    n = result.null
    print(f"null (unrelated pairs): median {n.median:.3f}  p99 {n.p99:.3f}  max {n.maximum:.4f}")
    print(f"threshold for a {100 * args.fpr:.1f}% {mode.replace('_', '-')} "
          f"false-positive rate: {result.threshold:.4f}")
    if result.recommended_threshold is not None:
        note = "" if result.recommended_threshold <= result.threshold + 1e-9 else \
            f"  (raised: the error-rate threshold merges more than " \
            f"{100 * result.max_component_fraction:.0f}% of the corpus)"
        print(f"recommended, also keeping the largest cluster under "
              f"{100 * result.max_component_fraction:.0f}%: "
              f"{result.recommended_threshold:.4f}{note}")
    if result.reference_threshold is not None:
        print(f"the usual threshold ({result.reference_threshold:.2f}) gives "
              f"{100 * result.fpr_at_reference:.2f}% here"
              + ("" if result.reference_is_usable else "  <-- does not transfer"))
    print("\npercolation (largest connected component):")
    for t, largest, frac in result.percolation:
        print(f"  {t:>6.3f}  {largest:>6}  {100 * frac:>6.1f}%")

    if args.json:
        Path(args.json).write_text(json.dumps(result.to_dict(), indent=2))
        print(f"\nwrote {args.json}")
    return 0


def cmd_analyse(args) -> int:
    from .analysis import plots
    from .analysis.aggregate import load_runs, substitution_curve, write_tables

    runs = load_runs(args.runs)
    if runs.empty:
        print(f"no run records under {args.runs}", file=sys.stderr)
        return 1

    out_dir = Path(args.out)
    tables = write_tables(runs, out_dir)
    curve = tables["curve"]

    plots.substitution_curve(curve, out_dir / "fig1_substitution.png")
    plots.tail_recall(runs, out_dir / "fig3_tail.png")
    if "kind" in runs and (runs["kind"] == "additive").any():
        plots.additive_curves(runs, out_dir / "fig2_additive.png")

    print(substitution_curve(runs).to_string(index=False))
    print(f"\ntables and figures written to {out_dir}")
    return 0


def cmd_smoke(args) -> int:
    from .smoke import run_smoke

    return run_smoke(Path(args.out), quick=args.quick, gap=args.gap)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="fitymi", description=__doc__)
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    def with_config(p):
        p.add_argument("--config", required=True, help="experiment YAML")
        p.add_argument("--output-dir", default=None)
        return p

    with_config(sub.add_parser("prepare", help="load, dedup, split")).set_defaults(func=cmd_prepare)
    with_config(sub.add_parser("generate", help="generate a synthetic pool")).set_defaults(
        func=cmd_generate
    )

    ft = with_config(sub.add_parser("finetune", help="fine-tune the closed-set generator"))
    ft.add_argument("--out", required=True)
    ft.set_defaults(func=cmd_finetune)

    sweep = with_config(sub.add_parser("sweep", help="run the mixing sweep"))
    sweep.add_argument(
        "--final-eval",
        action="store_true",
        help="unseal the real test split for a final evaluation; every use is logged",
    )
    sweep.set_defaults(func=cmd_sweep)

    with_config(sub.add_parser("controls", help="run the §8 controls")).set_defaults(
        func=cmd_controls
    )

    cal = sub.add_parser("calibrate",
                         help="what similarity threshold does this corpus actually support?")
    cal.add_argument("--root", default="data/acne04")
    cal.add_argument("--embedder", choices=("sscd", "clip", "face"), default="sscd")
    cal.add_argument("--fpr", type=float, default=0.01,
                     help="target per-item false-positive rate (default 1%%)")
    cal.add_argument("--device", default=None)
    cal.add_argument("--json", default=None)
    cal.set_defaults(func=cmd_calibrate)

    analyse = sub.add_parser("analyse", help="aggregate runs into tables and figures")
    analyse.add_argument("--runs", required=True)
    analyse.add_argument("--out", required=True)
    analyse.set_defaults(func=cmd_analyse)

    smoke = sub.add_parser("smoke", help="end-to-end pipeline check on procedural data")
    smoke.add_argument("--out", default="runs/smoke")
    smoke.add_argument("--quick", action="store_true", help="fewer images, seeds and epochs")
    smoke.add_argument(
        "--gap",
        type=float,
        default=0.6,
        help="mis-specification of the toy synthetic process. 0 runs the null check: "
        "the two processes are then identical and the mixing curve must be flat.",
    )
    smoke.set_defaults(func=cmd_smoke)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    _setup_logging(args.verbose)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
