"""Orchestration: prepare data, run an arm, run the sweep.

Kept separate from the CLI so the whole study is callable from a notebook or a
scheduler without going through argument parsing.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict
from pathlib import Path

from .config import ExperimentConfig
from .data.dedup import DedupConfig, assign_groups, merge_groups_by_identity, write_report
from .data.mixing import (
    MixSpec,
    additive_arms,
    build_mixture,
    control_arms,
    substitution_arms,
)
from .data.records import Corpus, Source
from .data.splits import SplitBundle, assert_disjoint, check_split_quality
from .train.loop import RunRecord, TrainConfig, evaluate_corpus, train_model
from .utils.provenance import RunProvenance, hash_obj

log = logging.getLogger(__name__)


def prepare_data(config: ExperimentConfig) -> SplitBundle:
    """Load, deduplicate, split, and persist. Run once; everything else reads it."""
    data = config.data
    splits_dir = Path(data.splits_dir)

    if data.dataset == "toy":
        from .data.toy import ToyConfig, make_toy_corpus

        corpus = make_toy_corpus(
            Path(data.root) / "real",
            data.toy_n_real,
            Source.REAL,
            ToyConfig(gap=0.0, size=data.toy_image_size),
            seed=data.split_seed,
            prefix="real",
        )
    elif data.dataset == "acne04":
        from .data.acne04 import load_acne04

        corpus = load_acne04(data.root, label_files=tuple(data.label_files))
    else:
        raise ValueError(f"unknown dataset {data.dataset!r}")

    if data.dedup:
        dedup_config = DedupConfig(
            phash_max_hamming=data.phash_max_hamming,
            embed_min_cosine=data.embed_min_cosine,
            use_embeddings=data.embedder != "none",
        )
        embedder = None
        from .controls.embedders import cached_embedder

        if data.embedder == "clip":
            from .controls.embedders import clip_embedder

            embedder = cached_embedder(clip_embedder(), data.embed_cache)
        elif data.embedder != "none":
            raise ValueError(f"unknown embedder {data.embedder!r}")
        corpus, report = assign_groups(corpus, dedup_config, embedder=embedder)
        if data.subject_grouping:
            from .controls.embedders import face_embedder

            corpus, subject_report = merge_groups_by_identity(
                corpus,
                cached_embedder(face_embedder(), data.subject_cache),
                data.subject_min_cosine,
            )
            write_report(subject_report, splits_dir / "subject_report.json")
            report["subject_grouping"] = subject_report
        write_report(report, splits_dir / "dedup_report.json")
        log.info(
            "dedup: %d images -> %d groups (%d duplicate clusters, largest %.1f%%)",
            report["n_images"], report["n_groups"], report["n_duplicate_clusters"],
            report["largest_cluster_fraction"] * 100,
        )

    bundle = SplitBundle.from_corpus(
        corpus, fractions=data.fractions, seed=data.split_seed, audit_log=data.audit_log
    )
    assert_disjoint(bundle.train, bundle.val, bundle._test)
    quality = check_split_quality(
        {"train": bundle.train, "val": bundle.val, "test": bundle._test}, data.fractions
    )
    write_report(quality, splits_dir / "split_quality.json")
    bundle.save(splits_dir)
    log.info("splits: %s", bundle.summary())
    return bundle


def load_synthetic_pool(config: ExperimentConfig) -> Corpus:
    pool_dir = Path(config.generator.pool_dir)
    manifest = pool_dir / "corpus.jsonl"
    if not manifest.exists():
        raise FileNotFoundError(
            f"no synthetic corpus at {manifest}. Run `fitymi generate` first, or point "
            "generator.pool_dir at an existing pool."
        )
    pool = Corpus.from_jsonl(manifest)
    expected = Source(config.generator.source)
    wrong = [r for r in pool if r.source is not expected]
    if wrong:
        raise ValueError(
            f"{len(wrong)} images in {pool_dir} are not tagged {expected.value}; "
            "mixing arms must not silently pool different generators"
        )
    return pool


def run_arm(
    spec: MixSpec,
    bundle: SplitBundle,
    pool: Corpus,
    config: ExperimentConfig,
    seed: int,
    overwrite: bool = False,
) -> RunRecord:
    """Train and evaluate one point on the mixing curve.

    Skips a run whose record already exists unless `overwrite`. The full study is
    around 250 runs; restarting from zero after a crash is not an acceptable
    recovery story, and the real-subset controls are identical across generator
    arms so re-running them is pure waste.
    """
    train_config = TrainConfig(**{**asdict(config.train), "seed": seed})
    mixture = build_mixture(bundle.train, pool, MixSpec(**{**asdict(spec), "seed": seed}))
    label = _generator_label(spec, mixture, config)

    run_config = {
        "arm": spec.name,
        "mix_kind": spec.kind,
        "synthetic_fraction": spec.synthetic_fraction,
        "real_budget_fraction": spec.real_budget_fraction,
        "synthetic_multiplier": spec.synthetic_multiplier,
        # An arm holding no synthetic images is the same experiment whichever
        # generator arm requested it. Labelling it "none" makes the run ids collide,
        # so the resume path trains it once instead of once per generator.
        "generator": label,
        # The pool's identity, so two arms drawing from different synthetic pools
        # (say, two guidance scales) never collide even though both are "synth_closed".
        "pool": config.generator.pool_dir if label.startswith("synth") else None,
        "arch": train_config.arch,
        "init": train_config.init,
        "seed": seed,
    }
    # Deliberately excludes config.name: the experiment's identity is its content, not
    # the file it was launched from. That is what lets the shared p=0 arm resolve to
    # one run across the closed-set and open-set sweeps instead of being trained twice.
    run_id = f"{label}_{spec.name}_s{seed}_{hash_obj(run_config)}"
    record_path = Path(config.output_dir) / f"run_{run_id}.json"

    if record_path.exists() and not overwrite:
        existing = json.loads(record_path.read_text())
        # A record written before --final-eval has no test block; that run must be
        # redone rather than silently reused for a final result.
        if not config.final_eval or existing.get("test") is not None:
            log.info("skipping %s: record already exists", run_id)
            return RunRecord(**existing)

    log.info("running %s: %s", run_id, mixture)

    model, history = train_model(mixture, bundle.val, train_config)
    val_result = evaluate_corpus(model, bundle.val, train_config)

    test_result = None
    if config.final_eval:
        test_corpus = bundle.unseal_test(
            reason=f"final evaluation of arm {spec.name} (seed {seed})", run_id=run_id
        )
        test_result = evaluate_corpus(model, test_corpus, train_config).to_dict()

    provenance = RunProvenance(
        run_id=run_id,
        config_hash=hash_obj(config.to_dict()),
        dataset_hash=hash_obj([r.path for r in mixture]),
    )

    record = RunRecord(
        run_id=run_id,
        arm=spec.name,
        config=run_config,
        provenance=provenance.to_dict(),
        train_summary={
            "n": len(mixture),
            "classes": mixture.class_counts(),
            "sources": mixture.source_counts(),
        },
        history=asdict(history),
        val=val_result.to_dict(),
        test=test_result,
    )
    record.save(record_path)
    return record


def _generator_label(spec: MixSpec, mixture: Corpus, config: ExperimentConfig) -> str:
    if spec.kind == "real_subset":
        return "real_subset"
    if len(mixture.synthetic) == 0:
        return "none"
    return config.generator.source


def build_arm_list(config: ExperimentConfig) -> list[MixSpec]:
    arms = list(substitution_arms(config.sweep.fractions))
    if config.sweep.include_real_subset_controls:
        # p=1 would mean "real-only with zero real images", which is not a control.
        # The 100% synthetic arm's comparator is the p=0 arm, not a degenerate one.
        fractions = tuple(f for f in config.sweep.fractions if 0 < f < 1)
        arms += control_arms(fractions)
    if config.sweep.include_additive:
        arms += additive_arms(config.sweep.real_budgets, config.sweep.multipliers)
    return arms


def run_sweep(config: ExperimentConfig) -> list[RunRecord]:
    """The whole experiment for one generator arm."""
    splits_dir = Path(config.data.splits_dir)
    bundle = (
        SplitBundle.load(splits_dir, audit_log=config.data.audit_log)
        if (splits_dir / "train.jsonl").exists()
        else prepare_data(config)
    )
    records: list[RunRecord] = []
    arms = build_arm_list(config)

    # A sweep whose arms are all real-only -- the p=0 baseline, or the real-subset
    # controls on their own -- needs no synthetic pool, and demanding one blocks the
    # single most useful run to do first: establishing where the real arm lands before
    # a generator exists. Load the pool only if some arm will actually draw from it.
    needs_pool = any(spec.synthetic_fraction > 0 or spec.synthetic_multiplier > 0 for spec in arms)
    pool = load_synthetic_pool(config) if needs_pool else Corpus([])
    if not needs_pool:
        log.info("no arm requests synthetic images; skipping the synthetic pool")

    total = len(arms) * len(config.sweep.seeds)
    log.info("sweep %s: %d arms x %d seeds = %d runs", config.name, len(arms), len(config.sweep.seeds), total)

    for spec in arms:
        for seed in config.sweep.seeds:
            records.append(run_arm(spec, bundle, pool, config, seed))
    return records
