"""Paired, baseline-protected fine-tuning; no teacher, critic or test-set selection."""

from __future__ import annotations

from dataclasses import asdict
import argparse
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Callable

import numpy as np
import torch
from torch import Tensor

from .atsp import WindATSPInstance
from .data import split_fingerprint
from .policy import EDGE_V2, build_policy
from .pool import TrainingPool, assert_disjoint, sample_training_indices
from .train import (
    TrainingConfig,
    _batch_features,
    _capture_rng_state,
    _pad_instance_batch,
    _restore_rng_state,
    _reward,
    _route_metrics_batch,
    _save_checkpoint,
    require_device,
    set_seed,
    validate_resume_environment,
    validation_is_better,
)
from .train_v2 import _validate, evaluate_edge_policy, leave_one_out_advantage


def write_json(path: Path, value: object) -> None:
    """Atomically replace a run-owned report, never a source checkpoint."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def entropy_coefficient(epoch: int, start: float, end: float, horizon: int) -> float:
    """Zero-based schedule: first update uses start, final scheduled epoch uses end."""
    if (
        not all(math.isfinite(x) and x >= 0 for x in (start, end))
        or horizon < 1
        or epoch < 0
    ):
        raise ValueError("Invalid entropy coefficient/schedule.")
    fraction = min(epoch / max(horizon - 1, 1), 1.0)
    return start + (end - start) * fraction


def reinforce_entropy_loss(
    advantage: Tensor, logp: Tensor, entropy: Tensor, coefficient: float
) -> tuple[Tensor, Tensor, Tensor]:
    """The decoder supplies differentiable per-trajectory valid-step mean entropy."""
    policy = -(advantage.detach() * logp).mean()
    bonus = coefficient * entropy.mean()
    return policy - bonus, policy, bonus


def _model_copy(model: torch.nn.Module) -> dict[str, Tensor]:
    return {
        key: value.detach().cpu().clone() for key, value in model.state_dict().items()
    }


def _score(validation: dict, kind: str) -> dict:
    return validation if kind == "sampling" else validation["greedy"]


def _materialize_best(state: dict, output: Path) -> None:
    # latest.pt contains both incumbents. A crash between any of these writes is
    # repaired from that one authoritative transaction on resume.
    for kind, best in state["best_models"].items():
        payload = {
            key: value
            for key, value in state.items()
            if key
            not in {"best_models", "optimizer", "scheduler", "rng_state", "history"}
        }
        payload.update(
            model=best["model"],
            epoch=best["epoch"],
            best_epoch=best["epoch"],
            validation_metrics=best["validation"],
            training_pool=best["training_pool"],
            selection_rule=f"{kind}_feasible_rate_then_normalized_flight_cost",
        )
        _save_checkpoint(payload, output / f"best_{kind}.pt")
        if kind == "sampling":
            _save_checkpoint(payload, output / "best.pt")


def _run_arm(
    parent: dict,
    config: TrainingConfig,
    output: Path,
    instances: list[WindATSPInstance],
    validation: list[WindATSPInstance],
    pool: TrainingPool | None,
    identity: dict,
    arm: str,
    start: float,
    end: float,
    resume: bool,
    references: dict | None,
) -> dict:
    if pool is not None:
        pool.load_state_dict(parent["training_pool"])
        # A new experiment has a local refresh clock, but continues the original
        # next-field stream. Both arms start with identical geometry/counters.
        pool.last_refresh_epoch = 0
        instances = pool.instances
    _validate(instances, validation, config, pool, references)
    device = require_device(config.device)
    set_seed(config.seed)
    model = build_policy(asdict(config)).to(device)
    model.load_state_dict(parent["model"])
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    ratio = config.lr_min / config.learning_rate
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda epoch: ratio
        + (1 - ratio)
        * (
            1
            + math.cos(
                math.pi
                * min(epoch, config.lr_schedule_epochs)
                / config.lr_schedule_epochs
            )
        )
        / 2,
    )
    generator = np.random.default_rng(config.seed)
    latest = output / "latest.pt"
    output.mkdir(parents=True, exist_ok=True)
    if latest.exists():
        if not resume:
            raise ValueError("Arm exists; use --resume.")
        state = torch.load(latest, map_location="cpu", weights_only=False)
        if state.get("experiment_identity") != identity or state.get("arm") != arm:
            raise ValueError("Arm checkpoint identity differs.")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        if pool is not None:
            pool.load_state_dict(state["training_pool"])
            instances = pool.instances
        _restore_rng_state(state["rng_state"], generator)
        history, best_models, first_epoch = (
            state["history"],
            state["best_models"],
            state["epoch"],
        )
    else:
        baseline = evaluate_edge_policy(model, validation, config, device, references)
        snapshot = {
            "model": _model_copy(model),
            "epoch": 0,
            "validation": baseline,
            "training_pool": pool.state_dict() if pool else None,
        }
        best_models = {"sampling": snapshot, "greedy": snapshot}
        history, first_epoch = [], 0
        state = {
            "architecture": EDGE_V2,
            "training_config": asdict(config),
            "wind_environment": asdict(instances[0].config),
            "initial_training_fingerprint": identity["initial_training_fingerprint"],
            "validation_fingerprint": identity["validation_fingerprint"],
            "experiment_identity": identity,
            "arm": arm,
            "source_checkpoint_sha256": identity["source_sha256"],
            "epoch": 0,
            "model": snapshot["model"],
            "validation_metrics": baseline,
            "training_pool": snapshot["training_pool"],
            "best_models": best_models,
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "rng_state": _capture_rng_state(generator),
            "history": history,
        }
        _save_checkpoint(state, output / "epoch0.pt")
        _save_checkpoint(state, latest)
    _materialize_best(state, output)
    if pool is not None:
        pool.write_snapshot(output / "pool_history")
    write_json(
        output / "config_snapshot.json",
        {
            **asdict(config),
            "entropy_start": start,
            "entropy_end": end,
            "entropy_decay_epochs": config.lr_schedule_epochs,
            "identity": identity,
        },
    )
    print(
        f"Fine-tune {arm}: epoch={first_epoch}/{config.epochs}; source epoch={parent['epoch']}",
        flush=True,
    )
    for epoch in range(first_epoch, config.epochs):
        started = time.perf_counter()
        if pool is not None and pool.maybe_refresh(epoch):
            instances = pool.instances
            pool.write_snapshot(output / "pool_history")
            print(
                f"{arm}: refreshed pool v{pool.version} before epoch {epoch + 1}",
                flush=True,
            )
        beta = entropy_coefficient(epoch, start, end, config.lr_schedule_epochs)
        values = []
        sampled = 0
        model.train()
        lr = optimizer.param_groups[0]["lr"]
        for _ in range(config.batches_per_epoch):
            indices = sample_training_indices(
                instances, config.batch_size, generator, config.stratified_sampling
            )
            batch = [instances[int(i)] for i in indices]
            coords, valid, times, hover, endurance = _pad_instance_batch(batch, device)
            cache = model.encode(
                _batch_features(coords, valid, batch),
                valid,
                time_matrix=times,
                hover=hover,
                endurance=endurance,
            )
            k = config.rollouts_per_instance
            routes, logp, details = model.decode(cache, rollouts=k, return_details=True)
            flight, energy, _, feasible = _route_metrics_batch(
                coords.repeat_interleave(k, 0),
                valid.repeat_interleave(k, 0),
                times.repeat_interleave(k, 0),
                routes,
                hover.repeat_interleave(k, 0),
                endurance.repeat_interleave(k, 0),
            )
            rewards = _reward(
                flight, energy, feasible, endurance.repeat_interleave(k, 0), config
            ).reshape(-1, k)
            advantage = leave_one_out_advantage(rewards)
            loss, policy, bonus = reinforce_entropy_loss(
                advantage, logp.reshape(-1, k), details["entropy"], beta
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gradient = torch.nn.utils.clip_grad_norm_(
                model.parameters(), config.gradient_clip_norm
            )
            if not torch.isfinite(loss) or not torch.isfinite(gradient):
                raise FloatingPointError(
                    "Nonfinite fine-tuning loss/gradient; latest was not replaced."
                )
            optimizer.step()
            orders = routes.detach().cpu().reshape(len(batch), k, -1).tolist()
            diversity = np.mean(
                [len({tuple(order) for order in graph}) / k for graph in orders]
            )
            values.append(
                [
                    rewards.mean().item(),
                    loss.item(),
                    policy.item(),
                    bonus.item(),
                    details["entropy"].mean().item(),
                    advantage.std(unbiased=False).item(),
                    gradient.item(),
                    float(gradient.item() > config.gradient_clip_norm),
                    float(diversity),
                    feasible.float().mean().item(),
                ]
            )
            sampled += len(batch) * k
        report = evaluate_edge_policy(model, validation, config, device, references)
        for kind in best_models:
            if validation_is_better(
                _score(report, kind), _score(best_models[kind]["validation"], kind)
            ):
                best_models[kind] = {
                    "model": _model_copy(model),
                    "epoch": epoch + 1,
                    "validation": report,
                    "training_pool": pool.state_dict() if pool else None,
                }
        names = (
            "mean_train_reward",
            "mean_total_loss",
            "mean_reinforce_loss",
            "entropy_bonus",
            "policy_entropy",
            "advantage_std",
            "mean_gradient_norm_before_clip",
            "gradient_clip_fraction",
            "unique_route_fraction",
            "train_feasible_rate",
        )
        metrics = {
            **dict(zip(names, map(float, np.mean(values, axis=0)))),
            "epoch": epoch + 1,
            "entropy_coefficient": beta,
            "learning_rate": lr,
            "validation": report,
            "best_sampling_epoch": best_models["sampling"]["epoch"],
            "best_greedy_epoch": best_models["greedy"]["epoch"],
            "epoch_seconds": time.perf_counter() - started,
            "sampled_route_count": sampled,
            "training_instance_count": len(instances),
            "pool_version": pool.version if pool else 0,
        }
        history.append(metrics)
        scheduler.step()
        state.update(
            model=_model_copy(model),
            epoch=epoch + 1,
            validation_metrics=report,
            best_models=best_models,
            optimizer=optimizer.state_dict(),
            scheduler=scheduler.state_dict(),
            training_pool=pool.state_dict() if pool else None,
            rng_state=_capture_rng_state(generator),
            history=history,
        )
        _save_checkpoint(state, latest)
        _materialize_best(state, output)
        write_json(output / "training_history.json", history)
        (output / "metrics.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in history), encoding="utf-8"
        )
        print(
            f"{arm} Epoch {epoch + 1:03d}/{config.epochs} | beta={beta:.6f} | "
            f"reward={metrics['mean_train_reward']:.4f} | feasible={report['feasible_rate']:.1%} | "
            f"val best={report['mean_normalized_flight_cost']:.6f} | "
            f"greedy={report['greedy']['mean_normalized_flight_cost']:.6f} | "
            f"best sampling/greedy epochs={metrics['best_sampling_epoch']}/{metrics['best_greedy_epoch']} | "
            f"{metrics['epoch_seconds']:.1f}s",
            flush=True,
        )
    # Repair reports even if the process stopped after the final latest.pt write.
    write_json(output / "training_history.json", history)
    (output / "metrics.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in history), encoding="utf-8"
    )
    summary = {
        "baseline": torch.load(
            output / "epoch0.pt", map_location="cpu", weights_only=False
        )["validation_metrics"],
        "sampling": {
            "epoch": best_models["sampling"]["epoch"],
            "validation": best_models["sampling"]["validation"],
        },
        "greedy": {
            "epoch": best_models["greedy"]["epoch"],
            "validation": best_models["greedy"]["validation"],
        },
    }
    write_json(output / "summary.json", summary)
    return summary


def run_comparison(
    checkpoint: Path,
    config: TrainingConfig,
    output: Path,
    instances: list[WindATSPInstance],
    validation: list[WindATSPInstance],
    *,
    pool_factory: Callable[[], TrainingPool] | None = None,
    entropy_start: float = 0.01,
    entropy_end: float = 0.0,
    resume: bool = False,
    references: dict | None = None,
) -> dict:
    """Same weights/data/seeds, fresh optimizer; old checkpoint participates in selection."""
    if not instances or not validation:
        raise ValueError(
            "Fine-tuning requires nonempty training and validation splits."
        )
    entropy_coefficient(0, entropy_start, entropy_end, config.lr_schedule_epochs)
    if output.resolve() in checkpoint.resolve().parents:
        raise ValueError("Fine-tuning output must not contain the source checkpoint.")
    parent = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if parent.get("architecture") != EDGE_V2 or config.architecture != EDGE_V2:
        raise ValueError("Fine-tuning requires edge_attention_v2 architecture.")
    validate_resume_environment(
        parent.get("wind_environment"), asdict(instances[0].config)
    )
    if parent["initial_training_fingerprint"] != split_fingerprint(instances) or parent[
        "validation_fingerprint"
    ] != split_fingerprint(validation):
        raise ValueError(
            "Fine-tuning training/validation fingerprints differ from source."
        )
    mutable = {
        "device",
        "seed",
        "epochs",
        "batches_per_epoch",
        "batch_size",
        "learning_rate",
        "lr_min",
        "lr_schedule_epochs",
    }
    for key, value in asdict(config).items():
        if key not in mutable and parent["training_config"].get(key) != value:
            raise ValueError(f"Fine-tuning changed protected setting: {key}.")
    if (parent.get("training_pool") is None) != (pool_factory is None):
        raise ValueError("Fine-tuning must preserve the source training-pool mode.")
    assert_disjoint(instances, validation)
    identity = {
        "schema_version": 1,
        "source_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "training_config": asdict(config),
        "entropy_start": entropy_start,
        "entropy_end": entropy_end,
        "initial_training_fingerprint": split_fingerprint(instances),
        "validation_fingerprint": split_fingerprint(validation),
        "references_sha256": hashlib.sha256(
            json.dumps(references, sort_keys=True).encode()
        ).hexdigest(),
    }
    manifest = output / "experiment.json"
    if manifest.exists():
        if not resume or json.loads(manifest.read_text(encoding="utf-8")) != identity:
            raise ValueError(
                "Experiment exists or identity/settings differ; use a fresh output or matching --resume."
            )
    else:
        if resume or any(
            (output / name).exists()
            for name in ("control", "entropy", "comparison.json", "selected.pt")
        ):
            raise ValueError(
                "Missing experiment identity or orphan output; use a fresh output."
            )
        # Validate both fresh pools before committing a new experiment manifest.
        if pool_factory is not None:
            pool_factory().load_state_dict(parent["training_pool"])
        _validate(
            instances,
            validation,
            config,
            pool_factory() if pool_factory else None,
            references,
        )
        output.mkdir(parents=True, exist_ok=True)
        write_json(manifest, identity)
    results = {}
    for arm, start, end in (
        ("control", 0.0, 0.0),
        ("entropy", entropy_start, entropy_end),
    ):
        results[arm] = _run_arm(
            parent,
            config,
            output / arm,
            instances,
            validation,
            pool_factory() if pool_factory else None,
            identity,
            arm,
            start,
            end,
            resume,
            references,
        )
    if results["control"]["baseline"] != results["entropy"]["baseline"]:
        raise ValueError("Paired arms have different baseline validation metrics.")
    baseline = results["control"]["baseline"]
    report = {
        "baseline": baseline,
        "arms": results,
        "identity": identity,
        "test_selection_note": "Validation-only selection. Test/87 is not loaded.",
    }
    for kind in ("sampling", "greedy"):
        selected, best = "baseline", _score(baseline, kind)
        for arm in ("control", "entropy"):
            candidate = _score(results[arm][kind]["validation"], kind)
            if validation_is_better(candidate, best):
                selected, best = arm, candidate
        path = output / (
            "control/epoch0.pt"
            if selected == "baseline"
            else f"{selected}/best_{kind}.pt"
        )
        payload = torch.load(path, map_location="cpu", weights_only=False)
        _save_checkpoint(payload, output / f"selected_{kind}.pt")
        if kind == "sampling":
            _save_checkpoint(payload, output / "selected.pt")
        report[f"selected_{kind}"] = selected
        print(
            f"Selected {kind}: {selected}; feasible={best['feasible_rate']:.1%}; "
            f"normalized flight={best['mean_normalized_flight_cost']:.8f}",
            flush=True,
        )
    write_json(output / "comparison.json", report)
    lines = [
        "# 验证集微调对照",
        "",
        "|模型|解码|最佳轮次|可行率|归一化飞行时间|有效区域加权gap|有效/总区域|",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for arm in ("baseline", "control", "entropy"):
        for kind in ("sampling", "greedy"):
            v = baseline if arm == "baseline" else results[arm][kind]["validation"]
            score = _score(v, kind)
            epoch = 0 if arm == "baseline" else results[arm][kind]["epoch"]
            gap = score.get("weighted_optimality_gap")
            gap_text = "N/A" if gap is None else f"{gap:.4%}"
            count = int(score.get("optimality_gap_count", 0))
            lines.append(
                f"|{arm}|{kind}|{epoch}|{score['feasible_rate']:.2%}|{score['mean_normalized_flight_cost']:.8f}|{gap_text}|{count}/{len(validation)}|"
            )
    lines.extend(
        [
            "",
            "gap仅统计可行且已证明最优的正成本区域；参考缺失或未证明时N/A，不冒充全部区域最优。",
            "仅依据验证集选模；单次微调结果不代表统计显著提升。",
        ]
    )
    (output / "comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def comparison_from_cli(
    args: argparse.Namespace, parser: argparse.ArgumentParser
) -> int:
    """Read existing prepared data only; inherit architecture/reward/physics from source."""
    from dataclasses import fields, replace
    from .benchmark import load_references
    from .run import _prepare_or_load_dataset

    if args.checkpoint is None or args.data_dir is None:
        parser.error(
            "finetune_compare requires --checkpoint and the original --data-dir."
        )
    if args.regions_json is not None:
        parser.error(
            "Fine-tuning uses the original validation split, not --regions-json/test data."
        )
    if not (args.data_dir / "data_manifest.json").exists():
        parser.error(
            "Fine-tuning requires existing prepared data; no dataset was generated."
        )
    parent = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if parent.get("architecture") != EDGE_V2 or parent.get("training_pool") is None:
        parser.error(
            "CLI fine-tuning requires an edge_attention_v2 checkpoint with its training pool."
        )
    protected = (
        "architecture",
        "energy_weight",
        "gust_speed_mps",
        "virtual_fields",
        "validation_fields",
        "data_seed",
        "refresh_every_epochs",
        "refresh_fields",
        "stratified_sampling",
        "rollouts_per_instance",
        "decode_strategy",
        "decode_samples",
        "decode_chunk_size",
        "evaluation_seed",
    )
    if any(getattr(args, key) is not None for key in protected):
        parser.error(
            "Fine-tuning inherits data/reward/decoding settings from checkpoint; do not override them."
        )
    settings = parent["training_pool"]["settings"]
    manifest = json.loads(
        (args.data_dir / "data_manifest.json").read_text(encoding="utf-8")
    )
    train, validation, _ = _prepare_or_load_dataset(
        args.data_dir,
        args.input,
        args.stage1_config,
        settings["seed"],
        settings["capacity"],
        manifest["validation_kde_fields"],
        parent["wind_environment"]["gust_speed_mps"],
        settings["max_attempts_per_field"],
    )
    allowed = {field.name for field in fields(TrainingConfig)}
    config = TrainingConfig(
        **{
            key: value
            for key, value in parent["training_config"].items()
            if key in allowed
        }
    )
    epochs = args.epochs if args.epochs is not None else 50
    config = replace(
        config,
        epochs=epochs,
        seed=args.seed if args.seed is not None else 1236,
        learning_rate=args.learning_rate if args.learning_rate is not None else 3e-5,
        lr_min=args.lr_min if args.lr_min is not None else 3e-6,
        lr_schedule_epochs=(
            args.lr_schedule_epochs if args.lr_schedule_epochs is not None else epochs
        ),
    )
    overrides = {
        key: getattr(args, key)
        for key in ("device", "batch_size", "batches_per_epoch")
        if getattr(args, key) is not None
    }
    config = replace(config, **overrides)

    def make_pool() -> TrainingPool:
        return TrainingPool(
            train,
            validation,
            args.input,
            args.stage1_config,
            seed=settings["seed"],
            refresh_every_epochs=settings["refresh_every_epochs"],
            refresh_fields=settings["refresh_fields"],
            max_attempts_per_field=settings["max_attempts_per_field"],
        )

    references = (
        load_references(args.reference_results, validation)
        if args.reference_results
        else None
    )
    run_comparison(
        args.checkpoint,
        config,
        args.output,
        train,
        validation,
        pool_factory=make_pool,
        entropy_start=args.entropy_start,
        entropy_end=args.entropy_end,
        resume=args.resume,
        references=references,
    )
    print(f"Validation-selected model: {args.output / 'selected.pt'}", flush=True)
    return 0
