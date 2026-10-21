"""Pure REINFORCE with independent same-graph rollouts and leave-one-out baseline."""

from __future__ import annotations

from dataclasses import asdict
import json
import math
from pathlib import Path
import time

import numpy as np
import torch
from torch import Tensor

from .atsp import WindATSPInstance
from .data import instance_fingerprint, split_fingerprint
from .policy import EDGE_V2, build_policy, infer_instance
from .pool import TrainingPool, assert_disjoint, sample_training_indices
from .train import (TrainingConfig, _batch_features, _capture_rng_state, _pad_instance_batch, _restore_rng_state,
                    _reward, _route_metrics_batch, _save_checkpoint, require_device, set_seed,
                    validate_resume_environment, validation_is_better)


def leave_one_out_advantage(rewards: Tensor) -> Tensor:
    """Each trajectory is compared only to its other independent same-graph draws."""
    if rewards.ndim != 2 or rewards.size(1) < 2:
        raise ValueError("Leave-one-out requires at least two rollouts per graph.")
    baseline = (rewards.sum(1, keepdim=True) - rewards) / (rewards.size(1) - 1)
    return (rewards - baseline).detach()


def evaluate_edge_policy(model: torch.nn.Module, instances: list[WindATSPInstance], config: TrainingConfig,
                         device: torch.device, references: dict[str, dict] | None = None) -> dict:
    """Fixed, per-instance local RNG; exact references never influence selection."""
    from .evaluation import nearest_neighbor_route
    if not instances:
        raise ValueError("Validation requires at least one independent routing instance.")
    main, greedy, normalized_energy, rewards, nn_improvements, candidates, diversity = [], [], [], [], [], [], []
    predictions, greedy_predictions, buckets = [], [], {}
    for instance in instances:
        selected = infer_instance(model, instance, device, config.decode_strategy,
                                  config.decode_samples, config.decode_chunk_size, config.evaluation_seed)
        auxiliary = infer_instance(model, instance, device, "greedy", 1,
                                   config.decode_chunk_size, config.evaluation_seed)
        result, single = selected["metrics"], auxiliary["metrics"]
        limit = instance.config.max_endurance_seconds
        main.append((float(result.feasible), result.flight_seconds / limit))
        greedy.append((float(single.feasible), single.flight_seconds / limit))
        normalized_energy.append(result.equivalent_energy_seconds / limit)
        reward = _reward(torch.tensor([result.flight_seconds]), torch.tensor([result.equivalent_energy_seconds]),
                         torch.tensor([result.feasible]), torch.tensor([limit]), config)
        rewards.append(reward.item())
        nearest = instance.evaluate_route(nearest_neighbor_route(instance)).flight_seconds
        nn_improvements.append((nearest - result.flight_seconds) / max(nearest, 1e-9))
        candidates.append(selected["feasible_candidate_count"] / selected["candidate_count"])
        diversity.append(selected["unique_route_fraction"])
        predictions.append((instance, result.flight_seconds, result.feasible))
        greedy_predictions.append((instance, single.flight_seconds, single.feasible))
        bucket = (max(instance.node_count - 2, 0)) // 10
        buckets.setdefault(bucket, []).append(main[-1])
    values = np.asarray(main)
    singles = np.asarray(greedy)
    report = {"feasible_rate": float(values[:, 0].mean()),
              "mean_normalized_flight_cost": float(values[:, 1].mean()),
              "mean_normalized_energy": float(np.mean(normalized_energy)), "mean_reward": float(np.mean(rewards)),
              "mean_relative_nn_improvement": float(np.mean(nn_improvements)), "instance_count": len(instances),
              "candidate_feasible_rate": float(np.mean(candidates)),
              "unique_route_fraction": float(np.mean(diversity)),
              "greedy": {"feasible_rate": float(singles[:, 0].mean()),
                         "mean_normalized_flight_cost": float(singles[:, 1].mean())},
              "decode_strategy": config.decode_strategy,
              "candidate_count": config.decode_samples if config.decode_strategy == "best_of_n" else 1}
    for bucket, entries in buckets.items():
        prefix = f"clients_{bucket * 10 + 1}_{bucket * 10 + 10}"
        report[f"{prefix}_count"] = len(entries)
        report[f"{prefix}_feasible_rate"] = float(np.mean([row[0] for row in entries]))
        report[f"{prefix}_normalized_flight_cost"] = float(np.mean([row[1] for row in entries]))
    if references is not None:
        from .benchmark import reference_summary
        report.update(reference_summary(predictions, references))
        report["greedy"].update(reference_summary(greedy_predictions, references))
    return report


def _validate(instances: list[WindATSPInstance], validation: list[WindATSPInstance], config: TrainingConfig,
              pool: TrainingPool | None, references: dict[str, dict] | None) -> None:
    if not instances or not validation:
        raise ValueError("Training and independent validation instances must be nonempty.")
    if any(item.depot_index != 0 for item in instances):
        raise ValueError("Prepared training graphs require depot index zero.")
    assert_disjoint(instances, validation)
    if min(config.batch_size, config.batches_per_epoch, config.epochs, config.lr_schedule_epochs) < 1:
        raise ValueError("Training counts and cosine horizon must be positive.")
    if config.rollouts_per_instance < 2:
        raise ValueError("edge_attention_v2 requires at least two rollouts per graph.")
    if (config.decode_strategy not in ("greedy", "best_of_n") or config.decode_samples < 1
            or config.decode_chunk_size < 1 or config.evaluation_seed < 0):
        raise ValueError("Invalid validation decoding settings.")
    if (not 0 <= config.lr_min <= config.learning_rate or config.learning_rate <= 0
            or config.gradient_clip_norm <= 0):
        raise ValueError("Invalid learning rate or gradient clipping settings.")
    environment = asdict(instances[0].config)
    if any(asdict(item.config) != environment for item in instances):
        raise ValueError("Training instances must share a wind and endurance configuration.")
    if pool is not None and (split_fingerprint(pool.instances) != split_fingerprint(instances)
                             or split_fingerprint(pool.validation_instances) != split_fingerprint(validation)):
        raise ValueError("Pool differs from the initial training or validation split.")
    if references is not None and set(references) - {instance_fingerprint(item) for item in validation}:
        raise ValueError("Reference results do not belong to the validation split.")


def train_edge_policy(instances: list[WindATSPInstance], config: TrainingConfig, output: Path,
                      validation_instances: list[WindATSPInstance], resume_checkpoint: Path | None = None,
                      pool: TrainingPool | None = None, references: dict[str, dict] | None = None) -> Path:
    """No labels/teacher/solver calls; save enough state for exact continuation."""
    _validate(instances, validation_instances, config, pool, references)
    if resume_checkpoint is None and any((output / name).exists() for name in ("best.pt", "latest.pt")):
        raise ValueError("Output contains checkpoints; use a fresh output or resume.")
    initial_fingerprint = split_fingerprint(instances)
    validation_fingerprint = split_fingerprint(validation_instances)
    environment = asdict(instances[0].config)
    payload = None
    if resume_checkpoint is not None:
        payload = torch.load(resume_checkpoint, map_location="cpu", weights_only=False)
        if payload.get("architecture", payload["training_config"].get("architecture", "legacy")) != EDGE_V2:
            raise ValueError("Resume architecture differs from the checkpoint; start a fresh run.")
        validate_resume_environment(payload.get("wind_environment"), environment)
        for key, value in asdict(config).items():
            if key not in ("epochs", "device") and payload["training_config"].get(key) != value:
                raise ValueError(f"Resume training setting changed: {key}.")
        if (payload["initial_training_fingerprint"] != initial_fingerprint
                or payload["validation_fingerprint"] != validation_fingerprint):
            raise ValueError("Resume training/validation split differs from the checkpoint.")
        if (payload.get("training_pool") is None) != (pool is None):
            raise ValueError("Resume requires the same training-pool mode.")
        if pool is not None:
            pool.load_state_dict(payload["training_pool"])
            instances = pool.instances
    device = require_device(config.device)
    set_seed(config.seed)
    output.mkdir(parents=True, exist_ok=True)
    model = build_policy(asdict(config)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    ratio = config.lr_min / config.learning_rate
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda epoch: ratio + (1 - ratio) *
        (1 + math.cos(math.pi * min(epoch, config.lr_schedule_epochs) / config.lr_schedule_epochs)) / 2)
    generator = np.random.default_rng(config.seed)
    start_epoch, history, best_validation, best_epoch = 0, [], None, 0
    if payload is not None:
        model.load_state_dict(payload["model"])
        optimizer.load_state_dict(payload["optimizer"])
        scheduler.load_state_dict(payload["scheduler"])
        history = list(payload["history"])
        start_epoch = int(payload["epoch"])
        best_validation, best_epoch = payload["best_validation"], payload["best_epoch"]
        source = resume_checkpoint.parent / "best.pt"
        # latest is itself the selected best at this epoch. It is authoritative
        # even when an interrupted second write left an older best.pt present.
        recover_from_latest = start_epoch == best_epoch
        if not source.exists() and not recover_from_latest:
            raise ValueError("Keep the selected best.pt beside latest.pt when resuming.")
        selected = payload if recover_from_latest else torch.load(source, map_location="cpu", weights_only=False)
        if selected["epoch"] != best_epoch or selected.get("architecture") != EDGE_V2:
            raise ValueError("best.pt does not match the resumed architecture/best epoch.")
        if source.resolve() != (output / "best.pt").resolve() or recover_from_latest:
            if (output / "best.pt").exists():
                if output.resolve() != resume_checkpoint.parent.resolve():
                    raise ValueError("Resume destination already contains a different best.pt.")
            _save_checkpoint(selected, output / "best.pt")
        _restore_rng_state(payload["rng_state"], generator)
    metrics_path = output / "metrics.jsonl"
    metrics_path.write_text("".join(json.dumps(row) + "\n" for row in history), encoding="utf-8")
    if pool is not None:
        pool.write_snapshot(output / "pool_history")
    print(f"Training {EDGE_V2} | device={device} | graphs/batch={config.batch_size} | "
          f"rollouts/graph={config.rollouts_per_instance} | epochs={config.epochs} | "
          f"validation={config.decode_strategy}:{config.decode_samples}", flush=True)
    for epoch in range(start_epoch, config.epochs):
        started = time.perf_counter()
        if pool is not None and pool.maybe_refresh(epoch):
            instances = pool.instances
            pool.write_snapshot(output / "pool_history")
            print(f"Refreshed pool v{pool.version} before epoch {epoch + 1}: {len(instances)} regions.", flush=True)
        model.train()
        values = []
        graphs_sampled = 0
        learning_rate = optimizer.param_groups[0]["lr"]
        for _ in range(config.batches_per_epoch):
            indices = sample_training_indices(instances, config.batch_size, generator, config.stratified_sampling)
            batch = [instances[int(index)] for index in indices]
            coordinates, valid, times, hover, endurance = _pad_instance_batch(batch, device)
            cache = model.encode(_batch_features(coordinates, valid, batch), valid, time_matrix=times,
                                 hover=hover, endurance=endurance)
            k = config.rollouts_per_instance
            routes, logp, details = model.decode(cache, rollouts=k, return_details=True)
            flight, energy, _, feasible = _route_metrics_batch(
                coordinates.repeat_interleave(k, 0), valid.repeat_interleave(k, 0), times.repeat_interleave(k, 0),
                routes, hover.repeat_interleave(k, 0), endurance.repeat_interleave(k, 0))
            reward = _reward(flight, energy, feasible, endurance.repeat_interleave(k, 0), config).reshape(-1, k)
            advantage = leave_one_out_advantage(reward)
            loss = -(advantage * logp.reshape(-1, k)).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gradient = torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_norm)
            if not torch.isfinite(loss) or not torch.isfinite(gradient):
                raise FloatingPointError("Nonfinite RL loss/gradient; checkpoint was not replaced.")
            optimizer.step()
            graph_orders = routes.detach().cpu().reshape(len(batch), k, -1).tolist()
            diversity = np.mean([len({tuple(order) for order in orders}) / k for orders in graph_orders])
            values.append([reward.mean().item(), loss.item(), feasible.float().mean().item(), gradient.item(),
                           details["entropy"].mean().item(), advantage.std(unbiased=False).item(), float(diversity),
                           float(gradient.item() > config.gradient_clip_norm)])
            graphs_sampled += len(batch)
        validation = evaluate_edge_policy(model, validation_instances, config, device, references)
        improved = validation_is_better(validation, best_validation)
        if improved:
            best_validation, best_epoch = validation, epoch + 1
        averages = np.mean(values, axis=0)
        names = ("mean_train_reward", "mean_reinforce_loss", "train_feasible_rate", "mean_gradient_norm_before_clip",
                 "policy_entropy", "advantage_std", "unique_route_fraction", "gradient_clip_fraction")
        metrics = {**dict(zip(names, map(float, averages))), "epoch": epoch + 1, "learning_rate": learning_rate,
                   "validation": validation, "best_validation": best_validation, "best_epoch": best_epoch,
                   "is_best": float(improved), "epoch_seconds": time.perf_counter() - started,
                   "training_instance_count": len(instances),
                   "sampled_route_count": graphs_sampled * config.rollouts_per_instance,
                   "pool_version": pool.version if pool else 0}
        history.append(metrics)
        scheduler.step()
        state = {"architecture": EDGE_V2, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                 "scheduler": scheduler.state_dict(), "epoch": epoch + 1, "training_config": asdict(config),
                 "history": history, "validation_metrics": validation, "best_validation": best_validation,
                 "best_epoch": best_epoch,
                 "selection_rule": f"{config.decode_strategy}_feasible_rate_then_normalized_flight_cost",
                 "wind_environment": environment, "initial_training_fingerprint": initial_fingerprint,
                 "validation_fingerprint": validation_fingerprint, "training_pool": pool.state_dict() if pool else None,
                 "rng_state": _capture_rng_state(generator), "reference_results": references}
        _save_checkpoint(state, output / "latest.pt")
        if improved:
            _save_checkpoint(state, output / "best.pt")
        with metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(metrics) + "\n")
        (output / "training_history.json").write_text(
            json.dumps(history, indent=2) + "\n", encoding="utf-8")
        print(f"Epoch {epoch + 1:03d}/{config.epochs} | reward={metrics['mean_train_reward']:.4f} | "
              f"train feasible={metrics['train_feasible_rate']:.1%} | "
              f"val feasible={validation['feasible_rate']:.1%} | "
              f"val best flight={validation['mean_normalized_flight_cost']:.4f} | "
              f"val greedy flight={validation['greedy']['mean_normalized_flight_cost']:.4f} | "
              f"entropy={metrics['policy_entropy']:.3f} | best epoch={best_epoch} | "
              f"{metrics['epoch_seconds']:.1f}s", flush=True)
    (output / "config_snapshot.json").write_text(json.dumps({**asdict(config),
        "validation_fingerprint": validation_fingerprint, "initial_training_fingerprint": initial_fingerprint,
        "pool_settings": pool.state_dict()["settings"] if pool else None}, indent=2) + "\n", encoding="utf-8")
    if not (output / "best.pt").exists():
        raise RuntimeError("No validation-selected checkpoint was produced.")
    return output / "best.pt"
