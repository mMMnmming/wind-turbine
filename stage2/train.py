"""GPU-first REINFORCE trainer with checkpointing and greedy rollout baseline."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch import Tensor

from .atsp import WindATSPInstance
from .model import GraphPointerNetwork, make_node_features


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    device: str = "cuda"
    seed: int = 1234
    hidden_dim: int = 128
    encoder_layers: int = 3
    heads: int = 8
    batch_size: int = 512
    epochs: int = 100
    batches_per_epoch: int = 100
    learning_rate: float = 1e-4
    gradient_clip_norm: float = 1.0


def require_device(requested: str) -> torch.device:
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable. Use --device cpu only for debugging.")
    return torch.device(requested)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def directed_time_matrix(coordinates: Tensor, instance: WindATSPInstance) -> Tensor:
    """Return the differentiable, direction-dependent flight-time matrix."""
    delta = coordinates.unsqueeze(1) - coordinates.unsqueeze(2)
    distance_m = torch.linalg.vector_norm(delta, dim=-1) * 1_000.0
    direction = torch.atan2(delta[..., 1], delta[..., 0])
    wind = torch.deg2rad(torch.tensor(instance.config.wind_direction_degrees, device=coordinates.device))
    speed = instance.config.max_horizontal_speed_mps + instance.config.wind_speed_mps * torch.cos(direction - wind)
    return torch.where(distance_m > 0, distance_m / speed, torch.zeros_like(distance_m))


def _directed_cost(coordinates: Tensor, route: Tensor, instance: WindATSPInstance) -> Tensor:
    ordered = torch.cat((torch.zeros_like(route[:, :1]), route, torch.zeros_like(route[:, :1])), dim=1)
    time_matrix = directed_time_matrix(coordinates, instance)
    flight = time_matrix.gather(1, ordered[:, :-1].unsqueeze(-1).expand(-1, -1, time_matrix.size(-1))).gather(
        2, ordered[:, 1:].unsqueeze(-1)
    ).squeeze(-1).sum(dim=1)
    return flight + (coordinates.size(1) - 1) * instance.config.hover_equivalent_seconds


def _sample_batch(instance: WindATSPInstance, batch_size: int, device: torch.device) -> Tensor:
    base = torch.tensor(instance.coordinates_km, dtype=torch.float32, device=device)
    # KDE-generated fields supply topology variation. Do not perturb a
    # validated Stage-1 group into an infeasible routing instance.
    return base.unsqueeze(0).expand(batch_size, -1, -1).clone()


def train(
    instances: list[WindATSPInstance],
    config: TrainingConfig,
    output: Path,
    resume_checkpoint: Path | None = None,
) -> Path:
    """Train one shared policy across Stage-1 local regions and save resumable state."""
    if not instances:
        raise ValueError("At least one Stage-1 routing instance is required.")
    device = require_device(config.device)
    set_seed(config.seed)
    output.mkdir(parents=True, exist_ok=True)
    model = GraphPointerNetwork(config.hidden_dim, config.encoder_layers, config.heads).to(device)
    baseline = GraphPointerNetwork(config.hidden_dim, config.encoder_layers, config.heads).to(device)
    baseline.load_state_dict(model.state_dict())
    baseline.eval()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    history: list[dict[str, float]] = []
    metrics_path = output / 'metrics.jsonl'
    if resume_checkpoint is None:
        metrics_path.write_text('', encoding='utf-8')
    start_epoch = 0
    if resume_checkpoint is not None:
        payload = torch.load(resume_checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(payload["model"])
        baseline.load_state_dict(payload["baseline"])
        optimizer.load_state_dict(payload["optimizer"])
        history = list(payload.get("history", []))
        start_epoch = int(payload["epoch"])
    best_cost = min((item["mean_equivalent_seconds"] for item in history), default=float("inf"))
    print(
        f"Training start | device={device} | regions={len(instances)} | "
        f"epochs={config.epochs} | batches/epoch={config.batches_per_epoch} | batch={config.batch_size}",
        flush=True,
    )
    for epoch in range(start_epoch, config.epochs):
        epoch_started = time.perf_counter()
        model.train()
        costs: list[float] = []
        baseline_costs: list[float] = []
        losses: list[float] = []
        advantages: list[float] = []
        gradient_norms: list[float] = []
        feasible_rates: list[float] = []
        for step in range(config.batches_per_epoch):
            instance = instances[(epoch * config.batches_per_epoch + step) % len(instances)]
            coordinates = _sample_batch(instance, config.batch_size, device)
            features = make_node_features(coordinates, instance.config.wind_speed_mps, instance.config.wind_direction_degrees)
            time_matrix = directed_time_matrix(coordinates, instance)
            route, log_probability = model(
                features, decode_type="sample", travel_time_seconds=time_matrix,
                hover_equivalent_seconds=instance.config.hover_equivalent_seconds,
                max_endurance_seconds=instance.config.max_endurance_seconds,
            )
            cost = _directed_cost(coordinates, route, instance)
            with torch.no_grad():
                baseline_route, _ = baseline(
                    features, decode_type="greedy", travel_time_seconds=time_matrix,
                    hover_equivalent_seconds=instance.config.hover_equivalent_seconds,
                    max_endurance_seconds=instance.config.max_endurance_seconds,
                )
                baseline_cost = _directed_cost(coordinates, baseline_route, instance)
            loss = ((cost - baseline_cost) * log_probability).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_norm)
            optimizer.step()
            costs.append(float(cost.mean().detach().cpu()))
            baseline_costs.append(float(baseline_cost.mean().detach().cpu()))
            losses.append(float(loss.detach().cpu()))
            advantages.append(float((cost - baseline_cost).mean().detach().cpu()))
            gradient_norms.append(float(gradient_norm.detach().cpu()))
            feasible_rates.append(float((cost <= instance.config.max_endurance_seconds).float().mean().detach().cpu()))
        mean_cost = float(np.mean(costs))
        improved = mean_cost <= best_cost
        if improved:
            baseline.load_state_dict(model.state_dict())
            best_cost = mean_cost
        epoch_metrics = {
            "epoch": float(epoch + 1),
            "mean_equivalent_seconds": mean_cost,
            "mean_baseline_equivalent_seconds": float(np.mean(baseline_costs)),
            "mean_advantage_seconds": float(np.mean(advantages)),
            "reinforce_loss": float(np.mean(losses)),
            "feasible_route_rate": float(np.mean(feasible_rates)),
            "gradient_norm_before_clip": float(np.mean(gradient_norms)),
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "epoch_seconds": time.perf_counter() - epoch_started,
            "best_equivalent_seconds": best_cost,
        }
        history.append(epoch_metrics)
        payload = {
            "model": model.state_dict(), "baseline": baseline.state_dict(), "optimizer": optimizer.state_dict(),
            "epoch": epoch + 1, "training_config": asdict(config), "history": history,
        }
        torch.save(payload, output / "latest.pt")
        if improved:
            torch.save(payload, output / "best.pt")
        with metrics_path.open("a", encoding="utf-8") as metrics_file:
            metrics_file.write(json.dumps(epoch_metrics) + "\n")
        (output / "training_history.json").write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")
        print(
            f"Epoch {epoch + 1:03d}/{config.epochs} | cost={mean_cost:.2f}s | "
            f"baseline={epoch_metrics['mean_baseline_equivalent_seconds']:.2f}s | "
            f"advantage={epoch_metrics['mean_advantage_seconds']:.2f}s | "
            f"loss={epoch_metrics['reinforce_loss']:.4f} | "
            f"feasible={epoch_metrics['feasible_route_rate']:.1%} | "
            f"grad={epoch_metrics['gradient_norm_before_clip']:.3f} | "
            f"best={best_cost:.2f}s | {epoch_metrics['epoch_seconds']:.1f}s",
            flush=True,
        )
    (output / "config_snapshot.json").write_text(json.dumps(asdict(config), indent=2) + "\n", encoding="utf-8")
    (output / "training_history.json").write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")
    return output / "latest.pt"
