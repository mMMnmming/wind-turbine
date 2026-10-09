"""Reproducible REINFORCE training with held-out validation selection."""

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
    distance_weight: float = 1.0
    energy_weight: float = 0.5
    completion_bonus: float = 5.0
    safety_penalty: float = 20.0
    safety_overshoot_penalty: float = 10.0


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
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _cartesian_wind_angle(wind_direction_degrees: float, device: torch.device) -> Tensor:
    """Convert paper compass bearing (0=N, 90=E) to Cartesian angle."""
    return torch.deg2rad(torch.tensor(90.0 - wind_direction_degrees, device=device))


def directed_time_matrix(coordinates: Tensor, instance: WindATSPInstance) -> Tensor:
    """Return differentiable directed flight times using the paper's compass convention."""
    delta = coordinates.unsqueeze(1) - coordinates.unsqueeze(2)
    distance_m = torch.linalg.vector_norm(delta, dim=-1) * 1_000.0
    direction = torch.atan2(delta[..., 1], delta[..., 0])
    wind = _cartesian_wind_angle(instance.config.wind_direction_degrees, coordinates.device)
    speed = instance.config.max_horizontal_speed_mps + instance.config.wind_speed_mps * torch.cos(direction - wind)
    return torch.where(distance_m > 0, distance_m / speed, torch.zeros_like(distance_m))


def _pad_instance_batch(
    instances: list[WindATSPInstance], device: torch.device
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Pad a mixed-topology batch and return coordinates, valid nodes, times, hovers, limits."""
    if not instances:
        raise ValueError("Cannot build an empty training batch.")
    max_nodes = max(instance.node_count for instance in instances)
    batch_size = len(instances)
    coordinates = torch.zeros(batch_size, max_nodes, 2, dtype=torch.float32, device=device)
    valid = torch.zeros(batch_size, max_nodes, dtype=torch.bool, device=device)
    times = torch.zeros(batch_size, max_nodes, max_nodes, dtype=torch.float32, device=device)
    for row, instance in enumerate(instances):
        count = instance.node_count
        coordinates[row, :count] = torch.as_tensor(instance.coordinates_km, dtype=torch.float32, device=device)
        valid[row, :count] = True
        times[row, :count, :count] = torch.as_tensor(instance.travel_time_seconds, dtype=torch.float32, device=device)
    hover = torch.tensor([item.config.hover_equivalent_seconds for item in instances], dtype=torch.float32, device=device)
    endurance = torch.tensor([item.config.max_endurance_seconds for item in instances], dtype=torch.float32, device=device)
    return coordinates, valid, times, hover, endurance


def _route_metrics_batch(
    coordinates: Tensor,
    valid_nodes: Tensor,
    time_matrix: Tensor,
    route: Tensor,
    hover: Tensor,
    endurance: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Return flight time, equivalent energy, route distance, and feasibility per row."""
    batch_size, max_nodes, _ = coordinates.shape
    client_counts = valid_nodes.sum(dim=1) - 1
    ordered = torch.cat((torch.zeros(batch_size, 1, dtype=torch.long, device=route.device), route,
                         torch.zeros(batch_size, 1, dtype=torch.long, device=route.device)), dim=1)
    legs = max_nodes
    source = ordered[:, :legs]
    target = ordered[:, 1:legs + 1]
    row = torch.arange(batch_size, device=route.device).unsqueeze(1)
    leg_indices = torch.arange(legs, device=route.device).unsqueeze(0)
    valid_legs = leg_indices <= client_counts.unsqueeze(1)
    flight_legs = time_matrix[row, source, target] * valid_legs
    flight = flight_legs.sum(dim=1)
    distance_legs = torch.linalg.vector_norm(coordinates[row, source] - coordinates[row, target], dim=-1)
    distance = (distance_legs * valid_legs).sum(dim=1)
    energy = flight + client_counts * hover
    feasible = energy <= endurance
    return flight, energy, distance, feasible


def _reward(
    flight: Tensor,
    energy: Tensor,
    feasible: Tensor,
    endurance: Tensor,
    config: TrainingConfig,
) -> Tensor:
    """Normalized distance/energy reward with terminal completion/safety terms."""
    overshoot = ((energy - endurance).clamp_min(0.0) / endurance.clamp_min(1e-6))
    return (
        -config.distance_weight * flight / endurance.clamp_min(1e-6)
        -config.energy_weight * energy / endurance.clamp_min(1e-6)
        +config.completion_bonus * feasible.float()
        -config.safety_penalty * (~feasible).float()
        -config.safety_overshoot_penalty * overshoot
    )


def _batch_features(coordinates: Tensor, valid: Tensor, instances: list[WindATSPInstance]) -> Tensor:
    rows = []
    for index, instance in enumerate(instances):
        count = instance.node_count
        local = make_node_features(
            coordinates[index:index + 1, :count],
            instance.config.wind_speed_mps,
            instance.config.wind_direction_degrees,
            instance.config.gust_speed_mps,
        )
        rows.append(torch.nn.functional.pad(local, (0, 0, 0, coordinates.size(1) - count)))
    return torch.cat(rows, dim=0)


@torch.no_grad()
def evaluate_policy(
    model: GraphPointerNetwork,
    instances: list[WindATSPInstance],
    config: TrainingConfig,
    device: torch.device,
) -> dict[str, float]:
    """Greedy validation; feasibility is primary, flight-time cost secondary."""
    if not instances:
        raise ValueError("Validation requires at least one routing instance.")
    model.eval()
    feasible_count = 0
    flight_costs: list[float] = []
    energy_costs: list[float] = []
    distances: list[float] = []
    rewards: list[float] = []
    for instance in instances:
        coordinates, valid, times, hover, endurance = _pad_instance_batch([instance], device)
        features = _batch_features(coordinates, valid, [instance])
        route, _ = model(features, decode_type="greedy", valid_node_mask=valid)
        flight, energy, distance, feasible = _route_metrics_batch(coordinates, valid, times, route, hover, endurance)
        feasible_value = bool(feasible.item())
        feasible_count += int(feasible_value)
        flight_costs.append(float((flight / endurance).item()))
        energy_costs.append(float((energy / endurance).item()))
        distances.append(float(distance.item()))
        rewards.append(float(_reward(flight, energy, feasible, endurance, config).item()))
    model.train()
    return {
        "feasible_rate": feasible_count / len(instances),
        "mean_normalized_flight_cost": float(np.mean(flight_costs)),
        "mean_normalized_energy": float(np.mean(energy_costs)),
        "mean_distance_km": float(np.mean(distances)),
        "mean_reward": float(np.mean(rewards)),
        "instance_count": float(len(instances)),
    }


def validation_is_better(candidate: dict[str, float], incumbent: dict[str, float] | None) -> bool:
    """Order checkpoints lexicographically: feasible rate, then flight cost."""
    if incumbent is None:
        return True
    return (candidate["feasible_rate"], -candidate["mean_normalized_flight_cost"]) > (
        incumbent["feasible_rate"], -incumbent["mean_normalized_flight_cost"]
    )


def validate_resume_environment(
    saved_environment: dict[str, float] | None,
    current_environment: dict[str, float],
) -> None:
    """Reject silent environment changes when continuing optimizer state."""
    if saved_environment is None:
        raise ValueError("Checkpoint has no recorded wind environment; start a fresh run for reproducibility.")
    if saved_environment != current_environment:
        raise ValueError(
            "Resume wind environment differs from the checkpoint. Keep the same Stage1 wind/endurance settings "
            "and gust speed, or start a new training run."
        )


def train(
    instances: list[WindATSPInstance],
    config: TrainingConfig,
    output: Path,
    validation_instances: list[WindATSPInstance],
    resume_checkpoint: Path | None = None,
) -> Path:
    """Train one policy over diverse regions and select best by held-out validation."""
    if not instances:
        raise ValueError("At least one Stage-1 or synthetic routing instance is required.")
    if not validation_instances:
        raise ValueError("At least one independent validation instance is required.")
    device = require_device(config.device)
    set_seed(config.seed)
    output.mkdir(parents=True, exist_ok=True)
    model = GraphPointerNetwork(config.hidden_dim, config.encoder_layers, config.heads).to(device)
    baseline = GraphPointerNetwork(config.hidden_dim, config.encoder_layers, config.heads).to(device)
    baseline.load_state_dict(model.state_dict())
    baseline.eval()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    history: list[dict[str, float]] = []
    metrics_path = output / "metrics.jsonl"
    if resume_checkpoint is None:
        metrics_path.write_text("", encoding="utf-8")
    start_epoch = 0
    if resume_checkpoint is not None:
        payload = torch.load(resume_checkpoint, map_location=device, weights_only=False)
        environment = asdict(instances[0].config)
        if any(asdict(instance.config) != environment for instance in instances):
            raise ValueError("Training instances must share one wind and endurance configuration.")
        validate_resume_environment(payload.get("wind_environment"), environment)
        model.load_state_dict(payload["model"])
        baseline.load_state_dict(payload["baseline"])
        optimizer.load_state_dict(payload["optimizer"])
        history = list(payload.get("history", []))
        start_epoch = int(payload["epoch"])
    prior_validation = next((item["validation"] for item in reversed(history) if "validation" in item), None)
    best_validation = prior_validation
    best_epoch = int(next((item["epoch"] for item in reversed(history) if item.get("is_best") == 1.0), 0))
    if resume_checkpoint is not None:
        source_best_path = resume_checkpoint.parent / "best.pt"
        if (output / "best.pt").exists():
            source_best_path = output / "best.pt"
        elif source_best_path.exists():
            best_payload = torch.load(source_best_path, map_location=device, weights_only=False)
            torch.save(best_payload, output / "best.pt")
        else:
            best_payload = torch.load(resume_checkpoint, map_location=device, weights_only=False)
            torch.save(best_payload, output / "best.pt")
        best_payload = torch.load(source_best_path, map_location=device, weights_only=False) if source_best_path.exists() else best_payload
        best_validation = best_payload.get("best_validation", best_validation)
        best_epoch = int(best_payload.get("best_epoch", best_epoch))
    generator = np.random.default_rng(config.seed)
    print(
        f"Training start | device={device} | train_instances={len(instances)} | "
        f"validation_instances={len(validation_instances)} | epochs={config.epochs} | "
        f"batches/epoch={config.batches_per_epoch} | batch={config.batch_size}", flush=True,
    )
    for epoch in range(start_epoch, config.epochs):
        epoch_started = time.perf_counter()
        model.train()
        losses: list[float] = []
        train_rewards: list[float] = []
        feasible_rates: list[float] = []
        gradient_norms: list[float] = []
        for _ in range(config.batches_per_epoch):
            batch_size = min(config.batch_size, len(instances))
            selected_indices = generator.choice(len(instances), size=batch_size, replace=False)
            selected_instances = [instances[int(index)] for index in selected_indices]
            coordinates, valid, times, hover, endurance = _pad_instance_batch(selected_instances, device)
            features = _batch_features(coordinates, valid, selected_instances)
            route, log_probability = model(features, decode_type="sample", valid_node_mask=valid)
            flight, energy, _, feasible = _route_metrics_batch(coordinates, valid, times, route, hover, endurance)
            reward = _reward(flight, energy, feasible, endurance, config)
            with torch.no_grad():
                baseline_route, _ = baseline(features, decode_type="greedy", valid_node_mask=valid)
                baseline_flight, baseline_energy, _, baseline_feasible = _route_metrics_batch(
                    coordinates, valid, times, baseline_route, hover, endurance
                )
                baseline_reward = _reward(baseline_flight, baseline_energy, baseline_feasible, endurance, config)
            loss = -((reward - baseline_reward) * log_probability).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_norm)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
            train_rewards.append(float(reward.mean().detach().cpu()))
            feasible_rates.append(float(feasible.float().mean().detach().cpu()))
            gradient_norms.append(float(gradient_norm.detach().cpu()))

        validation = evaluate_policy(model, validation_instances, config, device)
        improved = validation_is_better(validation, best_validation)
        if improved:
            best_validation = validation
            best_epoch = epoch + 1
            baseline.load_state_dict(model.state_dict())
        epoch_metrics: dict[str, object] = {
            "epoch": epoch + 1,
            "mean_train_reward": float(np.mean(train_rewards)),
            "mean_reinforce_loss": float(np.mean(losses)),
            "train_feasible_rate": float(np.mean(feasible_rates)),
            "mean_gradient_norm_before_clip": float(np.mean(gradient_norms)),
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "epoch_seconds": time.perf_counter() - epoch_started,
            "validation": validation,
            "is_best": float(improved),
            "best_epoch": best_epoch,
            "best_validation": best_validation,
        }
        history.append(epoch_metrics)  # type: ignore[arg-type]
        payload = {
            "model": model.state_dict(), "baseline": baseline.state_dict(), "optimizer": optimizer.state_dict(),
            "epoch": epoch + 1, "training_config": asdict(config), "history": history,
            "validation_metrics": validation, "best_validation": best_validation,
            "best_epoch": best_epoch, "selection_rule": "feasible_rate_then_normalized_flight_cost",
            "wind_environment": asdict(instances[0].config),
        }
        torch.save(payload, output / "latest.pt")
        if improved:
            torch.save(payload, output / "best.pt")
        with metrics_path.open("a", encoding="utf-8") as metrics_file:
            metrics_file.write(json.dumps(epoch_metrics) + "\n")
        (output / "training_history.json").write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")
        print(
            f"Epoch {epoch + 1:03d}/{config.epochs} | reward={epoch_metrics['mean_train_reward']:.4f} | "
            f"train feasible={epoch_metrics['train_feasible_rate']:.1%} | "
            f"val feasible={validation['feasible_rate']:.1%} | "
            f"val flight={validation['mean_normalized_flight_cost']:.4f} | "
            f"best epoch={best_epoch} | {epoch_metrics['epoch_seconds']:.1f}s", flush=True,
        )
    (output / "config_snapshot.json").write_text(json.dumps(asdict(config), indent=2) + "\n", encoding="utf-8")
    (output / "training_history.json").write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")
    if not (output / "best.pt").exists():
        raise RuntimeError("Training completed without producing a validation-selected best checkpoint.")
    return output / "best.pt"
