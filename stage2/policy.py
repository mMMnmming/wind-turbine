"""Policy factories and reproducible best-of-N inference (not uniform tours)."""

from __future__ import annotations

import hashlib
import time

import torch
from torch import nn

from .atsp import RouteMetrics, WindATSPInstance
from .data import instance_fingerprint
from .model import GraphPointerNetwork, make_node_features

LEGACY = "legacy"
EDGE_V2 = "edge_attention_v2"
TIMING_SCOPE = ("One graph encoding, greedy plus policy-sampled candidate decoding, host transfer, "
                "original-environment route checks and selection; synchronized on CUDA. Excludes loading, "
                "features/time-matrix construction, random baselines, plotting and file I/O; no warm-up/repeats.")


def build_policy(config: dict) -> nn.Module:
    """Old checkpoints without an architecture field retain their exact layout."""
    architecture = config.get("architecture", LEGACY)
    if architecture == EDGE_V2:
        from .edge_model import EdgeAttentionPolicy
        kind = EdgeAttentionPolicy
    elif architecture == LEGACY:
        kind = GraphPointerNetwork
    else:
        raise ValueError(f"Unknown policy architecture: {architecture}")
    return kind(config["hidden_dim"], config["encoder_layers"], config["heads"])


def candidate_key(metrics: RouteMetrics) -> tuple[bool, float, float]:
    """Feasible first; if none feasible, minimum overshoot then flight time."""
    return (not metrics.feasible, metrics.equivalent_energy_seconds if not metrics.feasible else 0,
            metrics.flight_seconds)


@torch.no_grad()
def infer_instance(model: nn.Module, instance: WindATSPInstance, device: torch.device,
                   strategy: str = "best_of_n", candidates: int = 128, chunk_size: int = 32,
                   seed: int = 1234) -> dict:
    """Reuse an encoding, include greedy, sample the network, never consult optima."""
    if strategy not in ("greedy", "best_of_n") or candidates < 1 or chunk_size < 1 or seed < 0:
        raise ValueError("Invalid decoding strategy, candidate count, chunk size or evaluation seed.")
    count = candidates if strategy == "best_of_n" else 1
    depot = instance.depot_index
    if instance.node_count == 1:
        route = (depot, depot)
        return {"route": route, "metrics": instance.evaluate_route(route), "runtime_seconds": 0.,
                "candidate_count": count, "feasible_candidate_count": count,
                "greedy_flight_seconds": 0., "unique_route_fraction": 1 / count}
    order = [depot, *(i for i in range(instance.node_count) if i != depot)]
    coordinates = torch.as_tensor(instance.coordinates_km[order], dtype=torch.float32, device=device)[None]
    features = make_node_features(coordinates, instance.config.wind_speed_mps,
                                  instance.config.wind_direction_degrees, instance.config.gust_speed_mps)
    times = torch.as_tensor(instance.travel_time_seconds, dtype=torch.float32, device=device)[None]
    times = times[:, order][:, :, order]
    hover = torch.tensor([instance.config.hover_equivalent_seconds], device=device)
    endurance = torch.tensor([instance.config.max_endurance_seconds], device=device)
    digest = hashlib.sha256(f"{seed}:{instance_fingerprint(instance)}".encode()).digest()
    local_seed = int.from_bytes(digest[:8], "little") % (2**63 - 1)
    generator = torch.Generator(device=device).manual_seed(local_seed)
    was_training = model.training
    model.eval()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    try:
        cache = model.encode(features, time_matrix=times, hover=hover, endurance=endurance)
        greedy_order, _ = model.decode(cache, "greedy")
        greedy_route = (depot, *(order[i] for i in greedy_order[0].cpu().tolist()), depot)
        greedy_metrics = instance.evaluate_route(greedy_route)
        best = greedy_metrics
        feasible_count = int(best.feasible)
        unique = {greedy_route}
        remaining = count - 1
        while remaining:
            size = min(chunk_size, remaining)
            sampled, _ = model.decode(cache, "sample", rollouts=size, generator=generator)
            for sampled_order in sampled.cpu().tolist():
                route = (depot, *(order[i] for i in sampled_order), depot)
                measured = instance.evaluate_route(route)
                unique.add(route)
                feasible_count += int(measured.feasible)
                if candidate_key(measured) < candidate_key(best):
                    best = measured
            remaining -= size
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        runtime = time.perf_counter() - started
    finally:
        model.train(was_training)
    return {"route": best.route, "metrics": best, "runtime_seconds": runtime,
            "candidate_count": count, "feasible_candidate_count": feasible_count,
            "greedy_flight_seconds": greedy_metrics.flight_seconds, "local_seed": local_seed,
            "unique_route_fraction": len(unique) / count}
