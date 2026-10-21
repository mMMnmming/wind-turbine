"""Freeze saved Stage1 deployments without rerunning deployment or a network."""

from __future__ import annotations

from dataclasses import asdict, fields
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from stage1.deployment import DeploymentConfig, effective_service_radius_km, load_turbines
from stage2.atsp import WindATSPConfig, WindATSPInstance
from stage2.data import instance_fingerprint, instance_to_record, split_fingerprint


def source_hash(path: Path) -> str:
    """Ignore only text-platform newline differences in source provenance."""
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def _close(actual: float, expected: float, label: str) -> None:
    if not math.isfinite(float(actual)) or not math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-9):
        raise ValueError(f"Stage1 {label} disagrees with the paired TSV/configuration.")


def load_stage1_input(summary_path: Path, turbines_path: Path) -> tuple[list[WindATSPInstance], dict]:
    """Use saved memberships/config and the matching TSV's ordinary coordinates."""
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("all_constraints_satisfied") is not True:
        raise ValueError("Stage1 summary must declare a valid completed deployment.")
    try:
        config = DeploymentConfig(**payload["config"])
        nests, assignments = payload["nests"], payload["assignments"]
    except (KeyError, TypeError) as exc:
        raise ValueError("Expected Stage1 native summary with config, nests and assignments.") from exc
    turbines = {item.identifier: item for item in load_turbines(turbines_path)}
    if not isinstance(nests, list) or not nests or not isinstance(assignments, dict):
        raise ValueError("Stage1 nests/assignments must be nonempty saved groups.")
    if set(assignments) != set(turbines):
        raise ValueError("Stage1 assignments and TSV turbine identifiers differ; do not mix 189 and 87 datasets.")
    nest_ids = [nest["identifier"] for nest in nests]
    if len(set(nest_ids)) != len(nest_ids) or payload.get("nest_count") != len(nests):
        raise ValueError("Stage1 nest count/identifiers are inconsistent.")
    if set(assignments.values()) != set(nest_ids) or not set(nest_ids) <= set(turbines):
        raise ValueError("Stage1 assignments refer to missing or unused nests.")
    radius = effective_service_radius_km(config)
    if "effective_service_radius_km" in payload:
        _close(payload["effective_service_radius_km"], radius, "effective service radius")
    routing_fields = {field.name for field in fields(WindATSPConfig)}
    environment = {key: value for key, value in asdict(config).items() if key in routing_fields}
    routing_config = WindATSPConfig(**environment, gust_speed_mps=12.0)
    instances = []
    for nest in nests:
        identifier = nest["identifier"]
        point = turbines[identifier]
        _close(nest["x_km"], point.x_km, f"nest {identifier} x")
        _close(nest["y_km"], point.y_km, f"nest {identifier} y")
        members = nest["turbine_ids"]
        expected = {key for key, value in assignments.items() if value == identifier}
        if (not isinstance(members, list) or len(members) != len(set(members))
                or set(members) != expected or assignments[identifier] != identifier):
            raise ValueError(f"Stage1 saved membership differs from assignments for {identifier}.")
        distances = [math.hypot(turbines[key].x_km - point.x_km, turbines[key].y_km - point.y_km)
                     for key in members]
        if max(distances, default=0.0) > radius + 1e-9:
            raise ValueError(f"Stage1 group {identifier} violates the saved service-radius constraint.")
        # A native summary also retains its conservative Stage1 tour metrics.
        # Check them against the paired coordinates without changing memberships.
        route_ids = nest.get("route_ids")
        if route_ids is not None:
            if len(route_ids) != len(members) or set(route_ids) != expected:
                raise ValueError(f"Stage1 audit route for {identifier} has different members.")
            closed = [identifier, *route_ids, identifier]
            distance = sum(math.hypot(turbines[a].x_km - turbines[b].x_km,
                                     turbines[a].y_km - turbines[b].y_km)
                           for a, b in zip(closed, closed[1:]))
            _close(nest["route_distance_km"], distance, f"route distance for {identifier}")
            flight = distance * 1000 / (config.max_horizontal_speed_mps - config.wind_speed_mps)
            equivalent = flight + len(members) * config.hover_seconds_per_turbine * config.hover_to_flight_energy_ratio
            _close(nest["flight_seconds"], flight, f"conservative flight time for {identifier}")
            _close(nest["equivalent_energy_seconds"], equivalent, f"conservative energy for {identifier}")
            if equivalent > config.max_endurance_seconds + 1e-9:
                raise ValueError(f"Stage1 group {identifier} violates saved endurance.")
        identifiers = (identifier, *(key for key in members if key != identifier))
        coordinates = np.array([(turbines[key].x_km, turbines[key].y_km) for key in identifiers])
        instances.append(WindATSPInstance(identifiers, coordinates, 0, routing_config))
    source = {
        "kind": "stage1_summary", "stage1_summary_sha256": source_hash(summary_path),
        "turbines_sha256": source_hash(turbines_path), "turbine_count": len(turbines),
        "stage1_config": asdict(config), "communication_radius_km": config.communication_radius_km,
        "gust_note": "Fixed 12 m/s design feature, not a simulated gust process.",
        "provenance_note": "Stage1 summaries lack the original TSV hash. These hashes identify the supplied frozen inputs; ID/geometry/audit consistency checks are not proof of historical source provenance.",
    }
    return instances, source


def freeze_regions(path: Path, instances: list[WindATSPInstance], source: dict, *, resume: bool = False) -> dict:
    """Create once, or verify an existing immutable full-physics snapshot."""
    payload = {
        "schema_version": 1, "split": "test", "frozen_reference": True, "source": source,
        "regions": [instance_to_record(item) for item in instances],
        "instance_fingerprints": [instance_fingerprint(item) for item in instances],
        "split_fingerprint": split_fingerprint(instances),
    }
    if path.exists():
        if not resume:
            raise FileExistsError(f"Frozen input already exists: {path}; use --resume or a new output.")
        if json.loads(path.read_text(encoding="utf-8")) != payload:
            raise ValueError("Frozen input differs from source files, ordered regions or environment.")
        return payload
    if resume:
        raise FileNotFoundError("Resume requires the original frozen regions.json; no input was regenerated.")
    path.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation never replaces another run's reference input.
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    return payload
