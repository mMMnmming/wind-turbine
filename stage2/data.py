"""Stage-1-backed real and synthetic routing instance preparation."""

from __future__ import annotations

from pathlib import Path
import json

import numpy as np
from scipy.stats import gaussian_kde

from stage1.deployment import Turbine, load_config, load_turbines, run_deployment

from .atsp import WindATSPConfig, WindATSPInstance


def build_deployment_instances(turbine_path: Path, deployment_config_path: Path) -> list[WindATSPInstance]:
    """Recreate Stage-1 groups in memory and convert them to depot-led ATSP instances."""
    deployment_config = load_config(deployment_config_path)
    turbines = load_turbines(turbine_path)
    result = run_deployment(turbines, deployment_config)
    by_identifier = {turbine.identifier: turbine for turbine in turbines}
    routing_config = WindATSPConfig(
        wind_speed_mps=deployment_config.wind_speed_mps,
        wind_direction_degrees=deployment_config.wind_direction_degrees,
        max_horizontal_speed_mps=deployment_config.max_horizontal_speed_mps,
        max_endurance_seconds=deployment_config.max_endurance_seconds,
        hover_seconds_per_turbine=deployment_config.hover_seconds_per_turbine,
        hover_to_flight_energy_ratio=deployment_config.hover_to_flight_energy_ratio,
    )
    instances: list[WindATSPInstance] = []
    for nest in result.nests:
        client_ids = tuple(identifier for identifier in nest.turbine_ids if identifier != nest.identifier)
        identifiers = (nest.identifier, *client_ids)
        coordinates = np.array(
            [(by_identifier[identifier].x_km, by_identifier[identifier].y_km) for identifier in identifiers],
            dtype=float,
        )
        instances.append(WindATSPInstance(identifiers, coordinates, 0, routing_config))
    return instances


def sample_kde_topology(points: np.ndarray, sample_size: int, seed: int) -> np.ndarray:
    """Draw a deterministic two-dimensional Gaussian-KDE topology."""
    coordinates = np.asarray(points, dtype=float)
    if coordinates.ndim != 2 or coordinates.shape[1] != 2 or len(coordinates) < 3:
        raise ValueError("KDE requires at least three two-dimensional points.")
    generator = np.random.default_rng(seed)
    return gaussian_kde(coordinates.T).resample(sample_size, seed=generator).T


def build_kde_instances(
    turbine_path: Path,
    deployment_config_path: Path,
    field_count: int,
    seed: int,
) -> list[WindATSPInstance]:
    """Generate seeded virtual farms, rerun Stage 1, and return legal ATSP groups.

    The KDE is fitted to Walney-102 coordinates.  Every sampled topology is
    passed back through the same Stage-1 deployment routine used for real
    data; no synthetic grouping is fabricated in Stage 2.
    """
    if field_count < 1:
        raise ValueError("field_count must be positive.")
    deployment_config = load_config(deployment_config_path)
    source_turbines = load_turbines(turbine_path)
    source_points = np.array([(item.x_km, item.y_km) for item in source_turbines], dtype=float)
    routing_config = WindATSPConfig(
        wind_speed_mps=deployment_config.wind_speed_mps,
        wind_direction_degrees=deployment_config.wind_direction_degrees,
        max_horizontal_speed_mps=deployment_config.max_horizontal_speed_mps,
        max_endurance_seconds=deployment_config.max_endurance_seconds,
        hover_seconds_per_turbine=deployment_config.hover_seconds_per_turbine,
        hover_to_flight_energy_ratio=deployment_config.hover_to_flight_energy_ratio,
    )
    instances: list[WindATSPInstance] = []
    for field_index in range(field_count):
        coordinates = sample_kde_topology(source_points, len(source_turbines), seed + field_index)
        turbines = [Turbine(f"virtual-{field_index:03d}-{index:03d}", float(x), float(y)) for index, (x, y) in enumerate(coordinates)]
        deployment = run_deployment(turbines, deployment_config)
        by_identifier = {item.identifier: item for item in turbines}
        for nest in deployment.nests:
            client_ids = tuple(item for item in nest.turbine_ids if item != nest.identifier)
            identifiers = (nest.identifier, *client_ids)
            node_coordinates = np.array([(by_identifier[item].x_km, by_identifier[item].y_km) for item in identifiers])
            instance = WindATSPInstance(identifiers, node_coordinates, 0, routing_config)
            # Stage 1 validates its nearest-neighbour closed route.  Retain a
            # second explicit validation point for future alternate generators.
            if instance.node_count > 1:
                instances.append(instance)
    return instances


def load_region_instances(region_json_path: Path, deployment_config_path: Path) -> list[WindATSPInstance]:
    """Load externally supplied Stage-1 region JSON for evaluation.

    Accepted records are the auditable ``training_regions.json`` shape:
    ``identifiers``, ``coordinates_km`` and an optional ``depot_index``.
    """
    payload = json.loads(region_json_path.read_text(encoding="utf-8"))
    records = payload["regions"] if isinstance(payload, dict) and "regions" in payload else payload
    if not isinstance(records, list):
        raise ValueError("Region JSON must be a list or an object with a 'regions' list.")
    deployment_config = load_config(deployment_config_path)
    routing_config = WindATSPConfig(
        wind_speed_mps=deployment_config.wind_speed_mps,
        wind_direction_degrees=deployment_config.wind_direction_degrees,
        max_horizontal_speed_mps=deployment_config.max_horizontal_speed_mps,
        max_endurance_seconds=deployment_config.max_endurance_seconds,
        hover_seconds_per_turbine=deployment_config.hover_seconds_per_turbine,
        hover_to_flight_energy_ratio=deployment_config.hover_to_flight_energy_ratio,
    )
    return [
        WindATSPInstance(
            tuple(record["identifiers"]), np.asarray(record["coordinates_km"], dtype=float),
            int(record.get("depot_index", 0)), routing_config,
        )
        for record in records
    ]
