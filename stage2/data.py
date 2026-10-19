"""Stage-1-backed real and synthetic routing instance preparation."""

from __future__ import annotations

from pathlib import Path
from dataclasses import asdict
import hashlib
import json
import logging

import numpy as np
from scipy.stats import gaussian_kde

from stage1.deployment import InfeasibleDeploymentError, Turbine, load_config, load_turbines, run_deployment

from .atsp import WindATSPConfig, WindATSPInstance


LOGGER = logging.getLogger(__name__)
KDE_RETRY_SEED_STRIDE = 2**32


def instance_to_record(instance: WindATSPInstance) -> dict[str, object]:
    """Serialize geometry and the full environment, independent of local paths."""
    return {
        "identifiers": list(instance.identifiers), "coordinates_km": instance.coordinates_km.tolist(),
        "depot_index": instance.depot_index, "node_count": instance.node_count,
        "config": asdict(instance.config),
    }


def instance_from_record(record: dict[str, object]) -> WindATSPInstance:
    """Restore a complete prepared-data/checkpoint record."""
    return WindATSPInstance(
        tuple(record["identifiers"]), np.asarray(record["coordinates_km"], dtype=float),
        int(record["depot_index"]), WindATSPConfig(**record["config"]),
    )


def instance_fingerprint(instance: WindATSPInstance) -> str:
    """Identify the routing problem by geometry and physics, not node labels."""
    record = instance_to_record(instance)
    record.pop("identifiers")
    return hashlib.sha256(json.dumps(record, sort_keys=True, allow_nan=False).encode("utf-8")).hexdigest()


def split_fingerprint(instances: list[WindATSPInstance]) -> str:
    """Hash an ordered split; its order affects seeded batch selection."""
    return hashlib.sha256("\n".join(instance_fingerprint(item) for item in instances).encode()).hexdigest()


def build_deployment_instances(
    turbine_path: Path, deployment_config_path: Path, gust_speed_mps: float = 12.0
) -> list[WindATSPInstance]:
    """Recreate Stage-1 groups in memory and convert them to depot-led ATSP instances."""
    deployment_config = load_config(deployment_config_path)
    turbines = load_turbines(turbine_path)
    result = run_deployment(turbines, deployment_config)
    by_identifier = {turbine.identifier: turbine for turbine in turbines}
    routing_config = WindATSPConfig(
        wind_speed_mps=deployment_config.wind_speed_mps,
        wind_direction_degrees=deployment_config.wind_direction_degrees,
        gust_speed_mps=gust_speed_mps,
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
    field_offset: int = 0,
    gust_speed_mps: float = 12.0,
    max_attempts_per_field: int = 20,
    generation_records: list[dict[str, object]] | None = None,
) -> list[WindATSPInstance]:
    """Generate seeded virtual farms, rerun Stage 1, and return legal ATSP groups.

    The KDE is fitted to Walney-102 coordinates.  Every sampled topology is
    passed back through the same Stage-1 deployment routine used for real
    data; no synthetic grouping is fabricated in Stage 2. Rejected synthetic
    deployments are replaced with seeded draws under the same constraints.
    A field's first seed remains ``seed + field_index``; retry ``k`` adds
    ``k * 2**32``, keeping retries out of the initial train/validation streams.
    """
    if field_count < 1:
        raise ValueError("field_count must be positive.")
    if max_attempts_per_field < 1:
        raise ValueError("max_attempts_per_field must be positive.")
    if seed < 0 or seed + field_count > KDE_RETRY_SEED_STRIDE:
        raise ValueError("Initial KDE seeds must lie in [0, 2**32) to keep retry streams disjoint.")
    deployment_config = load_config(deployment_config_path)
    source_turbines = load_turbines(turbine_path)
    source_points = np.array([(item.x_km, item.y_km) for item in source_turbines], dtype=float)
    routing_config = WindATSPConfig(
        wind_speed_mps=deployment_config.wind_speed_mps,
        wind_direction_degrees=deployment_config.wind_direction_degrees,
        gust_speed_mps=gust_speed_mps,
        max_horizontal_speed_mps=deployment_config.max_horizontal_speed_mps,
        max_endurance_seconds=deployment_config.max_endurance_seconds,
        hover_seconds_per_turbine=deployment_config.hover_seconds_per_turbine,
        hover_to_flight_energy_ratio=deployment_config.hover_to_flight_energy_ratio,
    )
    instances: list[WindATSPInstance] = []
    for field_index in range(field_count):
        field_id = field_index + field_offset
        for attempt in range(max_attempts_per_field):
            sample_seed = seed + field_index + attempt * KDE_RETRY_SEED_STRIDE
            coordinates = sample_kde_topology(source_points, len(source_turbines), sample_seed)
            turbines = [
                Turbine(f"virtual-{field_id:05d}-{index:03d}", float(x), float(y))
                for index, (x, y) in enumerate(coordinates)
            ]
            record: dict[str, object] = {
                "field_id": field_id, "attempt": attempt + 1, "seed": sample_seed,
            }
            try:
                deployment = run_deployment(turbines, deployment_config)
                if not any(len(set(nest.turbine_ids) - {nest.identifier}) > 0 for nest in deployment.nests):
                    raise InfeasibleDeploymentError("Synthetic deployment has no multi-node routing regions.")
            except InfeasibleDeploymentError as exc:
                record.update(accepted=False, reason=str(exc))
                if generation_records is not None:
                    generation_records.append(record)
                LOGGER.warning(
                    "Rejected KDE field=%s seed=%s attempt=%s/%s: %s",
                    field_id, sample_seed, attempt + 1, max_attempts_per_field, exc,
                )
                if attempt + 1 == max_attempts_per_field:
                    raise InfeasibleDeploymentError(
                        f"Unable to generate feasible KDE field {field_id} after {max_attempts_per_field} attempts "
                        f"(initial seed={seed + field_index}). Last rejection: {exc}"
                    ) from exc
                continue

            by_identifier = {item.identifier: item for item in turbines}
            field_instances: list[WindATSPInstance] = []
            for nest in deployment.nests:
                client_ids = tuple(item for item in nest.turbine_ids if item != nest.identifier)
                identifiers = (nest.identifier, *client_ids)
                node_coordinates = np.array([
                    (by_identifier[item].x_km, by_identifier[item].y_km) for item in identifiers
                ])
                if len(identifiers) > 1:
                    field_instances.append(WindATSPInstance(identifiers, node_coordinates, 0, routing_config))
            record.update(accepted=True, region_count=len(field_instances))
            if generation_records is not None:
                generation_records.append(record)
            instances.extend(field_instances)
            if (field_index + 1) % 16 == 0 or field_index + 1 == field_count:
                LOGGER.info("Prepared KDE fields %s/%s (%s routing regions so far).",
                            field_index + 1, field_count, len(instances))
            break
    return instances


def build_training_validation_instances(
    turbine_path: Path,
    deployment_config_path: Path,
    virtual_fields: int,
    seed: int,
    validation_fields: int = 16,
    gust_speed_mps: float = 12.0,
    max_attempts_per_field: int = 20,
    generation_records: dict[str, list[dict[str, object]]] | None = None,
) -> tuple[list[WindATSPInstance], list[WindATSPInstance]]:
    """Build 102-site deployment training regions and disjoint KDE validation fields."""
    if virtual_fields < 1 or validation_fields < 1:
        raise ValueError("Training and validation topology counts must both be positive.")
    train = build_deployment_instances(turbine_path, deployment_config_path, gust_speed_mps)
    train.extend(build_kde_instances(
        turbine_path, deployment_config_path, virtual_fields, seed, field_offset=0,
        gust_speed_mps=gust_speed_mps,
        max_attempts_per_field=max_attempts_per_field,
        generation_records=generation_records.setdefault("training", []) if generation_records is not None else None,
    ))
    validation = build_kde_instances(
        turbine_path,
        deployment_config_path,
        validation_fields,
        seed + 1_000_000,
        field_offset=1_000_000,
        gust_speed_mps=gust_speed_mps,
        max_attempts_per_field=max_attempts_per_field,
        generation_records=generation_records.setdefault("validation", []) if generation_records is not None else None,
    )
    return train, validation


def load_region_instances(
    region_json_path: Path, deployment_config_path: Path, gust_speed_mps: float = 12.0
) -> list[WindATSPInstance]:
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
        gust_speed_mps=gust_speed_mps,
        max_horizontal_speed_mps=deployment_config.max_horizontal_speed_mps,
        max_endurance_seconds=deployment_config.max_endurance_seconds,
        hover_seconds_per_turbine=deployment_config.hover_seconds_per_turbine,
        hover_to_flight_energy_ratio=deployment_config.hover_to_flight_energy_ratio,
    )
    return [
        WindATSPInstance(
            tuple(record["identifiers"]), np.asarray(record["coordinates_km"], dtype=float),
            int(record.get("depot_index", 0)),
            WindATSPConfig(**record["config"]) if "config" in record else routing_config,
        )
        for record in records
    ]
