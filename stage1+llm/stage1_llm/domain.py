"""Independent domain model and conservative UAV deployment physics.

This module intentionally duplicates only the published modelling assumptions; it
does not import or execute :mod:`stage1`.
"""
from __future__ import annotations

import csv
import math
from dataclasses import asdict, dataclass
from pathlib import Path


EPSILON = 1e-9


@dataclass(frozen=True, slots=True)
class Turbine:
    identifier: str
    x_km: float
    y_km: float


@dataclass(frozen=True, slots=True)
class DeploymentConfig:
    communication_radius_km: float = 7.0
    max_horizontal_speed_mps: float = 23.0
    wind_speed_mps: float = 6.0
    max_endurance_seconds: float = 2460.0
    hover_seconds_per_turbine: float = 10.0
    hover_to_flight_energy_ratio: float = 1.14
    wind_direction_degrees: float = 90.0
    coordinate_unit: str = "km"

    def __post_init__(self) -> None:
        values = tuple(getattr(self, field) for field in self.__dataclass_fields__ if field != "coordinate_unit")
        if not all(isinstance(value, (int, float)) and math.isfinite(value) for value in values):
            raise ValueError("Deployment parameters must be finite numbers.")
        if self.communication_radius_km <= 0 or self.max_horizontal_speed_mps <= 0:
            raise ValueError("Communication radius and maximum speed must be positive.")
        if self.max_endurance_seconds <= 0 or self.hover_seconds_per_turbine < 0 or self.hover_to_flight_energy_ratio <= 0:
            raise ValueError("Invalid endurance or hover-energy parameters.")
        if self.wind_speed_mps < 0 or self.coordinate_unit != "km" or worst_case_ground_speed_mps(self) <= 0:
            raise ValueError("Coordinates must be km and headwind ground speed must be positive.")


@dataclass(frozen=True, slots=True)
class NestResult:
    identifier: str
    x_km: float
    y_km: float
    turbine_ids: tuple[str, ...]
    route_ids: tuple[str, ...]
    route_distance_km: float
    flight_seconds: float
    equivalent_energy_seconds: float
    service_radius_margin_km: float
    endurance_margin_seconds: float
    proxy_route_ids: tuple[str, ...] = ()
    proxy_flight_seconds: float = 0.0
    proxy_equivalent_energy_seconds: float = 0.0
    task_count: int = 0


@dataclass(frozen=True, slots=True)
class DeploymentResult:
    config: DeploymentConfig
    effective_service_radius_km: float
    assignments: dict[str, str]
    nests: tuple[NestResult, ...]
    all_constraints_satisfied: bool


def load_turbines(path: Path) -> list[Turbine]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.reader(handle, delimiter="\t"))
    if not rows:
        raise ValueError("Turbine input is empty.")
    turbines: list[Turbine] = []
    ids: set[str] = set()
    for line, row in enumerate(rows, 1):
        if len(row) != 3:
            raise ValueError(f"Line {line} must contain exactly three tab-separated fields.")
        identifier = row[0].strip()
        if not identifier or identifier in ids:
            raise ValueError(f"Line {line} has an empty or duplicate turbine ID.")
        try:
            x, y = float(row[1]), float(row[2])
        except ValueError as error:
            raise ValueError(f"Line {line} has non-numeric coordinates.") from error
        if not math.isfinite(x) or not math.isfinite(y):
            raise ValueError(f"Line {line} has non-finite coordinates.")
        ids.add(identifier)
        turbines.append(Turbine(identifier, x, y))
    return turbines


def worst_case_ground_speed_mps(config: DeploymentConfig) -> float:
    return config.max_horizontal_speed_mps - config.wind_speed_mps


def effective_service_radius_km(config: DeploymentConfig) -> float:
    return min(config.communication_radius_km, worst_case_ground_speed_mps(config) * config.max_endurance_seconds / 2000.0)


def distance_km(first: Turbine, second: Turbine) -> float:
    return math.hypot(first.x_km - second.x_km, first.y_km - second.y_km)


def mission_metrics(nest: Turbine, members: tuple[Turbine, ...], config: DeploymentConfig) -> tuple[tuple[Turbine, ...], float, float, float]:
    unvisited, route, current = list(members), [], nest
    while unvisited:
        nxt = min(unvisited, key=lambda item: (distance_km(current, item), item.identifier))
        route.append(nxt)
        unvisited.remove(nxt)
        current = nxt
    distance = 0.0 if not route else distance_km(nest, route[0]) + sum(distance_km(a, b) for a, b in zip(route, route[1:])) + distance_km(route[-1], nest)
    flight = distance * 1000.0 / worst_case_ground_speed_mps(config)
    energy = flight + len(members) * config.hover_seconds_per_turbine * config.hover_to_flight_energy_ratio
    return tuple(route), distance, flight, energy


def result_as_dict(result: DeploymentResult) -> dict:
    return {"all_constraints_satisfied": result.all_constraints_satisfied, "assignments": dict(sorted(result.assignments.items())),
            "config": asdict(result.config), "effective_service_radius_km": result.effective_service_radius_km,
            "nest_count": len(result.nests), "nests": [asdict(item) for item in result.nests]}
