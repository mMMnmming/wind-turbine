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
