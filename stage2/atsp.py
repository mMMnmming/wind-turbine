"""Constraint-aware, wind-dependent asymmetric TSP primitives."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


class RouteValidationError(ValueError):
    """Raised when a route is not a single feasible depot-to-depot tour."""


@dataclass(frozen=True, slots=True)
class WindATSPConfig:
    wind_speed_mps: float = 6.0
    wind_direction_degrees: float = 90.0
    max_horizontal_speed_mps: float = 23.0
    max_endurance_seconds: float = 2460.0
    hover_seconds_per_turbine: float = 10.0
    hover_to_flight_energy_ratio: float = 1.14

    @property
    def hover_equivalent_seconds(self) -> float:
        return self.hover_seconds_per_turbine * self.hover_to_flight_energy_ratio


@dataclass(frozen=True, slots=True)
class RouteMetrics:
    route: tuple[int, ...]
    visited_identifiers: tuple[str, ...]
    flight_seconds: float
    equivalent_energy_seconds: float
    feasible: bool


@dataclass(frozen=True, slots=True)
class WindATSPInstance:
    identifiers: tuple[str, ...]
    coordinates_km: np.ndarray
    depot_index: int
    config: WindATSPConfig

    def __post_init__(self) -> None:
        coordinates = np.asarray(self.coordinates_km, dtype=float)
        if coordinates.ndim != 2 or coordinates.shape[1] != 2:
            raise ValueError("coordinates_km must have shape (node_count, 2).")
        if len(self.identifiers) != len(coordinates) or len(set(self.identifiers)) != len(self.identifiers):
            raise ValueError("Every coordinate requires one unique identifier.")
        if not 0 <= self.depot_index < len(coordinates):
            raise ValueError("depot_index is outside the coordinate array.")
        if not np.isfinite(coordinates).all():
            raise ValueError("Coordinates must be finite.")
        object.__setattr__(self, "coordinates_km", coordinates)

    @property
    def node_count(self) -> int:
        return len(self.identifiers)

    @property
    def travel_time_seconds(self) -> np.ndarray:
        delta = self.coordinates_km[None, :, :] - self.coordinates_km[:, None, :]
        distance_m = np.linalg.norm(delta, axis=-1) * 1_000.0
        direction = np.arctan2(delta[..., 1], delta[..., 0])
        wind_angle = math.radians(self.config.wind_direction_degrees)
        ground_speed = self.config.max_horizontal_speed_mps + self.config.wind_speed_mps * np.cos(
            direction - wind_angle
        )
        if np.any(ground_speed <= 0):
            raise ValueError("Wind makes at least one directed edge non-flyable.")
        times = np.divide(distance_m, ground_speed, out=np.zeros_like(distance_m), where=distance_m > 0)
        return times

    def safe_action_mask(self, current_index: int, visited: set[int], elapsed_seconds: float) -> np.ndarray:
        times = self.travel_time_seconds
        mask = np.zeros(self.node_count, dtype=bool)
        for candidate in range(self.node_count):
            if candidate == self.depot_index or candidate in visited:
                continue
            projected = (
                elapsed_seconds
                + times[current_index, candidate]
                + self.config.hover_equivalent_seconds
                + times[candidate, self.depot_index]
            )
            mask[candidate] = projected <= self.config.max_endurance_seconds
        return mask

    def evaluate_route(self, route: tuple[int, ...]) -> RouteMetrics:
        if len(route) != self.node_count + 1 or route[0] != self.depot_index or route[-1] != self.depot_index:
            raise RouteValidationError("Route must start and end at the depot and include every node once.")
        clients = route[1:-1]
        expected = set(range(self.node_count)) - {self.depot_index}
        if set(clients) != expected or len(set(clients)) != len(clients):
            raise RouteValidationError("Route must visit each non-depot turbine exactly once.")
        times = self.travel_time_seconds
        flight_seconds = float(sum(times[first, second] for first, second in zip(route, route[1:])))
        equivalent = flight_seconds + len(clients) * self.config.hover_equivalent_seconds
        return RouteMetrics(
            route=route,
            visited_identifiers=tuple(self.identifiers[index] for index in clients),
            flight_seconds=flight_seconds,
            equivalent_energy_seconds=equivalent,
            feasible=equivalent <= self.config.max_endurance_seconds,
        )
