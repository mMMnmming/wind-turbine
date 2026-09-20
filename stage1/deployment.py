"""Deterministic, constraint-aware UAV nest deployment for Walney-189."""

from __future__ import annotations

import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


_EPSILON = 1e-9


class InputValidationError(ValueError):
    """Raised when a turbine coordinate file violates the input contract."""


class InfeasibleDeploymentError(ValueError):
    """Raised when the configured UAV cannot inspect even one turbine."""


@dataclass(frozen=True, slots=True)
class Turbine:
    """A wind turbine in the projected wind-farm coordinate system."""

    identifier: str
    x_km: float
    y_km: float


@dataclass(frozen=True, slots=True)
class DeploymentConfig:
    """Paper defaults plus clearly named, replaceable modeling assumptions."""

    communication_radius_km: float = 7.0
    max_horizontal_speed_mps: float = 23.0
    wind_speed_mps: float = 6.0
    max_endurance_seconds: float = 41.0 * 60.0
    hover_seconds_per_turbine: float = 10.0
    hover_to_flight_energy_ratio: float = 1.14
    wind_direction_degrees: float = 90.0
    coordinate_unit: str = "km"

    def __post_init__(self) -> None:
        numeric_values = (
            self.communication_radius_km,
            self.max_horizontal_speed_mps,
            self.wind_speed_mps,
            self.max_endurance_seconds,
            self.hover_seconds_per_turbine,
            self.hover_to_flight_energy_ratio,
            self.wind_direction_degrees,
        )
        if not all(math.isfinite(value) for value in numeric_values):
            raise ValueError("All deployment parameters must be finite numbers.")
        if self.communication_radius_km <= 0 or self.max_horizontal_speed_mps <= 0:
            raise ValueError("Communication radius and maximum speed must be positive.")
        if self.max_endurance_seconds <= 0 or self.hover_seconds_per_turbine < 0:
            raise ValueError("Endurance must be positive and hover time cannot be negative.")
        if self.hover_to_flight_energy_ratio <= 0:
            raise ValueError("Hover-to-flight energy ratio must be positive.")
        if self.coordinate_unit != "km":
            raise ValueError("This reproduction accepts projected coordinates in km only.")
        if worst_case_ground_speed_mps(self) <= 0:
            raise ValueError("Wind speed must remain below maximum horizontal speed.")


@dataclass(frozen=True, slots=True)
class NestResult:
    """A retained nest and its final, uniquely assigned inspection group."""

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


@dataclass(frozen=True, slots=True)
class IterationRecord:
    """One replayable pruning state after zero or one redundant-nest removal."""

    iteration: int
    event: str
    removed_nest_id: str | None
    active_nest_ids: tuple[str, ...]
    active_nest_count: int
    covered_turbine_count: int
    minimum_cover_count: int
    maximum_cover_count: int


@dataclass(frozen=True, slots=True)
class DeploymentResult:
    """Complete auditable output of one deterministic deployment run."""

    config: DeploymentConfig
    effective_service_radius_km: float
    assignments: dict[str, str]
    nests: tuple[NestResult, ...]
    iterations: tuple[IterationRecord, ...]
    all_constraints_satisfied: bool


def load_config(path: Path) -> DeploymentConfig:
    """Load a JSON configuration without allowing unknown fields."""
    try:
        values = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise InputValidationError(f"Configuration file does not exist: {path}") from error
    except json.JSONDecodeError as error:
        raise InputValidationError(f"Invalid JSON configuration: {path}") from error

    if not isinstance(values, dict):
        raise InputValidationError("Configuration root must be a JSON object.")
    try:
        return DeploymentConfig(**values)
    except (TypeError, ValueError) as error:
        raise InputValidationError(f"Invalid deployment configuration: {error}") from error


def load_turbines(path: Path) -> list[Turbine]:
    """Read three-column TSV rows: turbine identifier, x km, y km."""
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.reader(handle, delimiter="\t"))
    except FileNotFoundError as error:
        raise InputValidationError(f"Turbine input does not exist: {path}") from error

    if not rows:
        raise InputValidationError("Turbine input is empty.")

    turbines: list[Turbine] = []
    identifiers: set[str] = set()
    for line_number, row in enumerate(rows, start=1):
        if len(row) != 3:
            raise InputValidationError(
                f"Line {line_number} must contain exactly three tab-separated fields."
            )
        identifier = row[0].strip()
        if not identifier:
            raise InputValidationError(f"Line {line_number} has an empty turbine identifier.")
        if identifier in identifiers:
            raise InputValidationError(f"Duplicate turbine identifier: {identifier}")
        try:
            x_km, y_km = float(row[1]), float(row[2])
        except ValueError as error:
            raise InputValidationError(
                f"Line {line_number} has non-numeric coordinates."
            ) from error
        if not math.isfinite(x_km) or not math.isfinite(y_km):
            raise InputValidationError(f"Line {line_number} has non-finite coordinates.")
        identifiers.add(identifier)
        turbines.append(Turbine(identifier=identifier, x_km=x_km, y_km=y_km))
    return turbines


def worst_case_ground_speed_mps(config: DeploymentConfig) -> float:
    """Return the conservative headwind ground speed used for range checks."""
    return config.max_horizontal_speed_mps - config.wind_speed_mps


def effective_service_radius_km(config: DeploymentConfig) -> float:
    """Return min(communication radius, conservative round-trip flight radius)."""
    wind_limited_radius = (
        worst_case_ground_speed_mps(config) * config.max_endurance_seconds / 2_000.0
    )
    return min(config.communication_radius_km, wind_limited_radius)


def _distance_km(first: Turbine, second: Turbine) -> float:
    return math.hypot(first.x_km - second.x_km, first.y_km - second.y_km)


def _nearest_neighbor_route(nest: Turbine, members: tuple[Turbine, ...]) -> tuple[Turbine, ...]:
    """Build a deterministic closed-tour visit order; the return leg is implicit."""
    unvisited = list(members)
    route: list[Turbine] = []
    current = nest
    while unvisited:
        next_turbine = min(
            unvisited,
            key=lambda candidate: (_distance_km(current, candidate), candidate.identifier),
        )
        route.append(next_turbine)
        unvisited.remove(next_turbine)
        current = next_turbine
    return tuple(route)


def _route_distance_km(nest: Turbine, route: tuple[Turbine, ...]) -> float:
    if not route:
        return 0.0
    distance = _distance_km(nest, route[0])
    distance += sum(_distance_km(first, second) for first, second in zip(route, route[1:]))
    return distance + _distance_km(route[-1], nest)


def _mission_metrics(
    nest: Turbine, members: tuple[Turbine, ...], config: DeploymentConfig
) -> tuple[tuple[Turbine, ...], float, float, float]:
    route = _nearest_neighbor_route(nest, members)
    route_distance = _route_distance_km(nest, route)
    flight_seconds = route_distance * 1_000.0 / worst_case_ground_speed_mps(config)
    equivalent_energy_seconds = flight_seconds + (
        len(members)
        * config.hover_seconds_per_turbine
        * config.hover_to_flight_energy_ratio
    )
    return route, route_distance, flight_seconds, equivalent_energy_seconds


def _is_feasible_group(
    nest: Turbine, members: tuple[Turbine, ...], config: DeploymentConfig
) -> bool:
    radius = effective_service_radius_km(config)
    if any(_distance_km(nest, member) > radius + _EPSILON for member in members):
        return False
    _, _, _, equivalent_energy = _mission_metrics(nest, members, config)
    return equivalent_energy <= config.max_endurance_seconds + _EPSILON


def _build_candidate_groups(
    turbines: list[Turbine], config: DeploymentConfig
) -> dict[str, tuple[str, ...]]:
    """Construct one capacity-feasible candidate group per possible nest."""
    ordered_turbines = sorted(turbines, key=lambda item: item.identifier)
    radius = effective_service_radius_km(config)
    groups: dict[str, tuple[str, ...]] = {}
    for nest in ordered_turbines:
        members = [nest]
        candidates = sorted(
            (
                turbine
                for turbine in ordered_turbines
                if turbine.identifier != nest.identifier
                and _distance_km(nest, turbine) <= radius + _EPSILON
            ),
            key=lambda item: (_distance_km(nest, item), item.identifier),
        )
        for candidate in candidates:
            proposed_members = tuple(members + [candidate])
            if _is_feasible_group(nest, proposed_members, config):
                members.append(candidate)
        if not _is_feasible_group(nest, tuple(members), config):
            raise InfeasibleDeploymentError(
                f"Nest {nest.identifier} cannot feasibly inspect its own turbine."
            )
        groups[nest.identifier] = tuple(member.identifier for member in members)
    return groups


def _record_iteration(
    iteration: int,
    event: str,
    removed_nest_id: str | None,
    active: dict[str, tuple[str, ...]],
    support_count: dict[str, int],
) -> IterationRecord:
    supports = tuple(support_count.values())
    return IterationRecord(
        iteration=iteration,
        event=event,
        removed_nest_id=removed_nest_id,
        active_nest_ids=tuple(sorted(active)),
        active_nest_count=len(active),
        covered_turbine_count=sum(count > 0 for count in supports),
        minimum_cover_count=min(supports),
        maximum_cover_count=max(supports),
    )


def _prune_redundant_groups(
    groups: dict[str, tuple[str, ...]]
) -> tuple[dict[str, tuple[str, ...]], tuple[IterationRecord, ...]]:
    """Remove a group only when every member keeps another active cover."""
    active = dict(groups)
    support_count: dict[str, int] = {}
    for group in active.values():
        for turbine_id in group:
            support_count[turbine_id] = support_count.get(turbine_id, 0) + 1

    records = [_record_iteration(0, "initial_candidate_groups", None, active, support_count)]
    iteration = 0

    while True:
        removable = [
            nest_id
            for nest_id, group in active.items()
            if all(support_count[turbine_id] >= 2 for turbine_id in group)
        ]
        if not removable:
            return active, tuple(records)
        nest_id = min(removable, key=lambda item: (len(active[item]), item))
        for turbine_id in active[nest_id]:
            support_count[turbine_id] -= 1
        del active[nest_id]
        iteration += 1
        records.append(
            _record_iteration(
                iteration,
                "remove_redundant_nest",
                nest_id,
                active,
                support_count,
            )
        )


def assign_turbines(
    turbines: list[Turbine], active_groups: dict[str, tuple[str, ...]]
) -> dict[str, str]:
    """Assign each turbine to its nearest active feasible nest, then break ties by ID."""
    by_identifier = {turbine.identifier: turbine for turbine in turbines}
    assignments: dict[str, str] = {}
    for turbine in sorted(turbines, key=lambda item: item.identifier):
        eligible_nests = [
            nest_id for nest_id, group in active_groups.items() if turbine.identifier in group
        ]
        if not eligible_nests:
            raise InfeasibleDeploymentError(
                f"Turbine {turbine.identifier} is not covered by any active nest."
            )
        assignments[turbine.identifier] = min(
            eligible_nests,
            key=lambda nest_id: (_distance_km(turbine, by_identifier[nest_id]), nest_id),
        )
    return assignments


def run_deployment(turbines: list[Turbine], config: DeploymentConfig) -> DeploymentResult:
    """Run candidate construction, redundancy pruning, unique assignment, and checks."""
    if not turbines:
        raise InputValidationError("At least one turbine is required for deployment.")
    identifiers = [turbine.identifier for turbine in turbines]
    if len(set(identifiers)) != len(identifiers):
        raise InputValidationError("Turbine identifiers must be unique.")

    by_identifier = {turbine.identifier: turbine for turbine in turbines}
    candidate_groups = _build_candidate_groups(turbines, config)
    active_groups, iterations = _prune_redundant_groups(candidate_groups)
    assignments = assign_turbines(turbines, active_groups)
    radius = effective_service_radius_km(config)

    nests: list[NestResult] = []
    for nest_id in sorted(active_groups):
        nest = by_identifier[nest_id]
        assigned_ids = tuple(
            turbine_id for turbine_id, assigned_nest in assignments.items() if assigned_nest == nest_id
        )
        assigned_members = tuple(by_identifier[turbine_id] for turbine_id in assigned_ids)
        if not _is_feasible_group(nest, assigned_members, config):
            raise InfeasibleDeploymentError(
                f"Final assignment for nest {nest_id} violates a deployment constraint."
            )
        route, route_distance, flight_seconds, equivalent_energy = _mission_metrics(
            nest, assigned_members, config
        )
        largest_member_distance = max(
            (_distance_km(nest, member) for member in assigned_members), default=0.0
        )
        nests.append(
            NestResult(
                identifier=nest_id,
                x_km=nest.x_km,
                y_km=nest.y_km,
                turbine_ids=assigned_ids,
                route_ids=tuple(member.identifier for member in route),
                route_distance_km=route_distance,
                flight_seconds=flight_seconds,
                equivalent_energy_seconds=equivalent_energy,
                service_radius_margin_km=radius - largest_member_distance,
                endurance_margin_seconds=config.max_endurance_seconds - equivalent_energy,
            )
        )

    all_constraints_satisfied = (
        set(assignments) == set(identifiers)
        and set(assignments.values()) == {nest.identifier for nest in nests}
        and all(nest.service_radius_margin_km >= -_EPSILON for nest in nests)
        and all(nest.endurance_margin_seconds >= -_EPSILON for nest in nests)
    )
    return DeploymentResult(
        config=config,
        effective_service_radius_km=radius,
        assignments=assignments,
        nests=tuple(nests),
        iterations=iterations,
        all_constraints_satisfied=all_constraints_satisfied,
    )


def result_as_dict(result: DeploymentResult) -> dict[str, Any]:
    """Convert a result to JSON-safe, stably ordered audit data."""
    return {
        "all_constraints_satisfied": result.all_constraints_satisfied,
        "assignments": dict(sorted(result.assignments.items())),
        "config": asdict(result.config),
        "effective_service_radius_km": result.effective_service_radius_km,
        "nest_count": len(result.nests),
        "nests": [asdict(nest) for nest in result.nests],
        "iterations": [asdict(record) for record in result.iterations],
    }
