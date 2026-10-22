"""Lexicographic deployment objective and bounded scalar reward."""

from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True)
class CandidateMetrics:
    feasible: bool
    nest_count: int
    minimum_endurance_margin_seconds: float
    minimum_service_margin_km: float
    total_route_distance_km: float
    total_proxy_flight_seconds: float = 0.0
    maximum_region_tasks: int = 0
    region_task_counts: tuple[int, ...] = ()


def compare_metrics(first: CandidateMetrics, second: CandidateMetrics) -> int:
    if first.feasible != second.feasible:
        return -1 if first.feasible else 1
    if not first.feasible:
        return 0
    left = (
        first.nest_count,
        -first.minimum_endurance_margin_seconds,
        -first.minimum_service_margin_km,
        first.total_route_distance_km,
    )
    right = (
        second.nest_count,
        -second.minimum_endurance_margin_seconds,
        -second.minimum_service_margin_km,
        second.total_route_distance_km,
    )
    return -1 if left < right else (1 if left > right else 0)


def candidate_reward(
    metrics: CandidateMetrics,
    lower_bound: int = 1,
    max_endurance: float = 2460.0,
    service_radius: float = 7.0,
    incumbent: CandidateMetrics | None = None,
) -> float:
    if not metrics.feasible:
        return -1.0
    if incumbent is not None:
        relation = compare_metrics(metrics, incumbent)
        return 0.25 if relation < 0 else (-0.25 if relation > 0 else 0.05)
    gap = max(0, metrics.nest_count - lower_bound)
    # One nest-count improvement dominates the complete range of tie-break metrics.
    tie = (
        0.5
        * max(-1.0, min(1.0, metrics.minimum_endurance_margin_seconds / max_endurance))
        + 0.25 * max(-1.0, min(1.0, metrics.minimum_service_margin_km / service_radius))
        - 0.25 * min(1.0, metrics.total_route_distance_km / 10_000.0)
    )
    return max(0.0, min(1.0, 1.0 / (1.0 + 10.0 * gap) + 0.04 * tie / (1.0 + gap)))


def directional_reward(
    candidate: CandidateMetrics,
    parent: CandidateMetrics | None,
    violation_count: int = 0,
) -> float:
    """Give an edit operator credit without replacing the deployment objective."""
    if not candidate.feasible:
        return -1.0 - min(1.0, violation_count * 0.1)
    if parent is None or not parent.feasible:
        return 1.0
    relation = compare_metrics(candidate, parent)
    return 0.75 if relation < 0 else (-0.50 if relation > 0 else 0.0)


OBJECTIVE_VERSION = (
    "mean_nests_per_turbine,mean_proxy_seconds_per_turbine,max_tasks_fraction:v1"
)


def dominates(first: list[float], second: list[float]) -> bool:
    return all(a <= b + 1e-9 for a, b in zip(first, second)) and any(
        a < b - 1e-9 for a, b in zip(first, second)
    )


def pareto_front(records: list[dict]) -> list[dict]:
    return [
        row
        for row in records
        if not any(
            dominates(other["objectives"], row["objectives"])
            for other in records
            if other is not row
        )
    ]


def parent_pool(records: list[dict], limit: int) -> list[dict]:
    if len(records) <= limit:
        return list(records)
    scores = {row["candidate_id"]: 0.0 for row in records}
    for axis in range(3):
        ordered = sorted(
            records, key=lambda row: (row["objectives"][axis], row["candidate_id"])
        )
        span = ordered[-1]["objectives"][axis] - ordered[0]["objectives"][axis]
        if span <= 1e-9:
            continue
        scores[ordered[0]["candidate_id"]] = float("inf")
        scores[ordered[-1]["candidate_id"]] = float("inf")
        for i in range(1, len(ordered) - 1):
            scores[ordered[i]["candidate_id"]] += (
                ordered[i + 1]["objectives"][axis] - ordered[i - 1]["objectives"][axis]
            ) / span
    return sorted(
        records, key=lambda row: (-scores[row["candidate_id"]], row["candidate_id"])
    )[:limit]


def search_reward(
    feasible: bool,
    objectives: list[float] | None,
    parent: list[float] | None,
    *,
    duplicate: bool = False,
    new_front: bool = False,
    violation_count: int = 0,
    violation_severity: float | None = None,
) -> float:
    if duplicate:
        return 0.0
    if not feasible:
        return -1.0 - min(
            1.0,
            (
                0.1 * violation_count
                if violation_severity is None
                else max(0.0, violation_severity)
            ),
        )
    if parent is None:
        return 1.0
    if dominates(objectives, parent):
        return 0.75
    if dominates(parent, objectives):
        return -0.5
    return 0.25 if new_front else 0.0
