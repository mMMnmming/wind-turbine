"""Independent, trusted feasibility checks for generated deployment proposals."""

from __future__ import annotations

from dataclasses import dataclass

from .domain import (
    DeploymentConfig,
    DeploymentResult,
    NestResult,
    Turbine,
    distance_km,
    mission_metrics,
    effective_service_radius_km,
)
from .candidate import CandidateProposal
from .reward import CandidateMetrics
from .routing import proxy_route
import math


@dataclass(frozen=True)
class Evaluation:
    feasible: bool
    metrics: CandidateMetrics
    deployment: DeploymentResult | None
    violations: tuple[str, ...]
    diagnostics: tuple[dict, ...] = ()


def evaluate_proposal(
    turbines: list[Turbine], config: DeploymentConfig, proposal: CandidateProposal
) -> Evaluation:
    violations: list[str] = []
    diagnostics: list[dict] = []
    by_id = {t.identifier: t for t in turbines}
    if not turbines or len(by_id) != len(turbines):
        violations.append("Evaluator input is empty or has duplicate turbine IDs.")
    if any(not math.isfinite(t.x_km) or not math.isfinite(t.y_km) for t in turbines):
        violations.append("Input coordinates must be finite.")
    nests = list(proposal.nest_ids)
    if not nests or len(set(nests)) != len(nests):
        violations.append("nest_ids must be non-empty and unique.")
    if any(nid not in by_id for nid in nests):
        violations.append("Every nest ID must refer to an input turbine coordinate.")
    expected = set(by_id)
    assigned = set(proposal.assignments)
    if assigned != expected:
        missing, extra = sorted(expected - assigned), sorted(assigned - expected)
        violations.append(
            f"Assignment coverage mismatch; missing={missing[:8]}, extra={extra[:8]}."
        )
    if any(nid not in nests for nid in proposal.assignments.values()):
        violations.append("Assignments reference an undeclared nest.")
    if any(proposal.assignments.get(nid) != nid for nid in nests):
        violations.append(
            "Nest self-assignment required: every nest turbine belongs to its own region."
        )
    if violations:
        return _invalid(violations)

    radius = effective_service_radius_km(config)
    nest_results: list[NestResult] = []
    for nest_id in sorted(nests):
        members_ids = sorted(
            tid for tid, owner in proposal.assignments.items() if owner == nest_id
        )
        if not members_ids:
            violations.append(f"Nest {nest_id} has no assigned turbines.")
            continue
        nest = by_id[nest_id]
        members = tuple(by_id[tid] for tid in members_ids)
        largest_dist = max(distance_km(nest, turbine) for turbine in members)
        if largest_dist > radius + 1e-9:
            violations.append(
                f"Nest {nest_id} service-radius violation: {largest_dist:.6f} > {radius:.6f} km."
            )
            diagnostics.append(
                {
                    "kind": "service_radius",
                    "nest_id": nest_id,
                    "actual": largest_dist,
                    "limit": radius,
                    "excess": largest_dist - radius,
                    "normalized_excess": (largest_dist - radius) / radius,
                    "task_count": len(members) - 1,
                }
            )
        route, distance, flight, energy = mission_metrics(nest, members, config)
        endurance_margin = config.max_endurance_seconds - energy
        if endurance_margin < -1e-9:
            violations.append(
                f"Nest {nest_id} endurance violation: {energy:.3f} > {config.max_endurance_seconds:.3f} s."
            )
            diagnostics.append(
                {
                    "kind": "endurance",
                    "nest_id": nest_id,
                    "actual": energy,
                    "limit": config.max_endurance_seconds,
                    "excess": -endurance_margin,
                    "normalized_excess": -endurance_margin
                    / config.max_endurance_seconds,
                    "task_count": len(members) - 1,
                }
            )
        proxy_ids, proxy_flight = (), 0.0
        if largest_dist <= radius + 1e-9 and endurance_margin >= -1e-9:
            points = [nest, *(t for t in members if t.identifier != nest_id)]
            proxy_ids, proxy_flight = proxy_route(
                points, config, tuple(t.identifier for t in route)
            )
        tasks = len(members) - 1
        nest_results.append(
            NestResult(
                identifier=nest_id,
                x_km=nest.x_km,
                y_km=nest.y_km,
                turbine_ids=tuple(members_ids),
                route_ids=tuple(item.identifier for item in route),
                route_distance_km=distance,
                flight_seconds=flight,
                equivalent_energy_seconds=energy,
                service_radius_margin_km=radius - largest_dist,
                endurance_margin_seconds=endurance_margin,
                proxy_route_ids=proxy_ids,
                proxy_flight_seconds=proxy_flight,
                proxy_equivalent_energy_seconds=proxy_flight
                + tasks
                * config.hover_seconds_per_turbine
                * config.hover_to_flight_energy_ratio,
                task_count=tasks,
            )
        )
    if violations:
        return _invalid(violations, diagnostics)

    metrics = CandidateMetrics(
        True,
        len(nest_results),
        min(n.endurance_margin_seconds for n in nest_results),
        min(n.service_radius_margin_km for n in nest_results),
        sum(n.route_distance_km for n in nest_results),
        sum(n.proxy_flight_seconds for n in nest_results),
        max(n.task_count for n in nest_results),
        tuple(n.task_count for n in nest_results),
    )
    deployment = DeploymentResult(
        config, radius, dict(proposal.assignments), tuple(nest_results), True
    )
    return Evaluation(True, metrics, deployment, ())


def _invalid(
    violations: list[str], diagnostics: list[dict] | None = None
) -> Evaluation:
    return Evaluation(
        False,
        CandidateMetrics(False, 10**9, -1e9, -1e9, 1e9),
        None,
        tuple(violations),
        tuple(diagnostics or [{"kind": "contract", "message": v} for v in violations]),
    )
