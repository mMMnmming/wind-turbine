"""Versioned, instance-local physics oracle. Candidate claims are never trusted.

The runner exposes bound public methods only. This is a convenience and work
budget, not an OS security boundary or a guarantee that arbitrary code cooperates.
"""

from __future__ import annotations

import copy
import math
import time

import numpy as np

from .domain import DeploymentConfig, Turbine, effective_service_radius_km
from .routing import directed_times, proxy_route

TOOLS_VERSION = 1
PUBLIC_METHODS = (
    "distance",
    "directed_time",
    "region",
    "deployment",
    "tick",
    "remaining",
    "phase",
    "remember",
    "best",
    "split_repair",
    "stats",
)


class BudgetExceeded(RuntimeError):
    """Cooperative limit; only an explicitly remembered feasible answer may return."""


class CandidateTools:
    def __init__(
        self,
        points: list[Turbine],
        config: DeploymentConfig,
        *,
        soft_seconds: float = 120,
        max_region_evaluations: int = 5000,
        max_move_attempts: int = 20000,
        observer=None,
    ):
        self.points = tuple(points)
        self._config = config
        self._by_id = {t.identifier: i for i, t in enumerate(points)}
        if not points or len(self._by_id) != len(points):
            raise ValueError("Tools require nonempty unique turbines.")
        xy = np.array([(t.x_km, t.y_km) for t in points])
        if not np.isfinite(xy).all():
            raise ValueError("Coordinates must be finite.")
        if not math.isfinite(soft_seconds) or soft_seconds <= 0:
            raise ValueError("soft_seconds must be positive and finite.")
        if any(
            type(v) is not int or v < 1
            for v in (max_region_evaluations, max_move_attempts)
        ):
            raise ValueError("Work limits must be positive integers.")
        # Match the existing audit's math.hypot exactly, including distance ties.
        self._distance = np.array(
            [
                [math.hypot(a.x_km - b.x_km, a.y_km - b.y_km) for b in points]
                for a in points
            ]
        )
        self._times = directed_times(list(self.points), config)
        self._start = time.perf_counter()
        self._seconds = soft_seconds
        self._region_limit, self._move_limit = max_region_evaluations, max_move_attempts
        self._evaluations = self._hits = self._moves = 0
        self._cache: dict = {}
        self._best: dict | None = None
        self._observer = observer
        self._phase = "initialization"
        self._last_progress = self._start

    def _check(self):
        if time.perf_counter() - self._start >= self._seconds:
            raise BudgetExceeded("soft time budget exhausted")
        if time.perf_counter() - self._last_progress >= 10:
            self._emit()

    def _emit(self):
        self._last_progress = time.perf_counter()
        if self._observer:
            self._observer(self.stats())

    def distance(self, first: str, second: str) -> float:
        self._check()
        return float(self._distance[self._by_id[first], self._by_id[second]])

    def directed_time(self, first: str, second: str) -> float:
        self._check()
        return float(self._times[self._by_id[first], self._by_id[second]])

    def tick(self, count: int = 1) -> None:
        if type(count) is not int or count < 1:
            raise ValueError("tick count must be positive integer.")
        self._check()
        if self._moves + count > self._move_limit:
            raise BudgetExceeded("move attempt budget exhausted")
        self._moves += count

    def stats(self) -> dict:
        return {
            "region_evaluations": self._evaluations,
            "cache_hits": self._hits,
            "move_attempts": self._moves,
            "phase": self._phase,
            "seconds": time.perf_counter() - self._start,
            "tools_version": TOOLS_VERSION,
        }

    def remaining(self) -> dict:
        return {
            "seconds": max(0.0, self._seconds - (time.perf_counter() - self._start)),
            "region_evaluations": self._region_limit - self._evaluations,
            "move_attempts": self._move_limit - self._moves,
        }

    def phase(self, name: str) -> None:
        self._check()
        changed = self._phase != str(name)[:80]
        self._phase = str(name)[:80]
        if changed and time.perf_counter() - self._last_progress >= 0.25:
            self._emit()

    def region(self, nest_id: str, members, *, proxy: bool = True) -> dict:
        """Same tie-breaking/hover audit as evaluator; proxy may be deferred."""
        self._check()
        ids = tuple(sorted(members))
        if nest_id not in self._by_id or any(t not in self._by_id for t in ids):
            raise ValueError("Unknown nest/member ID.")
        if len(set(ids)) != len(ids) or nest_id not in ids:
            return {
                "feasible": False,
                "diagnostics": [
                    {"kind": "self_assignment_or_duplicates", "nest_id": nest_id}
                ],
                "task_count": max(0, len(ids) - 1),
            }
        key = (nest_id, ids, bool(proxy))
        if key in self._cache:
            self._hits += 1
            return copy.deepcopy(self._cache[key])
        if self._evaluations >= self._region_limit:
            raise BudgetExceeded("region evaluation budget exhausted")
        self._evaluations += 1
        depot = self._by_id[nest_id]
        indices = [self._by_id[t] for t in ids]
        pending, route, current = set(indices), [], depot
        while pending:
            nxt = min(
                pending,
                key=lambda j: (self._distance[current, j], self.points[j].identifier),
            )
            route.append(nxt)
            pending.remove(nxt)
            current = nxt
        closed = [depot, *route, depot]
        distance = float(sum(self._distance[a, b] for a, b in zip(closed, closed[1:])))
        config = self._config
        flight = (
            distance * 1000 / (config.max_horizontal_speed_mps - config.wind_speed_mps)
        )
        energy = (
            flight
            + len(ids)
            * config.hover_seconds_per_turbine
            * config.hover_to_flight_energy_ratio
        )
        radius = effective_service_radius_km(config)
        actual_radius = float(max(self._distance[depot, j] for j in indices))
        diagnostics = []
        for kind, actual, limit in (
            ("service_radius", actual_radius, radius),
            ("endurance", energy, config.max_endurance_seconds),
        ):
            if actual > limit + 1e-9:
                diagnostics.append(
                    {
                        "kind": kind,
                        "nest_id": nest_id,
                        "actual": actual,
                        "limit": limit,
                        "excess": actual - limit,
                        "normalized_excess": (actual - limit) / limit,
                        "task_count": len(ids) - 1,
                    }
                )
        result = {
            "feasible": not diagnostics,
            "diagnostics": diagnostics,
            "nest_id": nest_id,
            "task_count": len(ids) - 1,
            "radius_km": actual_radius,
            "radius_margin_km": radius - actual_radius,
            "distance_km": distance,
            "flight_seconds": flight,
            "energy_seconds": energy,
            "endurance_margin_seconds": config.max_endurance_seconds - energy,
            "audit_route_ids": [self.points[j].identifier for j in closed],
            "proxy_flight_seconds": None,
            "proxy_route_ids": None,
        }
        if proxy and not diagnostics:
            order = [depot, *(j for j in indices if j != depot)]
            proxy_ids, proxy_flight = proxy_route(
                [self.points[j] for j in order],
                config,
                [self.points[j].identifier for j in route],
                times=self._times[np.ix_(order, order)],
            )
            result.update(
                proxy_flight_seconds=proxy_flight, proxy_route_ids=list(proxy_ids)
            )
        self._cache[key] = result
        return copy.deepcopy(result)

    def deployment(self, proposal: dict, *, proxy: bool = True) -> dict:
        self._check()
        nests, assignments = proposal.get("nest_ids", []), proposal.get(
            "assignments", {}
        )
        ids = set(self._by_id)
        if (
            not nests
            or len(set(nests)) != len(nests)
            or not set(nests) <= ids
            or set(assignments) != ids
            or not set(assignments.values()) <= set(nests)
            or any(assignments.get(n) != n for n in nests)
        ):
            return {
                "feasible": False,
                "diagnostics": [{"kind": "coverage_or_self_assignment"}],
                "objectives": None,
                "regions": [],
            }
        groups: dict[str, list[str]] = {n: [] for n in nests}
        for t, n in assignments.items():
            groups[n].append(t)
        regions = [self.region(n, groups[n], proxy=proxy) for n in sorted(nests)]
        diagnostics = [d for r in regions for d in r["diagnostics"]]
        flight = sum(r["proxy_flight_seconds"] or 0 for r in regions)
        objectives = (
            None
            if diagnostics or not proxy
            else [
                len(nests) / len(ids),
                flight / len(ids),
                max(r["task_count"] for r in regions) / len(ids),
            ]
        )
        return {
            "feasible": not diagnostics,
            "diagnostics": diagnostics,
            "objectives": objectives,
            "regions": regions,
        }

    def remember(self, proposal: dict) -> bool:
        # Explicit candidate choice, no framework-generated fallback or hidden weighting.
        checked = self.deployment(proposal, proxy=False)
        if not checked["feasible"]:
            return False
        self._best = copy.deepcopy(proposal)
        return True

    def best(self):
        return copy.deepcopy(self._best)

    def split_repair(self, assignments: dict) -> dict:
        """Optional explicit primitive: open nests until every region passes audit.

        May increase K by up to N; not a nest minimizer. Does not assume NN cost
        decreases after each removal. An infeasible intermediate is allowed.
        """
        if set(assignments) != set(self._by_id) or not set(assignments.values()) <= set(
            self._by_id
        ):
            raise ValueError(
                "split_repair requires complete assignments to input turbines."
            )
        answer = dict(assignments)
        for n in set(answer.values()):
            answer[n] = n
        for _ in range(len(answer) + 1):
            self.tick()
            nests = sorted(set(answer.values()))
            for n in nests:
                members = sorted(t for t, owner in answer.items() if owner == n)
                row = self.region(n, members, proxy=False)
                if row["feasible"]:
                    continue
                tasks = [t for t in members if t != n]
                if not tasks:
                    raise ValueError(
                        "Even a singleton violates hover/endurance constraints."
                    )
                opened = max(tasks, key=lambda t: (self.distance(n, t), t))
                answer[opened] = opened
                break
            else:
                return {"nest_ids": nests, "assignments": answer}
        raise ValueError("Split repair failed to terminate within N openings.")
