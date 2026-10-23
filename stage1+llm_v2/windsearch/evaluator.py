"""Canonical full-deployment validation; the strategy has no acceptance authority."""
from __future__ import annotations

import math
import time
from collections import defaultdict
from dataclasses import asdict

from .artifacts import signature
from .domain import Turbine
from .sandbox import CandidateDeadlineExceeded


class WorkBudgetExceeded(RuntimeError):
    pass


def problem_input(points, config, oracle):
    import numpy as np
    from stage2.atsp import WindATSPConfig, WindATSPInstance

    instance = WindATSPInstance(tuple(p.identifier for p in points),
                               np.asarray([(p.x_km, p.y_km) for p in points]), 0,
                               WindATSPConfig(**oracle["environment"]))
    return {"turbines": [asdict(p) for p in points], "config": asdict(config),
            "environment": oracle["environment"],
            "travel_time_seconds": instance.travel_time_seconds.tolist(),
            "time_matrix_ids": [p.identifier for p in points],
            "objective_order": ["nest_count", "total_flight_seconds", "maximum_flight_seconds"]}


class Evaluator:
    def __init__(self, points, config, planner, records, scope, heartbeat):
        self.points = {p.identifier: p for p in points}
        self.config, self.planner, self.records = config, planner, records
        self.scope, self.heartbeat = scope, heartbeat
        self.seconds = 0.0

    def check(self, assignments, *, deadline=None, inference_limit=None):
        start = time.perf_counter()
        try:
            return self._check(assignments, deadline, inference_limit)
        finally:
            self.seconds += time.perf_counter() - start

    def _check(self, assignments, deadline, inference_limit):
        diagnostics, regions = [], []
        if not isinstance(assignments, dict) or any(
                not isinstance(t, str) or not isinstance(n, str) for t, n in assignments.items()):
            diagnostics.append({"kind": "invalid_mapping", "message": "Expected dict[str,str]"})
        elif set(assignments) != set(self.points):
            diagnostics.append({"kind": "coverage", "missing": sorted(set(self.points) - set(assignments)),
                                "extra": sorted(set(assignments) - set(self.points))})
        elif not set(assignments.values()) <= set(self.points):
            diagnostics.append({"kind": "invalid_nest"})
        elif any(assignments.get(n) != n for n in set(assignments.values())):
            diagnostics.append({"kind": "nest_self_assignment"})
        if diagnostics:
            return self.result(assignments, diagnostics, regions)
        groups = defaultdict(list)
        for t, n in assignments.items():
            groups[n].append(self.points[t])
        # Reject radius violations before invoking the network for any region.
        for n, members in groups.items():
            p = self.points[n]
            radius = max(math.hypot(t.x_km-p.x_km, t.y_km-p.y_km) for t in members)
            if radius > self.config.communication_radius_km + 1e-9:
                diagnostics.append({"kind": "service_radius", "nest_id": n,
                                    "actual_km": radius, "limit_km": self.config.communication_radius_km})
        if diagnostics:
            return self.result(assignments, diagnostics, regions)
        for n in sorted(groups):
            remaining = None if deadline is None else deadline - time.perf_counter()
            if remaining is not None and remaining <= 0:
                raise CandidateDeadlineExceeded("Search deadline before region evaluation")
            if inference_limit is not None and self.planner.stats().get("inferences", 0) >= inference_limit:
                raise WorkBudgetExceeded("Stage2 inference budget exhausted")
            result = self.planner.region(groups[n], n, timeout=remaining)
            ref = self.records.blob(result)
            self.records.append("diagnostics/tool_calls/" + self.scope + ".jsonl",
                                {"kind": "stage2_region", "nest": n,
                                 "members": [p.identifier for p in groups[n]], "result": ref})
            regions.append(result)
            diagnostics.extend(dict(d, nest_id=n) for d in result["diagnostics"])
        return self.result(assignments, diagnostics, regions)

    @staticmethod
    def result(assignments, diagnostics, regions):
        feasible = not diagnostics and bool(regions)
        metrics = None
        if regions and all("flight_seconds" in r for r in regions):
            total = sum(r["flight_seconds"] for r in regions)
            metrics = {"nest_count": len(regions), "total_flight_seconds": total,
                       "mean_flight_seconds": total / len(regions),
                       "maximum_flight_seconds": max(r["flight_seconds"] for r in regions),
                       "minimum_endurance_margin_seconds": min(r["endurance_margin_seconds"] for r in regions)}
        objectives = ([metrics["nest_count"], metrics["total_flight_seconds"],
                       metrics["maximum_flight_seconds"]] if feasible else None)
        return {"feasible": feasible, "metrics": metrics, "objectives": objectives,
                "diagnostics": diagnostics, "regions": regions,
                "assignment_sha256": signature(assignments)}


def solution(assignments, checked):
    return {"assignments": dict(assignments), "nest_ids": sorted(set(assignments.values())),
            "objectives": checked["objectives"], "metrics": checked["metrics"]}


def compact_feedback(checked):
    regions = checked.get("regions", [])
    longest = max(regions, key=lambda r: r.get("flight_seconds", 0), default=None)
    tightest = min(regions, key=lambda r: r.get("endurance_margin_seconds", 0), default=None)
    def brief(r):
        return {k: r.get(k) for k in ("nest_id", "task_count", "flight_seconds",
                                      "endurance_margin_seconds")} if r else None
    return {"feasible": checked["feasible"], "objectives": checked["objectives"],
            "diagnostics": checked["diagnostics"][:4],
            "longest_region": brief(longest), "tightest_region": brief(tightest)}
