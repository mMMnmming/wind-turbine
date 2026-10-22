"""Versioned, instance-local physics oracle. Candidate claims are never trusted.

The runner exposes bound public methods only. This is a convenience and work
budget, not an OS security boundary or a guarantee that arbitrary code cooperates.
"""

from __future__ import annotations

import copy
import inspect
import math
import time

import numpy as np

from .domain import DeploymentConfig, Turbine, effective_service_radius_km
from .routing import directed_times, proxy_route

TOOLS_VERSION = 3
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
    "make_proposal",
    "assignments_from_proposal",
    "repartition_repair",
    "objective_dominates",
    "prefer_objectives",
)

# Signatures are taken from executable methods; semantic schemas remain explicit.
_TOOL_NOTES = {
    "distance": "两个参数都是输入风机ID字符串，不是坐标tuple或风机对象；单位km。例distance(t1.identifier,t2.identifier)。",
    "directed_time": "两个参数都是输入风机ID字符串，不是坐标；返回有向秒，反向可不同。",
    "region": "members为含机巢的ID序列/集合。返回字典feasible,diagnostics,task_count；合法成员时另有energy_seconds等。proxy=True且可行才有代理航时，粗筛显式proxy=False。",
    "deployment": "proposal恰有nest_ids:list[str]及assignments:dict[str,str]。返回字典feasible,diagnostics,regions,objectives；objectives仅可行且proxy=True时为三元素列表，否则None。",
    "tick": "邻域尝试计数；正整数count，预算耗尽抛BudgetExceeded。",
    "remaining": "返回字典seconds,region_evaluations,move_attempts；用remaining()['seconds']，不能[0]。",
    "phase": "报告阶段字符串，仅诊断，不证明实际进行了优化。",
    "remember": "输入完整proposal，返回bool；独立审核后保存显式选择的可行解，不自动比较质量。",
    "best": "返回已保存完整proposal深拷贝或None。",
    "stats": "返回工具实际观测的计数/阶段/错误字典；不是优化成功证明。",
    "make_proposal": "输入dict[str,str]完整归属，返回完整proposal深拷贝；不补齐/修复物理违规。",
    "assignments_from_proposal": "输入完整proposal，返回归属映射深拷贝，不是完整部署。",
    "split_repair": "旧修复原语：输入完整归属，返回完整proposal；逐个打开单点巢，可能产生大量零任务区域，不是优化器。",
    "repartition_repair": "可选新修复原语：输入完整归属，保留可行区域，将违规区域重分配到逐步新增的机巢，每次加入任务按原保守约束审核。返回完整proposal，不自动保存、不自动继续优化、不保证最优。",
    "objective_dominates": "输入两个有限三元素目标列表；返回first是否Pareto支配second（全部不差且至少一项更好）。",
    "prefer_objectives": "三目标内部偏好：必须显式给非负weights及正scales，各长3；先比较支配关系，折中时比较sum(weights[i]*objective[i]/scales[i])，相等保留旧解。不影响外部三目标/Pareto选档。",
}


def tool_reference(names=None) -> str:
    """Render the same API metadata for initial and targeted repair prompts."""
    rows = []
    for name in PUBLIC_METHODS if names is None else names:
        signature = inspect.signature(getattr(CandidateTools, name))
        signature = signature.replace(
            parameters=list(signature.parameters.values())[1:]
        )
        rows.append(f"- {name}{signature}: {_TOOL_NOTES[name]}")
    return "\n".join(rows)


def deployment_quality(task_counts, turbine_count: int, telemetry: dict) -> dict:
    """Descriptive diagnostics only: never feasibility or a fourth objective."""
    loads = list(task_counts)
    zeros = sum(t == 0 for t in loads)
    average = sum(loads) / len(loads) if loads else 0.0
    flags = ["zero_task_regions_present"] if zeros else []
    observable = telemetry.get("tools_version", 0) >= 3
    if observable and telemetry.get("repair_completed_count", 0):
        if not telemetry.get("post_repair_evaluation_calls", 0):
            flags.append(
                "repair_return_without_external_checks"
                if telemetry.get("termination") == "returned"
                else "repair_exit_without_external_checks"
            )
    return {
        "region_task_counts": loads,
        "zero_task_regions": zeros,
        "zero_task_fraction": zeros / len(loads) if loads else 0.0,
        "nest_fraction": len(loads) / turbine_count,
        "maximum_region_tasks": max(loads, default=0),
        "task_imbalance_cv": (
            math.sqrt(sum((t - average) ** 2 for t in loads) / len(loads)) / average
            if average
            else 0.0
        ),
        "repair_observability": "available" if observable else "unavailable",
        "repair_calls": (
            copy.deepcopy(telemetry.get("repair_calls")) if observable else None
        ),
        "post_repair_evaluation_calls": (
            telemetry.get("post_repair_evaluation_calls") if observable else None
        ),
        "changed_feasible_saves_after_repair": (
            telemetry.get("changed_feasible_saves_after_repair") if observable else None
        ),
        "flags": flags,
        "note": "Tool-observed checks/changed saves do not prove objective improvement; zero-task regions may be necessary.",
    }


def quality_summary(records: list[dict], expected: dict | None = None) -> dict:
    """Keep split-level quality and coverage separate from Pareto objectives."""
    result = {}
    for split in [
        "all",
        *sorted({r.get("split", "unspecified") for r in records} | set(expected or {})),
    ]:
        rows = (
            records
            if split == "all"
            else [r for r in records if r.get("split", "unspecified") == split]
        )
        valid = [r for r in rows if r.get("feasible") and r.get("metrics")]
        qualities = [r["quality"] for r in valid if r.get("quality")]
        result[split] = {
            "covered_instances": len(rows),
            "feasible_instances": len(valid),
            "expected_instances": (
                (sum(expected.values()) if split == "all" else expected.get(split, 0))
                if expected
                else len(rows)
            ),
            "mean_nest_count": (
                sum(r["metrics"]["nest_count"] for r in valid) / len(valid)
                if valid
                else None
            ),
            "mean_total_proxy_flight_seconds": (
                sum(r["metrics"]["total_proxy_flight_seconds"] for r in valid)
                / len(valid)
                if valid
                else None
            ),
            "maximum_region_tasks": max(
                (r["metrics"]["maximum_region_tasks"] for r in valid), default=None
            ),
            "mean_zero_task_fraction": (
                sum(q["zero_task_fraction"] for q in qualities) / len(qualities)
                if qualities
                else None
            ),
            "repair_observable_instances": sum(
                q["repair_observability"] == "available" for q in qualities
            ),
            "repair_without_followup_instances": sum(
                "repair_return_without_external_checks" in q["flags"] for q in qualities
            ),
            "worst_zero_task_instances": [
                r["name"]
                for r in sorted(
                    valid,
                    key=lambda r: r.get("quality", {}).get("zero_task_fraction", 0),
                    reverse=True,
                )[:3]
            ],
        }
    return result


class BudgetExceeded(RuntimeError):
    """Cooperative limit; only an explicitly remembered feasible answer may return."""


class ToolContractError(ValueError):
    """Malformed tool data, not a physically infeasible deployment."""


class ContractResult(dict):
    """Dictionary-compatible result with diagnosed incorrect field reads.

    Explicit dict(result) conversion can remove this guard. The final schema and
    independent evaluator remain mandatory; this is not general program tracing.
    """

    def __init__(self, value: dict, tool: str, report):
        super().__init__(value)
        self._tool, self._report = tool, report

    def _guard(self, key, *, required):
        if not isinstance(key, str):
            self._report(
                self._tool, "result key", "string field name", key, "invalid_result_key"
            )
        if required and key not in self:
            fields = ", ".join(sorted(str(k) for k in self)[:12])
            self._report(
                self._tool,
                f"result[{key[:80]}]",
                f"available fields: {fields}",
                key,
                "unknown_result_field",
            )

    def __getitem__(self, key):
        self._guard(key, required=True)
        return super().__getitem__(key)

    def get(self, key, default=None):
        self._guard(key, required=False)
        return super().get(key, default)

    def pop(self, key, *default):
        self._guard(key, required=not default)
        return super().pop(key, *default)

    def setdefault(self, key, default=None):
        self._guard(key, required=False)
        return super().setdefault(key, default)

    def copy(self):
        return ContractResult(dict(self), self._tool, self._report)

    def __deepcopy__(self, memo):
        return ContractResult(copy.deepcopy(dict(self), memo), self._tool, self._report)


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
        candidate_filename: str | None = None,
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
        self._candidate_filename = candidate_filename
        self._contract_error_count = 0
        self._contract_errors: list[dict] = []
        self._repair_calls = {"split_repair": 0, "repartition_repair": 0}
        self._repair_completed_count = 0
        self._last_repair_proposal = None
        self._post_repair_evaluation_calls = 0
        self._changed_saves_after_repair = 0
        self._last_preference: dict | None = None

    def _record_external_call(self, name):
        # Called by the facade, not internal remember()/repair audit calls.
        if self._last_repair_proposal is not None and name in ("region", "deployment"):
            self._post_repair_evaluation_calls += 1

    def _repair_start(self, name):
        self._repair_calls[name] += 1
        if self._observer:
            self._observer(
                {
                    "kind": "repair_start",
                    "tool": name,
                    "calls": self._repair_calls[name],
                }
            )

    def _repair_done(self, name, proposal):
        self._repair_completed_count += 1
        self._last_repair_proposal = copy.deepcopy(dict(proposal))
        self._post_repair_evaluation_calls = 0
        self._changed_saves_after_repair = 0
        if self._observer:
            self._observer(
                {
                    "kind": "repair_end",
                    "tool": name,
                    "nests": len(proposal["nest_ids"]),
                    "completed": self._repair_completed_count,
                }
            )
        return self._result(proposal, name)

    def _error(self, tool, field, expected, value, code="invalid_shape"):
        line = None
        frame = inspect.currentframe()
        try:
            while frame is not None:
                if frame.f_code.co_filename == self._candidate_filename:
                    line = frame.f_lineno
                    break
                frame = frame.f_back
        finally:
            del frame
        field, expected = str(field)[:120], str(expected)[:240]
        message = (
            f"{tool}: {field} requires {expected}; actual type={type(value).__name__}"
        )
        record = {
            "tool": tool,
            "code": code,
            "field": field,
            "expected": expected,
            "actual_type": type(value).__name__,
            "candidate_line": line,
            "message": message,
        }
        self._contract_error_count += 1
        if len(self._contract_errors) < 5:
            self._contract_errors.append(record)
            if self._observer:
                self._observer({"kind": "tool_contract_error", **copy.deepcopy(record)})
        raise ToolContractError(message)

    def _result(self, value, tool):
        return ContractResult(copy.deepcopy(dict(value)), tool, self._error)

    def _assignments_schema(self, tool, assignments):
        if not isinstance(assignments, dict):
            self._error(
                tool, "assignments", "dict[str,str], not a proposal", assignments
            )
        for key, value in assignments.items():
            if not isinstance(key, str):
                self._error(tool, "assignment key", "turbine ID string", key)
            if not isinstance(value, str):
                self._error(
                    tool,
                    f"assignments[{key[:60]}]",
                    "single nest ID string, not list/dict",
                    value,
                )

    def _proposal_schema(self, tool, proposal):
        if not isinstance(proposal, dict) or set(proposal) != {
            "nest_ids",
            "assignments",
        }:
            self._error(
                tool,
                "proposal",
                "dict with exactly nest_ids and assignments; use make_proposal(assignments)",
                proposal,
            )
        nests = proposal["nest_ids"]
        if not isinstance(nests, list):
            self._error(tool, "nest_ids", "list[str]", nests)
        for index, nest in enumerate(nests):
            if not isinstance(nest, str):
                self._error(tool, f"nest_ids[{index}]", "nest ID string", nest)
        self._assignments_schema(tool, proposal["assignments"])

    def make_proposal(self, assignments: dict) -> dict:
        self._check()
        self._assignments_schema("make_proposal", assignments)
        return self._result(
            {
                "nest_ids": sorted(set(assignments.values())),
                "assignments": dict(assignments),
            },
            "make_proposal",
        )

    def assignments_from_proposal(self, proposal: dict) -> dict:
        self._check()
        self._proposal_schema("assignments_from_proposal", proposal)
        return copy.deepcopy(dict(proposal["assignments"]))

    def _query_id(self, tool, field, value):
        if not isinstance(value, str) or value not in self._by_id:
            self._error(
                tool,
                field,
                "existing input turbine ID string",
                value,
                "invalid_reference",
            )

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
        self._query_id("distance", "first", first)
        self._query_id("distance", "second", second)
        return float(self._distance[self._by_id[first], self._by_id[second]])

    def directed_time(self, first: str, second: str) -> float:
        self._check()
        self._query_id("directed_time", "first", first)
        self._query_id("directed_time", "second", second)
        return float(self._times[self._by_id[first], self._by_id[second]])

    def tick(self, count: int = 1) -> None:
        if type(count) is not int or count < 1:
            self._error("tick", "count", "positive integer", count)
        self._check()
        if self._moves + count > self._move_limit:
            raise BudgetExceeded("move attempt budget exhausted")
        self._moves += count

    def stats(self) -> dict:
        return self._result(
            {
                "region_evaluations": self._evaluations,
                "cache_hits": self._hits,
                "move_attempts": self._moves,
                "phase": self._phase,
                "seconds": time.perf_counter() - self._start,
                "tools_version": TOOLS_VERSION,
                "contract_error_count": self._contract_error_count,
                "contract_errors": copy.deepcopy(self._contract_errors),
                "repair_calls": dict(self._repair_calls),
                "repair_completed_count": self._repair_completed_count,
                "post_repair_evaluation_calls": self._post_repair_evaluation_calls,
                "changed_feasible_saves_after_repair": self._changed_saves_after_repair,
                "last_objective_preference": copy.deepcopy(self._last_preference),
            },
            "stats",
        )

    def remaining(self) -> dict:
        return self._result(
            {
                "seconds": max(
                    0.0, self._seconds - (time.perf_counter() - self._start)
                ),
                "region_evaluations": self._region_limit - self._evaluations,
                "move_attempts": self._move_limit - self._moves,
            },
            "remaining",
        )

    def phase(self, name: str) -> None:
        self._check()
        changed = self._phase != str(name)[:80]
        self._phase = str(name)[:80]
        if changed and time.perf_counter() - self._last_progress >= 0.25:
            self._emit()

    def region(self, nest_id: str, members, *, proxy: bool = True) -> dict:
        """Same tie-breaking/hover audit as evaluator; proxy may be deferred."""
        self._check()
        if not isinstance(nest_id, str):
            self._error("region", "nest_id", "string", nest_id)
        if not isinstance(members, (list, tuple, set, frozenset)):
            self._error(
                "region",
                "members",
                "list/tuple/set/frozenset of turbine ID strings",
                members,
            )
        for item in members:
            if not isinstance(item, str):
                self._error("region", "member", "turbine ID string", item)
        if type(proxy) is not bool:
            self._error("region", "proxy", "bool", proxy)
        ids = tuple(sorted(members))
        if nest_id not in self._by_id or any(t not in self._by_id for t in ids):
            return self._result(
                {
                    "feasible": False,
                    "diagnostics": [{"kind": "invalid_node"}],
                    "task_count": max(0, len(ids) - 1),
                },
                "region",
            )
        if len(set(ids)) != len(ids) or nest_id not in ids:
            return self._result(
                {
                    "feasible": False,
                    "diagnostics": [
                        {"kind": "self_assignment_or_duplicates", "nest_id": nest_id}
                    ],
                    "task_count": max(0, len(ids) - 1),
                },
                "region",
            )
        key = (nest_id, ids, bool(proxy))
        if key in self._cache:
            self._hits += 1
            return self._result(self._cache[key], "region")
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
        return self._result(result, "region")

    def deployment(self, proposal: dict, *, proxy: bool = True) -> dict:
        self._check()
        self._proposal_schema("deployment", proposal)
        if type(proxy) is not bool:
            self._error("deployment", "proxy", "bool", proxy)
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
            return self._result(
                {
                    "feasible": False,
                    "diagnostics": [{"kind": "coverage_or_self_assignment"}],
                    "objectives": None,
                    "regions": [],
                },
                "deployment",
            )
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
        return self._result(
            {
                "feasible": not diagnostics,
                "diagnostics": diagnostics,
                "objectives": objectives,
                "regions": regions,
            },
            "deployment",
        )

    def remember(self, proposal: dict) -> bool:
        # Explicit candidate choice, no framework-generated fallback or hidden weighting.
        self._proposal_schema("remember", proposal)
        checked = self.deployment(proposal, proxy=False)
        if not checked["feasible"]:
            return False
        changed = (
            self._best is None or self._best["assignments"] != proposal["assignments"]
        )
        self._best = copy.deepcopy(dict(proposal))
        if (
            changed
            and self._last_repair_proposal is not None
            and proposal["assignments"] != self._last_repair_proposal["assignments"]
        ):
            self._changed_saves_after_repair += 1
        return True

    def best(self):
        return self._result(self._best, "best") if self._best is not None else None

    def split_repair(self, assignments: dict) -> dict:
        """Optional explicit primitive: open nests until every region passes audit.

        May increase K by up to N; not a nest minimizer. Does not assume NN cost
        decreases after each removal. An infeasible intermediate is allowed.
        """
        self._assignments_schema("split_repair", assignments)
        if set(assignments) != set(self._by_id) or not set(assignments.values()) <= set(
            self._by_id
        ):
            self._error(
                "split_repair",
                "assignments",
                "complete mapping to input turbine IDs",
                assignments,
                "repair_precondition",
            )
        self._repair_start("split_repair")
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
                return self._repair_done(
                    "split_repair", {"nest_ids": nests, "assignments": answer}
                )
        raise ValueError("Split repair failed to terminate within N openings.")

    def repartition_repair(self, assignments: dict) -> dict:
        """Repack only infeasible groups; bounded, optional feasibility primitive."""
        self._assignments_schema("repartition_repair", assignments)
        if set(assignments) != set(self._by_id) or not set(assignments.values()) <= set(
            self._by_id
        ):
            self._error(
                "repartition_repair",
                "assignments",
                "complete mapping to input turbine IDs",
                assignments,
                "repair_precondition",
            )
        self._repair_start("repartition_repair")
        answer = dict(assignments)
        for nest in set(answer.values()):
            answer[nest] = nest
        groups = {
            n: sorted(t for t in answer if answer[t] == n)
            for n in sorted(set(answer.values()))
        }
        for nest, members in groups.items():
            if self.region(nest, members, proxy=False)["feasible"]:
                continue
            pending = set(members)
            seed = nest
            opened = []
            while pending:
                self.tick()
                if not self.region(seed, [seed], proxy=False)["feasible"]:
                    raise ValueError(
                        "Even a singleton violates hover/endurance constraints."
                    )
                group = [seed]
                pending.remove(seed)
                answer[seed] = seed
                opened.append(seed)
                for tid in sorted(pending, key=lambda t: (self.distance(seed, t), t)):
                    self.tick()
                    if (
                        self.distance(seed, tid)
                        > effective_service_radius_km(self._config) + 1e-9
                    ):
                        continue
                    if self.region(seed, group + [tid], proxy=False)["feasible"]:
                        group.append(tid)
                        pending.remove(tid)
                        answer[tid] = seed
                if pending:
                    seed = max(
                        sorted(pending),
                        key=lambda t: (min(self.distance(t, n) for n in opened), t),
                    )
        return self._repair_done(
            "repartition_repair",
            {"nest_ids": sorted(set(answer.values())), "assignments": answer},
        )

    def _objective_vector(self, tool, field, value):
        valid = isinstance(value, (list, tuple)) and len(value) == 3
        if valid:
            try:
                valid = all(
                    not isinstance(x, bool)
                    and isinstance(x, (int, float))
                    and math.isfinite(x)
                    for x in value
                )
            except OverflowError:
                valid = False
        if not valid:
            self._error(tool, field, "three finite numeric objectives", value)
        return [float(x) for x in value]

    def objective_dominates(self, first: list, second: list) -> bool:
        self._check()
        a = self._objective_vector("objective_dominates", "first", first)
        b = self._objective_vector("objective_dominates", "second", second)
        return all(x <= y for x, y in zip(a, b)) and any(x < y for x, y in zip(a, b))

    def prefer_objectives(
        self, candidate: list, incumbent: list, *, weights: list, scales: list
    ) -> bool:
        self._check()
        a = self._objective_vector("prefer_objectives", "candidate", candidate)
        b = self._objective_vector("prefer_objectives", "incumbent", incumbent)
        w = self._objective_vector("prefer_objectives", "weights", weights)
        s = self._objective_vector("prefer_objectives", "scales", scales)
        if min(w) < 0 or sum(w) <= 0:
            self._error(
                "prefer_objectives", "weights", "nonnegative with positive sum", weights
            )
        if min(s) <= 0:
            self._error("prefer_objectives", "scales", "positive scales", scales)
        preference = {"weights": list(w), "scales": list(s)}
        if preference != self._last_preference:
            self._last_preference = preference
            if self._observer:
                self._observer({"kind": "objective_preference", **preference})
        if self.objective_dominates(a, b):
            return True
        if self.objective_dominates(b, a):
            return False
        left = sum(wi * ai / si for wi, ai, si in zip(w, a, s))
        right = sum(wi * bi / si for wi, bi, si in zip(w, b, s))
        if not math.isfinite(left) or not math.isfinite(right):
            self._error(
                "prefer_objectives",
                "normalized score",
                "finite weighted scores; choose safe weights/scales",
                left,
            )
        return left < right - 1e-12
