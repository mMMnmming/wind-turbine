"""Tool contract v5: explicit structural moves and frozen wind-aware routing."""

from __future__ import annotations

import copy
import inspect
import math
from collections import Counter

from .candidate_tools import BudgetExceeded, CandidateTools
from .candidate_tools import ToolContractError as ToolContractError
from .domain import DeploymentConfig, Turbine
from .joint_evidence import SearchEvidence
from .joint_moves import transform, validate

TOOLS_VERSION = 5
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
    "repartition_repair",
    "stats",
    "make_proposal",
    "assignments_from_proposal",
    "better_metrics",
    "reassign_tasks",
    "remove_nest",
    "relocate_nest",
    "merge_nests",
)


def ranking_key(values) -> tuple:
    if (
        not isinstance(values, (list, tuple))
        or len(values) != 3
        or any(
            isinstance(v, bool)
            or not isinstance(v, (int, float))
            or not math.isfinite(v)
            or v < 0
            for v in values
        )
    ):
        raise ValueError(
            "Ranking requires three finite nonnegative values [nests, total flight, maximum flight]."
        )
    return tuple(values)


def aggregate_metrics(datasets: list[dict]) -> dict:
    if not datasets or not all(d["feasible"] for d in datasets):
        raise ValueError("Only complete feasible dataset records may be ranked.")
    metrics = [d["metrics"] for d in datasets]
    return {
        "mean_nest_count": sum(m["nest_count"] for m in metrics) / len(metrics),
        "mean_total_flight_seconds": sum(m["total_flight_seconds"] for m in metrics)
        / len(metrics),
        "mean_region_flight_seconds": sum(m["mean_flight_seconds"] for m in metrics)
        / len(metrics),
        "maximum_flight_seconds": max(m["maximum_flight_seconds"] for m in metrics),
    }


class JointTools(CandidateTools):
    """Reuse strict ID/contract bookkeeping, but never legacy route evaluation."""

    def __init__(
        self, points: list[Turbine], config: DeploymentConfig, *, planner, **kwargs
    ):
        super().__init__(points, config, **kwargs)
        import numpy as np

        from stage2.atsp import WindATSPConfig, WindATSPInstance

        self._planner = planner
        self._requests = 0
        self._evidence = SearchEvidence()
        self._lifetime_post_repair_calls = 0
        self._region_failure_counts = Counter()
        self._operation_counts = Counter()
        self._times = WindATSPInstance(
            tuple(p.identifier for p in self.points),
            np.asarray([(p.x_km, p.y_km) for p in self.points]),
            0,
            WindATSPConfig(**planner.metadata["environment"]),
        ).travel_time_seconds

    def _region(self, nest_id: str, members) -> dict:
        self._check()
        self._query_id("region", "nest_id", nest_id)
        if not isinstance(members, (list, tuple, set, frozenset)):
            self._error(
                "region",
                "members",
                "sequence/set of ID strings including nest",
                members,
            )
        for member in members:
            if not isinstance(member, str):
                self._error("region", "member", "ID string", member)
        ids = sorted(set(members))
        self._requests += 1
        if (
            len(ids) != len(members)
            or nest_id not in ids
            or not set(ids) <= set(self._by_id)
        ):
            return self._result(
                {
                    "feasible": False,
                    "diagnostics": [{"kind": "invalid_region_members"}],
                    "task_count": max(0, len(ids) - 1),
                },
                "region",
            )
        key = (nest_id, tuple(ids))
        if key in self._cache:
            self._hits += 1
            return self._result(copy.deepcopy(self._cache[key]), "region")
        radius = max(self.distance(nest_id, tid) for tid in ids)
        if radius > self._config.communication_radius_km + 1e-9:
            result = {
                "feasible": False,
                "diagnostics": [
                    {
                        "kind": "service_radius",
                        "actual": radius,
                        "limit": self._config.communication_radius_km,
                        "excess": radius - self._config.communication_radius_km,
                    }
                ],
                "task_count": len(ids) - 1,
                "radius_km": radius,
            }
        else:
            if self._evaluations >= self._region_limit:
                raise BudgetExceeded(
                    "uncached network region evaluation budget exhausted"
                )
            result = self._planner.region(
                [self.points[self._by_id[i]] for i in ids], nest_id
            )
            # Blocking inference time belongs to the candidate's soft budget too.
            self._check()
            hit = result.pop("cache_hit", False)
            if hit:
                self._hits += 1
            else:
                self._evaluations += 1
            if self._observer and not hit:
                self._observer(
                    {
                        "kind": "network_region",
                        "nest_id": nest_id,
                        **{
                            k: result.get(k)
                            for k in (
                                "task_count",
                                "feasible",
                                "flight_seconds",
                                "energy_seconds",
                                "endurance_margin_seconds",
                                "runtime_seconds",
                            )
                        },
                    }
                )
        self._cache[key] = copy.deepcopy(result)
        return self._result(result, "region")

    def region(self, nest_id: str, members) -> dict:
        checked = self._region(nest_id, members)
        for diagnostic in checked["diagnostics"]:
            self._region_failure_counts[diagnostic["kind"]] += 1
        return checked

    def _deployment(self, proposal: dict) -> dict:
        self._check()
        self._proposal_schema("deployment", proposal)
        nests, assignments = proposal["nest_ids"], proposal["assignments"]
        valid = (
            nests
            and len(set(nests)) == len(nests)
            and set(nests) <= set(self._by_id)
            and set(assignments) == set(self._by_id)
            and set(assignments.values()) <= set(nests)
            and all(assignments.get(n) == n for n in nests)
        )
        if not valid:
            return self._result(
                {
                    "feasible": False,
                    "diagnostics": [
                        {
                            "kind": "coverage_or_self_assignment",
                            "missing_ids": sorted(set(self._by_id) - set(assignments))[
                                :5
                            ],
                            "extra_ids": sorted(set(assignments) - set(self._by_id))[
                                :5
                            ],
                            "invalid_targets": sorted(
                                set(assignments.values()) - set(nests)
                            )[:5],
                            "invalid_self_assignments": sorted(
                                n for n in nests if assignments.get(n) != n
                            )[:5],
                        }
                    ],
                    "regions": [],
                    "metrics": None,
                    "objectives": None,
                },
                "deployment",
            )
        regions = [
            self.region(n, [t for t, owner in assignments.items() if owner == n])
            for n in sorted(nests)
        ]
        diagnostics = [
            dict(d, nest_id=n)
            for n, r in zip(sorted(nests), regions)
            for d in r["diagnostics"]
        ]
        complete_routes = all("flight_seconds" in r for r in regions)
        total = sum(r["flight_seconds"] for r in regions) if complete_routes else None
        metrics = {
            "nest_count": len(nests),
            "total_flight_seconds": total,
            "mean_flight_seconds": total / len(nests) if total is not None else None,
            "maximum_flight_seconds": (
                max(r["flight_seconds"] for r in regions) if complete_routes else None
            ),
            "minimum_endurance_margin_seconds": min(
                (r.get("endurance_margin_seconds", 0) for r in regions)
            ),
            "region_task_counts": [r["task_count"] for r in regions],
            "network_runtime_seconds": sum(
                r.get("runtime_seconds", 0) for r in regions
            ),
        }
        feasible = not diagnostics
        objectives = (
            [len(nests), total, metrics["maximum_flight_seconds"]] if feasible else None
        )
        return self._result(
            {
                "feasible": feasible,
                "diagnostics": diagnostics,
                "regions": regions,
                "metrics": metrics,
                "objectives": objectives,
            },
            "deployment",
        )

    def deployment(self, proposal: dict) -> dict:
        checked = self._deployment(proposal)
        self._evidence.observe(proposal, checked)
        return checked

    def _move(self, operation, assignments, **params):
        self.tick()
        self._operation_counts[operation] += 1
        try:
            result = transform(assignments, self._by_id, operation, **params)
        except (ValueError, TypeError) as error:
            self._error(operation, "assignments/updates", str(error), assignments)
        self._evidence.observe_move(assignments, result["assignments"])
        return self._result(result, operation)

    def reassign_tasks(self, assignments: dict, updates: dict) -> dict:
        return self._move("reassign_tasks", assignments, updates=updates)

    def remove_nest(self, assignments: dict, nest_id: str, updates: dict) -> dict:
        return self._move("remove_nest", assignments, source=nest_id, updates=updates)

    def relocate_nest(self, assignments: dict, old_nest: str, new_nest: str) -> dict:
        return self._move(
            "relocate_nest", assignments, source=old_nest, target=new_nest
        )

    def merge_nests(
        self, assignments: dict, source_nest: str, target_nest: str
    ) -> dict:
        return self._move(
            "merge_nests", assignments, source=source_nest, target=target_nest
        )

    def _record_external_call(self, name):
        super()._record_external_call(name)
        if self._last_repair_proposal is not None and name in ("region", "deployment"):
            self._lifetime_post_repair_calls += 1

    def finalize_return(self, proposal):
        # No additional inference or budget-consuming audit at worker shutdown.
        self._evidence.mark(proposal, returned=True)

    def remember(self, proposal: dict) -> bool:
        self._proposal_schema("remember", proposal)
        if not self.deployment(proposal)["feasible"]:
            return False
        changed = (
            self._best is None or self._best["assignments"] != proposal["assignments"]
        )
        self._best = copy.deepcopy(dict(proposal))
        self._evidence.mark(proposal)
        if (
            changed
            and self._last_repair_proposal is not None
            and proposal["assignments"] != self._last_repair_proposal["assignments"]
        ):
            self._changed_saves_after_repair += 1
        return True

    def better_metrics(self, candidate: list, incumbent: list) -> bool:
        self._check()
        for name, values in (("candidate", candidate), ("incumbent", incumbent)):
            try:
                ranking_key(values)
            except (ValueError, OverflowError):
                self._error(
                    "better_metrics",
                    name,
                    "finite nonnegative [nest count, total seconds, max seconds]",
                    values,
                )
        return tuple(candidate) < tuple(incumbent)

    def stats(self) -> dict:
        result = dict(super().stats())
        result.update(
            tools_version=TOOLS_VERSION,
            region_requests=self._requests,
            search_evidence=self._evidence.snapshot(),
            region_failure_counts=dict(self._region_failure_counts),
            structural_move_attempts=dict(self._operation_counts),
            lifetime_post_repair_evaluation_calls=self._lifetime_post_repair_calls,
        )
        return self._result(result, "stats")

    def _repair(self, assignments: dict, grouped: bool):
        name = "repartition_repair" if grouped else "split_repair"
        self._assignments_schema(name, assignments)
        if set(assignments) != set(self._by_id) or not set(assignments.values()) <= set(
            self._by_id
        ):
            try:
                validate(assignments, self._by_id)
            except ValueError as error:
                self._error(name, "assignments", str(error), assignments)
        self._repair_start(name)
        answer = dict(assignments)
        for nest in set(answer.values()):
            answer[nest] = nest
        for _ in range(len(answer) + 1):
            self.tick()
            groups = {
                n: sorted(t for t in answer if answer[t] == n)
                for n in sorted(set(answer.values()))
            }
            for nest, members in groups.items():
                if self.region(nest, members)["feasible"]:
                    continue
                tasks = [m for m in members if m != nest]
                if not tasks:
                    raise ValueError("Singleton could not be evaluated as feasible.")
                if not grouped:
                    opened = max(tasks, key=lambda t: (self.distance(nest, t), t))
                    answer[opened] = opened
                    break
                pending = set(members)
                seed = nest
                while pending:
                    self.tick()
                    group = [seed]
                    pending.remove(seed)
                    answer[seed] = seed
                    for tid in sorted(
                        pending, key=lambda t: (self.distance(seed, t), t)
                    ):
                        self.tick()
                        if self.region(seed, group + [tid])["feasible"]:
                            group.append(tid)
                            pending.remove(tid)
                            answer[tid] = seed
                    if pending:
                        seed = min(pending)
                break
            else:
                return self._repair_done(
                    name,
                    {"nest_ids": sorted(set(answer.values())), "assignments": answer},
                )
        raise ValueError("Repair did not terminate within N openings.")

    def split_repair(self, assignments: dict) -> dict:
        return self._repair(assignments, False)

    def repartition_repair(self, assignments: dict) -> dict:
        return self._repair(assignments, True)


def tool_reference() -> str:
    notes = {
        "region": "members含机巢ID；返回feasible,diagnostics,route_ids,flight_seconds,energy_seconds,endurance_margin_seconds。无proxy参数，全部为冻结网络贪心路线。",
        "deployment": "输入完整proposal；返回feasible,diagnostics,regions,metrics,objectives。可行时objectives=[机巢数,全场总飞行秒,最长区域飞行秒]；否则None。",
        "better_metrics": "输入两个deployment的objectives列表，按机巢数→总秒→最长秒严格优先比较；同分False。",
        "remember": "只保存显式选择且网络路线可行的proposal；不自动选优。",
        "best": "返回最后显式保存的proposal深拷贝或None，不代表精确最优。",
        "distance": "输入两个风机ID字符串，返回km，不是飞行时间。",
        "directed_time": "输入两个ID字符串，返回与Stage2一致的含风有向秒。",
        "remaining": "返回seconds,region_evaluations,move_attempts字典。",
        "split_repair": "完整assignments映射→完整proposal，显式新增单点机巢修复，不自动保存。",
        "repartition_repair": "完整assignments映射→完整proposal，以网络贪心可行性重组违规区域，不自动保存/优化。",
        "reassign_tasks": "完整映射+非机巢任务updates→{assignments,affected_nests}，保持巢数；结构操作不审核物理约束。",
        "remove_nest": "完整映射+源巢+其全部成员(含旧巢风机)的新归属updates→{assignments,affected_nests}；目标为剩余巢。",
        "relocate_nest": "完整映射+旧巢+非机巢新位置→{assignments,affected_nests}，保留旧巢任务，含新位置原归属区域。",
        "merge_nests": "完整映射+源巢+目标巢→{assignments,affected_nests}，源巢风机继续作为任务。",
    }
    return "\n".join(
        f"{name}{str(inspect.signature(getattr(JointTools, name))).replace('self, ', '')}: {notes.get(name, '沿用严格工具契约；输入ID必须为字符串。')}"
        for name in PUBLIC_METHODS
    )
