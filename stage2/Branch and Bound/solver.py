"""Directed edge-inclusion/exclusion branch and bound, without a MIP solver.

Assignment relaxations may contain several cycles. An include/exclude split on
one unfixed edge covers the parent's tours disjointly; forced proper cycles are
forbidden. Only relaxation bounds, never heuristic estimates, prune the tree.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Callable

import numpy as np
from scipy.optimize import linear_sum_assignment

from stage2.atsp import WindATSPInstance
from stage2.data import instance_fingerprint, instance_to_record


ALGORITHM_VERSION = "directed-assignment-bnb-v1"
TIMING_SCOPE = "Initial incumbent construction and B&B search/proof, including normal logging/checkpoint overhead; excludes input loading and stopped time."


def write_json_atomic(path: Path, payload: dict) -> None:
    """Replace only after a complete finite-valued JSON document is written."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                         encoding="utf-8")
    temporary.replace(path)


@dataclass(frozen=True, slots=True)
class Options:
    time_limit: float = 0.0
    log_interval: float = 10.0
    checkpoint_interval: float = 60.0
    tolerance: float = 1e-8

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            if not math.isfinite(value) or value < 0 or (name != "time_limit" and value == 0):
                raise ValueError(f"Invalid {name}: must be finite and {'nonnegative' if name == 'time_limit' else 'positive'}.")
        if self.tolerance > 1e-8:
            raise ValueError("Tolerance may be tightened, not loosened beyond 1e-8 seconds.")


@dataclass(frozen=True, slots=True)
class SearchNode:
    forced: tuple[int, ...]
    forbidden: tuple[tuple[int, int], ...] = ()
    bound: float = 0.0
    assignment: tuple[int, ...] = ()


class _TimeLimit(Exception):
    """Internal cooperative stop, never a proof of optimality."""


def _cycles(successors: tuple[int, ...]) -> list[tuple[int, ...]]:
    remaining = set(range(len(successors)))
    cycles = []
    while remaining:
        current = min(remaining)
        cycle = []
        while current in remaining:
            remaining.remove(current)
            cycle.append(current)
            current = successors[current]
        cycles.append(tuple(cycle))
    return sorted(cycles, key=lambda cycle: (len(cycle), cycle))


class BranchAndBound:
    """Deterministic, checkpointable ATSP search with depot index zero."""

    def __init__(
        self, costs: np.ndarray, budget: float, *, options: Options | None = None,
        checkpoint_path: Path | None = None, resume: bool = False,
        progress: Callable[[str], None] | None = print, label: str = "",
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        self.costs = np.asarray(costs, dtype=np.float64).copy()
        if (self.costs.ndim != 2 or self.costs.shape[0] == 0
                or self.costs.shape[0] != self.costs.shape[1]
                or np.isnan(self.costs).any() or np.any(self.costs < 0)):
            raise ValueError("Costs must be a nonempty square nonnegative matrix without NaN.")
        if not math.isfinite(budget):
            raise ValueError("Endurance flight budget must be finite.")
        # Diagonal entries are not travel legs; a zero-task tour has zero cost.
        np.fill_diagonal(self.costs, 0.0)
        self.n = len(self.costs)
        self.budget = float(budget)
        self.options = options or Options()
        self.checkpoint_path = checkpoint_path
        self.progress, self.label, self.clock = progress, label, clock
        signature = self.costs.astype("<f8").tobytes() + repr(self.budget).encode("ascii")
        self.signature = hashlib.sha256(signature).hexdigest()
        self.pending = [SearchNode((-1,) * self.n)]
        self.best_route: tuple[int, ...] | None = None
        self.best = math.inf
        self.closed_floor = math.inf
        self.expanded = self.pruned = 0
        self.initialized = False
        self.elapsed = 0.0
        self.started: float | None = None
        self.last_log = self.last_checkpoint = 0.0
        self.completed_result: dict | None = None
        if checkpoint_path is not None and checkpoint_path.exists():
            if not resume:
                raise FileExistsError(f"Checkpoint exists: {checkpoint_path}. Use --resume or a new output.")
            self._restore(json.loads(checkpoint_path.read_text(encoding="utf-8")))
        elif resume:
            raise FileNotFoundError(f"Resume checkpoint missing: {checkpoint_path}")

    def _margin(self, value: float) -> float:
        return min(self.options.tolerance, abs(value) * 1e-10)

    def _runtime(self) -> float:
        return self.elapsed + (self.clock() - self.started if self.started is not None else 0.0)

    def _lower_bound(self) -> float:
        return min(self.best, self.closed_floor, min((item.bound for item in self.pending), default=math.inf))

    def _emit(self, event: str) -> None:
        if self.progress is None:
            return
        lower = self._lower_bound()
        gap = (self.best - lower) / self.best * 100 if math.isfinite(self.best) and self.best > 0 else None
        best_text = f"{self.best:.9f}s" if math.isfinite(self.best) else "none"
        lower_text = f"{lower:.9f}s" if math.isfinite(lower) else "none"
        self.progress(
            f"{self.label} {event} | nodes={self.n} | elapsed={self._runtime():.3f}s | "
            f"expanded={self.expanded} | pending={len(self.pending)} | "
            f"best={best_text} | lower_bound={lower_text} | "
            f"gap={f'{gap:.6f}%' if gap is not None else 'N/A'}"
        )

    def _save(self, result: dict | None = None) -> None:
        if self.checkpoint_path is None:
            return
        write_json_atomic(self.checkpoint_path, {
            "schema_version": 1, "algorithm": ALGORITHM_VERSION, "matrix_fingerprint": self.signature,
            "tolerance": self.options.tolerance, "pending": [asdict(node) for node in self.pending],
            "best_route": self.best_route, "best": self.best if math.isfinite(self.best) else None,
            "closed_floor": self.closed_floor if math.isfinite(self.closed_floor) else None,
            "expanded": self.expanded, "pruned": self.pruned, "initialized": self.initialized,
            "runtime_seconds": self._runtime(), "result": result,
        })

    def _restore(self, payload: dict) -> None:
        if (payload.get("schema_version") != 1 or payload.get("algorithm") != ALGORITHM_VERSION
                or payload.get("matrix_fingerprint") != self.signature
                or payload.get("tolerance") != self.options.tolerance):
            raise ValueError("Checkpoint input, environment, algorithm version or tolerance differs.")
        self.pending = [SearchNode(tuple(node["forced"]), tuple(tuple(edge) for edge in node["forbidden"]),
                                   float(node["bound"]), tuple(node["assignment"])) for node in payload["pending"]]
        self.best_route = tuple(payload["best_route"]) if payload["best_route"] is not None else None
        self.best = float(payload["best"]) if payload["best"] is not None else math.inf
        self.closed_floor = float(payload["closed_floor"]) if payload["closed_floor"] is not None else math.inf
        self.expanded, self.pruned = int(payload["expanded"]), int(payload["pruned"])
        self.initialized, self.elapsed = bool(payload["initialized"]), float(payload["runtime_seconds"])
        if self.best_route is not None:
            self._validate_route(self.best_route)
            if self._cost(self.best_route) != self.best or self.best > self.budget:
                raise ValueError("Checkpoint incumbent route/cost is inconsistent.")
        result = payload.get("result")
        if result and result["termination"] in ("OPTIMAL", "INFEASIBLE"):
            self.completed_result = result

    def _poll(self) -> None:
        now = self.clock()
        if now - self.last_log >= self.options.log_interval:
            self._emit("PROGRESS")
            self.last_log = now
        if now - self.last_checkpoint >= self.options.checkpoint_interval:
            self._save()
            self.last_checkpoint = now
        if self.options.time_limit and now - self.started >= self.options.time_limit:
            raise _TimeLimit

    def _cost(self, route: tuple[int, ...]) -> float:
        return float(sum(self.costs[i, j] for i, j in zip(route, route[1:])))

    def _validate_route(self, route: tuple[int, ...]) -> None:
        if (len(route) != self.n + 1 or route[0] != 0 or route[-1] != 0
                or set(route[:-1]) != set(range(self.n))):
            raise ValueError("Incumbent is not a complete depot-to-depot tour.")

    def _accept(self, route: tuple[int, ...]) -> None:
        cost = self._cost(route)
        if cost <= self.budget and cost < self.best:
            self.best_route, self.best = route, cost
            self._emit("NEW_INCUMBENT")

    def _initial_routes(self) -> None:
        if self.n == 1:
            self._accept((0, 0))
            return
        for first in range(1, self.n):
            self._poll()
            route = [0, first]
            unused = set(range(1, self.n)) - {first}
            while unused:
                following = min(unused, key=lambda j: (self.costs[route[-1], j], j))
                route.append(following)
                unused.remove(following)
            route.append(0)
            current = tuple(route)
            self._accept(current)
            # Relocation recomputes the full directed cost, not a symmetric 2-opt delta.
            while True:
                self._poll()
                improved = False
                current_cost = self._cost(current)
                for source in range(1, self.n):
                    rest = list(current[:-1])
                    node = rest.pop(source)
                    for target in range(1, self.n):
                        candidate = tuple(rest[:target] + [node] + rest[target:] + [0])
                        cost = self._cost(candidate)
                        improves = (cost < current_cost - self._margin(current_cost)
                                    if math.isfinite(current_cost) else math.isfinite(cost))
                        if improves:
                            current = candidate
                            self._accept(current)
                            improved = True
                            break
                    if improved:
                        break
                if not improved:
                    break

    def relax(self, node: SearchNode) -> SearchNode | None:
        """Return a guarded assignment bound, or reject an impossible branch."""
        if self.n == 1:
            return SearchNode(node.forced, node.forbidden, 0.0, (0,))
        forced = node.forced
        targets = [j for j in forced if j >= 0]
        if len(set(targets)) != len(targets) or len(forced) != self.n:
            return None
        matrix = self.costs.copy()
        np.fill_diagonal(matrix, np.inf)
        for i, j in node.forbidden:
            matrix[i, j] = np.inf
        predecessor = [-1] * self.n
        for i, j in enumerate(forced):
            if j >= 0:
                if j >= self.n or i == j or not math.isfinite(matrix[i, j]):
                    return None
                predecessor[j] = i
        # Detect closed forced components; a proper cycle can never be extended.
        for start in range(self.n):
            seen = set()
            current = start
            while current >= 0 and current not in seen:
                seen.add(current)
                current = forced[current]
            if current >= 0 and len(seen) < self.n:
                return None
        for i, j in enumerate(forced):
            if j >= 0:
                matrix[i, :] = np.inf
                matrix[:, j] = np.inf
                matrix[i, j] = self.costs[i, j]
        # An open forced path may only close once it contains every node.
        for head in range(self.n):
            if predecessor[head] >= 0:
                continue
            tail, length = head, 1
            while forced[tail] >= 0:
                tail = forced[tail]
                length += 1
            if length < self.n:
                matrix[tail, head] = np.inf
        try:
            rows, columns = linear_sum_assignment(matrix)
        except ValueError as exc:
            if "infeasible" not in str(exc).lower():
                raise
            return None
        value = math.fsum(float(self.costs[i, j]) for i, j in zip(rows, columns))
        bound = max(node.bound, 0.0, value - self._margin(value))
        return SearchNode(forced, node.forbidden, bound, tuple(int(j) for j in columns))

    def _prune(self, node: SearchNode) -> bool:
        if node.bound > self.budget:
            self.pruned += 1
            return True
        if math.isfinite(self.best) and node.bound >= self.best - self._margin(self.best):
            self.closed_floor = min(self.closed_floor, node.bound)
            self.pruned += 1
            return True
        return False

    def _expand(self, original: SearchNode) -> list[SearchNode]:
        node = original if original.assignment else self.relax(original)
        if node is None or self._prune(node):
            return []
        cycles = _cycles(node.assignment)
        if len(cycles) == 1:
            route = [0]
            for _ in range(self.n):
                route.append(node.assignment[route[-1]])
            if self._cost(tuple(route)) <= self.budget:
                self._accept(tuple(route))
                self.closed_floor = min(self.closed_floor, node.bound)
                return []
        # Usually a proper subtour; a near-budget infeasible complete assignment
        # is also split rather than accepted using a relaxed endurance tolerance.
        cycle = cycles[0]
        free = [(i, node.assignment[i]) for i in cycle if node.forced[i] < 0]
        if not free:
            return []
        i, j = min(free)
        forced = list(node.forced)
        forced[i] = j
        candidates = [SearchNode(tuple(forced), node.forbidden, node.bound),
                      SearchNode(node.forced, (*node.forbidden, (i, j)), node.bound)]
        children = []
        for candidate in candidates:
            child = self.relax(candidate)
            if child is not None and not self._prune(child):
                children.append(child)
        # Stack pop visits the lower bound first; stable ties are deterministic.
        return sorted(children, key=lambda child: child.bound, reverse=True)

    def _result(self, termination: str) -> dict:
        lower = self._lower_bound()
        gap = max(0.0, (self.best - lower) / self.best) if math.isfinite(self.best) and self.best > 0 else None
        if self.best == 0:
            gap = 0.0
        optimal = termination == "OPTIMAL"
        return {
            "algorithm": ALGORITHM_VERSION, "matrix_fingerprint": self.signature,
            "termination": termination, "status": {"OPTIMAL": 2, "INFEASIBLE": 3, "TIME_LIMIT": 9,
                                                      "INTERRUPTED": 11, "MEMORY_LIMIT": 11}[termination],
            "status_note": "Numeric status codes retain the existing Stage2 reference-file convention; this is not Gurobi.",
            "proven_optimal": optimal, "optimal_within_solver_tolerance": optimal,
            "feasible": self.best_route is not None, "route": list(self.best_route) if self.best_route is not None else None,
            "flight_seconds": self.best if math.isfinite(self.best) else None,
            "objective_bound_seconds": lower if math.isfinite(lower) else None, "mip_gap": gap,
            "absolute_tolerance_seconds": self.options.tolerance, "runtime_seconds": self._runtime(),
            "timing_scope": TIMING_SCOPE, "expanded_nodes": self.expanded,
            "pruned_nodes": self.pruned, "pending_nodes": len(self.pending),
            "proof_rule": "Complete include/exclude search with assignment bounds; abs <= 1e-8 s and relative <= 1e-10 numerical pruning margins, no cost rounding.",
        }

    def solve(self) -> dict:
        """Search until proof or an explicit interruption; preserve every branch."""
        if self.completed_result is not None:
            return self.completed_result
        self.started = self.clock()
        self.last_log = self.last_checkpoint = self.started
        try:
            self._emit("START")
            if not self.initialized:
                self._initial_routes()
                self.initialized = True
                self._save()
            while self.pending:
                self._poll()
                # Keep the parent pending until BOTH children have been bounded.
                # A signal in _expand leaves the entire parent available on resume.
                children = self._expand(self.pending[-1])
                self.pending = self.pending[:-1] + children
                self.expanded += 1
            termination = "OPTIMAL" if self.best_route is not None else "INFEASIBLE"
        except _TimeLimit:
            termination = "TIME_LIMIT"
        except KeyboardInterrupt:
            termination = "INTERRUPTED"
        except MemoryError:
            termination = "MEMORY_LIMIT"
        # A Python signal can land between the two attribute stores in _accept.
        # Recompute before persisting, so an interrupted incumbent stays coherent.
        if self.best_route is not None:
            self.best = self._cost(self.best_route)
        result = self._result(termination)
        self.elapsed = result["runtime_seconds"]
        self.started = None
        self._save(result)
        self._emit(termination)
        return result


def solve_instance(
    instance: WindATSPInstance, *, options: Options | None = None,
    checkpoint_path: Path | None = None, resume: bool = False,
    progress: Callable[[str], None] | None = print, label: str = "",
) -> dict:
    """Adapt the unchanged Stage2 environment and arbitrary depot to the solver."""
    order = [instance.depot_index, *(i for i in range(instance.node_count) if i != instance.depot_index)]
    matrix = instance.travel_time_seconds[np.ix_(order, order)]
    budget = instance.config.max_endurance_seconds - (instance.node_count - 1) * instance.config.hover_equivalent_seconds
    result = BranchAndBound(matrix, budget, options=options, checkpoint_path=checkpoint_path,
                            resume=resume, progress=progress, label=label).solve()
    result = {**result, "instance_fingerprint": instance_fingerprint(instance), "instance": instance_to_record(instance),
              "nest_id": instance.identifiers[instance.depot_index], "node_count": instance.node_count,
              "equivalent_energy_seconds": None, "route_distance_km": None}
    if result["route"] is not None:
        route = tuple(order[i] for i in result["route"])
        measured = instance.evaluate_route(route)
        if not measured.feasible:
            raise ValueError("Solver result fails the original Stage2 endurance check.")
        result.update(route=list(route), flight_seconds=measured.flight_seconds,
                      equivalent_energy_seconds=measured.equivalent_energy_seconds,
                      route_distance_km=float(sum(np.linalg.norm(instance.coordinates_km[i] - instance.coordinates_km[j])
                                                  for i, j in zip(route, route[1:]))))
    return result
