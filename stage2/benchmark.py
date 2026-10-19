"""Optional offline directed-ATSP references using the Gurobi command line.

No Gurobi Python dependency or solver call is required by RL training.
The MTZ model optimizes the very same flight-time matrix and endurance proxy.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
from statistics import mean
import subprocess

import numpy as np

from .atsp import WindATSPInstance
from .data import instance_fingerprint, load_region_instances, split_fingerprint


def build_atsp_lp(instance: WindATSPInstance) -> str:
    """Export directed degree, MTZ subtour, and complete-tour endurance constraints."""
    n = instance.node_count
    if n < 2:
        raise ValueError("An ATSP benchmark needs a depot and at least one client.")
    times = instance.travel_time_seconds
    edges = [(i, j) for i in range(n) for j in range(n) if i != j]
    clients = [i for i in range(n) if i != instance.depot_index]
    # Split long expressions onto continuation lines for portable LP parsing.
    cost = "\n    + ".join(f"{times[i, j]:.17g} x_{i}_{j}" for i, j in edges)
    lines = ["Minimize", f" flight: {cost}", "Subject To"]
    for i in range(n):
        lines.append(f" out_{i}: " + " + ".join(f"x_{i}_{j}" for j in range(n) if j != i) + " = 1")
        lines.append(f" in_{i}: " + " + ".join(f"x_{j}_{i}" for j in range(n) if j != i) + " = 1")
    for i in clients:
        for j in clients:
            if i != j:
                lines.append(f" mtz_{i}_{j}: u_{i} - u_{j} + {n - 1} x_{i}_{j} <= {n - 2}")
    budget = instance.config.max_endurance_seconds - len(clients) * instance.config.hover_equivalent_seconds
    lines.extend([f" endurance: {cost} <= {budget:.17g}", "Bounds"])
    lines.extend(f" 1 <= u_{i} <= {n - 1}" for i in clients)
    lines.append("Binary")
    lines.extend(f" x_{i}_{j}" for i, j in edges)
    lines.extend(["End", ""])
    return "\n".join(lines)


def _finite(value: object) -> float | None:
    if value is None:
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _closed_bound(objective: float, bound: object, gap: object) -> bool:
    """Require a closed objective bound, beyond merely OPTIMAL status."""
    bound_value, gap_value = _finite(bound), _finite(gap)
    return (bound_value is not None and gap_value is not None and gap_value <= 1e-8
            and math.isclose(objective, bound_value, rel_tol=1e-9, abs_tol=1e-6))


def parse_gurobi_solution(instance: WindATSPInstance, solution: dict) -> dict[str, object]:
    """Validate incumbent tours and never promote a time limit to optimality."""
    info = solution["SolutionInfo"]
    status = int(info["Status"])
    result: dict[str, object] = {
        "instance_fingerprint": instance_fingerprint(instance), "node_count": instance.node_count,
        "status": status, "proven_optimal": False, "optimal_within_solver_tolerance": status == 2,
        "runtime_seconds": _finite(info.get("Runtime")),
        "objective_bound_seconds": _finite(info.get("ObjBound")), "mip_gap": _finite(info.get("MIPGap")),
        "flight_seconds": None, "equivalent_energy_seconds": None, "route": None, "feasible": False,
    }
    if "ObjVal" not in info:
        if status == 2:
            raise ValueError("OPTIMAL status has no objective or tour.")
        return result
    edges = []
    for variable in solution.get("Vars", []):
        name = variable.get("VarName", "")
        if name.startswith("x_") and float(variable.get("X", 0.0)) > 0.5:
            _, first, second = name.split("_")
            edges.append((int(first), int(second)))
    successors = dict(edges)
    if (len(edges) != instance.node_count or len(successors) != instance.node_count
            or set(successors) != set(range(instance.node_count))
            or {j for _, j in edges} != set(range(instance.node_count))):
        raise ValueError("Solver output is not one complete tour.")
    route = [instance.depot_index]
    for _ in range(instance.node_count):
        route.append(successors[route[-1]])
    metrics = instance.evaluate_route(tuple(route))
    if not metrics.feasible:
        raise ValueError("Solver tour violates the environment's endurance constraint.")
    if not math.isclose(metrics.flight_seconds, float(info["ObjVal"]), rel_tol=1e-6, abs_tol=1e-6):
        raise ValueError("Solver objective differs from the environment tour cost.")
    result.update({
        "proven_optimal": status == 2 and _closed_bound(metrics.flight_seconds, info.get("ObjBound"), info.get("MIPGap")),
        "flight_seconds": metrics.flight_seconds,
        "equivalent_energy_seconds": metrics.equivalent_energy_seconds,
        "route": list(metrics.route), "feasible": True,
    })
    return result


def select_benchmark_instances(instances: list[WindATSPInstance], limit: int) -> list[WindATSPInstance]:
    """Choose fixed node-count quantiles, not the easiest solver outcomes."""
    if limit < 1 or not instances:
        raise ValueError("A positive limit and nonempty benchmark split are required.")
    ordered = sorted(instances, key=lambda item: (item.node_count, instance_fingerprint(item)))
    indices = np.linspace(0, len(ordered) - 1, min(limit, len(ordered)), dtype=int)
    return [ordered[int(index)] for index in indices]


def load_references(path: Path, instances: list[WindATSPInstance]) -> dict[str, dict]:
    """Only attach references belonging to the exact geometry and environment."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise ValueError("Unsupported reference-results schema.")
    allowed = {instance_fingerprint(item): item for item in instances}
    references = {}
    for record in payload["results"]:
        key = record["instance_fingerprint"]
        if key not in allowed:
            raise ValueError("Reference results contain problems outside the requested validation/test split.")
        if key in references:
            raise ValueError("Duplicate reference problem.")
        if record.get("route") is not None:
            measured = allowed[key].evaluate_route(tuple(record["route"]))
            if not measured.feasible or not math.isclose(measured.flight_seconds, record["flight_seconds"], rel_tol=1e-8):
                raise ValueError("Cached reference route/cost is inconsistent.")
        if record.get("proven_optimal") and (record["status"] != 2 or not record.get("feasible")):
            raise ValueError("Cached reference falsely declares optimality.")
        if record.get("proven_optimal") and not _closed_bound(
            record["flight_seconds"], record.get("objective_bound_seconds"), record.get("mip_gap"),
        ):
            raise ValueError("Cached reference claims optimality without a closed objective bound.")
        references[key] = record
    return references


def reference_summary(
    predictions: list[tuple[WindATSPInstance, float, bool]], references: dict[str, dict],
) -> dict[str, float]:
    """Report true-optimum gaps separately from best-known reference gaps."""
    optimal_gaps, known_gaps = [], []
    matched = 0
    for instance, flight, feasible in predictions:
        record = references.get(instance_fingerprint(instance))
        if record is None:
            continue
        matched += 1
        objective = record.get("flight_seconds")
        if not feasible or not record.get("feasible") or objective is None or objective <= 0:
            continue
        gap = (flight - objective) / objective
        if record.get("proven_optimal"):
            if gap < -1e-5:
                raise ValueError("A model route beats a claimed optimum; check the reference model/environment.")
            optimal_gaps.append(max(0.0, gap))
        else:
            known_gaps.append(gap)
    result = {"reference_matched_count": float(matched), "optimality_gap_count": float(len(optimal_gaps)),
              "best_known_gap_count": float(len(known_gaps))}
    if optimal_gaps:
        result["mean_optimality_gap"] = mean(optimal_gaps)
    if known_gaps:
        result["mean_best_known_gap"] = mean(known_gaps)
    return result


def solve_with_gurobi(
    instance: WindATSPInstance, output: Path, command: str, time_limit: float,
    threads: int, seed: int, mip_gap: float,
) -> dict[str, object]:
    """Use the installed CLI, avoiding a mandatory gurobipy dependency."""
    output.mkdir(parents=True, exist_ok=True)
    model_path = (output / "model.lp").resolve()
    result_path = (output / "solution.json").resolve()
    if any((output / name).exists() for name in ("model.lp", "solution.json", "solver.log", "solver_console.log")):
        raise ValueError(f"Existing solver artifacts in {output}. Use a fresh benchmark directory; nothing was overwritten.")
    model_path.write_text(build_atsp_lp(instance), encoding="utf-8")
    invocation = [command, f"TimeLimit={time_limit}", f"Threads={threads}", f"Seed={seed}",
                  f"MIPGap={mip_gap}", "JSONSolDetail=1", f"ResultFile={result_path}",
                  f"LogFile={(output / 'solver.log').resolve()}", str(model_path)]
    try:
        completed = subprocess.run(invocation, cwd=output.resolve(), capture_output=True,
                                   timeout=time_limit + 120.0, check=False)
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"Gurobi exceeded its process timeout; inspect {output}.") from exc
    (output / "solver_console.log").write_bytes(completed.stdout + completed.stderr)
    if completed.returncode or not result_path.exists():
        if b"license expired" in (completed.stdout + completed.stderr).lower():
            raise RuntimeError(f"Gurobi license expired; RL training is unaffected. See {output / 'solver_console.log'}.")
        raise RuntimeError(f"Gurobi failed or wrote no JSON result. Check licensing and {output / 'solver_console.log'}.")
    result = parse_gurobi_solution(instance, json.loads(result_path.read_text(encoding="utf-8")))
    result["model_sha256"] = hashlib.sha256(model_path.read_bytes()).hexdigest()
    version = re.search(rb"Gurobi Optimizer version ([^\r\n]+)", completed.stdout)
    result["solver_version"] = version.group(1).decode("ascii", errors="replace") if version else None
    result["proof_rule"] = "OPTIMAL, reported gap <= 1e-8, bound/cost close at rel=1e-9 and abs=1e-6."
    # The CLI's log retains the actual optimizer version, tolerances, and model statistics.
    result["solver_log"] = str(output / "solver.log")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--regions-json", type=Path, required=True)
    parser.add_argument("--stage1-config", type=Path, default=Path(__file__).resolve().parents[1] / "stage1/config.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-instances", type=int, default=32)
    parser.add_argument("--time-limit", type=float, default=120.0)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--mip-gap", type=float, default=0.0)
    parser.add_argument("--gurobi-command", default="gurobi_cl")
    args = parser.parse_args()
    if (not math.isfinite(args.time_limit) or args.time_limit <= 0 or args.threads < 1
            or not math.isfinite(args.mip_gap) or args.mip_gap < 0 or not 0 <= args.seed <= 2_000_000_000):
        parser.error("Solver time/threads must be positive; gap and seed must be valid and nonnegative.")
    command = shutil.which(args.gurobi_command)
    if command is None:
        parser.error("Gurobi CLI not found. Set --gurobi-command to its executable; RL training does not need it.")
    instances = load_region_instances(args.regions_json, args.stage1_config)
    selected = select_benchmark_instances(instances, args.max_instances)
    settings = {"time_limit": args.time_limit, "threads": args.threads, "seed": args.seed, "mip_gap": args.mip_gap,
                "selection": "node_count_quantiles", "selected": [instance_fingerprint(item) for item in selected],
                "split_fingerprint": split_fingerprint(instances)}
    args.output.mkdir(parents=True, exist_ok=True)
    report_path = args.output / "reference_results.json"
    if report_path.exists():
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if report.get("settings") != settings:
            raise ValueError("Existing benchmark settings differ. Use a new output directory.")
        load_references(report_path, instances)
    else:
        report = {"schema_version": 1, "settings": settings, "results": [],
                  "scope": "Fixed-deployment regional directed ATSP, not joint deployment/routing optimality."}
    completed_keys = {record["instance_fingerprint"] for record in report["results"]}
    for index, instance in enumerate(selected, start=1):
        key = instance_fingerprint(instance)
        if key in completed_keys:
            continue
        result = solve_with_gurobi(instance, args.output / key[:16], command,
                                   args.time_limit, args.threads, args.seed, args.mip_gap)
        report["results"].append(result)
        temporary = report_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        temporary.replace(report_path)
        print(f"Reference {index}/{len(selected)} | nodes={instance.node_count} | status={result['status']} | "
              f"optimal={result['proven_optimal']} | flight={result['flight_seconds']} | gap={result['mip_gap']}", flush=True)
    print(f"Offline references: {report_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
