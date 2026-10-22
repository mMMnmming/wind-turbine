"""Read-only algorithm comparison on a frozen bundle; never creates an API client."""

from dataclasses import asdict
from pathlib import Path
import time

from .artifacts import RunArtifacts, run_path, sha256_file
from .data import prepare_data, turbines_from_record
from .evaluator import evaluate_proposal
from .sandbox import run_candidate


def benchmark(settings, outputs: Path, run_id: str, candidates: list[Path]) -> Path:
    if run_path(outputs, run_id).exists():
        raise FileExistsError("Benchmark exists; use a new run ID.")
    for candidate in candidates:
        if not candidate.is_file():
            raise FileNotFoundError(candidate)
    bundle = prepare_data(settings)
    artifacts = RunArtifacts(outputs, run_id)
    results = []
    for candidate in candidates:
        rows = []
        artifacts.log(
            f"Baseline {candidate} start; no LLM requests", event="benchmark_start"
        )
        for record in bundle["datasets"]:
            started = time.perf_counter()
            telemetry = {}
            try:
                with artifacts.heartbeat(
                    f'benchmark {candidate.name} dataset={record["name"]}',
                    settings.heartbeat_seconds,
                ):
                    points = turbines_from_record(record)
                    proposal = run_candidate(
                        candidate,
                        points,
                        settings.deployment,
                        settings.seed,
                        settings.candidate_timeout_seconds,
                        limits={
                            "soft_seconds": min(
                                settings.soft_seconds,
                                settings.candidate_timeout_seconds * 0.8,
                            ),
                            "max_region_evaluations": settings.max_region_evaluations,
                            "max_move_attempts": settings.max_move_attempts,
                        },
                        telemetry=telemetry,
                    )
                    elapsed = time.perf_counter() - started
                    check_start = time.perf_counter()
                    checked = evaluate_proposal(points, settings.deployment, proposal)
                row = {
                    "name": record["name"],
                    "feasible": checked.feasible,
                    "turbine_count": len(points),
                    "metrics": asdict(checked.metrics) if checked.feasible else None,
                    "violations": list(checked.violations),
                    "diagnostics": list(checked.diagnostics),
                    "algorithm_seconds": elapsed,
                    "evaluation_seconds": time.perf_counter() - check_start,
                    "telemetry": telemetry,
                }
            except Exception as error:
                row = {
                    "name": record["name"],
                    "feasible": False,
                    "error": f"{type(error).__name__}: {error}",
                    "algorithm_seconds": time.perf_counter() - started,
                    "telemetry": telemetry,
                }
            rows.append(row)
            artifacts.log(
                f'{candidate.name} dataset={record["name"]} feasible={row["feasible"]} '
                f'algorithm={row["algorithm_seconds"]:.3f}s metrics={row.get("metrics")} error={row.get("error")}',
                event="benchmark_dataset",
            )
            artifacts.write_json(
                "benchmark.json",
                {
                    "data_fingerprint": bundle["fingerprint"],
                    "completed": results,
                    "active": {"candidate": str(candidate), "datasets": rows},
                    "complete": False,
                },
            )
        feasible = all(r["feasible"] for r in rows)
        result = {
            "candidate": str(candidate),
            "source_sha256": sha256_file(candidate),
            "datasets": rows,
            "feasible": feasible,
            "algorithm_seconds": sum(r["algorithm_seconds"] for r in rows),
        }
        if feasible:
            result["objectives"] = [
                sum(r["metrics"]["nest_count"] / r["turbine_count"] for r in rows)
                / len(rows),
                sum(
                    r["metrics"]["total_proxy_flight_seconds"] / r["turbine_count"]
                    for r in rows
                )
                / len(rows),
                max(
                    r["metrics"]["maximum_region_tasks"] / r["turbine_count"]
                    for r in rows
                ),
            ]
        results.append(result)
    artifacts.write_json(
        "benchmark.json",
        {
            "data_fingerprint": bundle["fingerprint"],
            "deployment": asdict(settings.deployment),
            "results": results,
            "complete": True,
            "note": "All fixed instances; heuristic proxy, no optimality proof. Single-run timing includes child startup.",
        },
    )
    artifacts.log(f"Benchmark complete output={artifacts.root}", event="benchmark_end")
    return artifacts.root
