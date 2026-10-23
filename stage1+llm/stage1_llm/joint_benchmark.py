"""Explicit offline-program comparison, never labelled as a new LLM search."""

import hashlib
import time
from dataclasses import asdict

from .artifacts import RunArtifacts, run_path
from .data import prepare_data, turbines_from_record
from .joint_planner import InfrastructureError
from .joint_provenance import source_provenance
from .joint_tools import JointTools, aggregate_metrics
from .sandbox import CandidateExecutionError, run_candidate


def benchmark(candidates, joint, outputs, run_id, planner):
    if run_id and run_path(outputs, run_id).exists():
        raise FileExistsError("Benchmark output exists")
    artifacts = RunArtifacts(outputs, run_id)
    records = []
    bundle = prepare_data(joint.search)
    datasets = bundle["datasets"]
    for candidate in candidates:
        for dataset in datasets:
            points = turbines_from_record(dataset)
            planner.reset_cache()
            telemetry = {}
            started = time.perf_counter()
            with artifacts.heartbeat(
                f"benchmark {candidate.name} {dataset['name']}",
                joint.search.heartbeat_seconds,
            ):
                try:
                    proposal = run_candidate(
                        candidate.resolve(),
                        points,
                        joint.search.deployment,
                        joint.search.seed,
                        joint.search.candidate_timeout_seconds,
                        planner=planner,
                        telemetry=telemetry,
                        limits={
                            "soft_seconds": joint.search.soft_seconds,
                            "max_region_evaluations": joint.search.max_region_evaluations,
                            "max_move_attempts": joint.search.max_move_attempts,
                        },
                    )
                    algorithm_seconds = time.perf_counter() - started
                    algorithm_counters = planner.stats()
                    checked = dict(
                        JointTools(
                            points,
                            joint.search.deployment,
                            planner=planner,
                            soft_seconds=300,
                        ).deployment(asdict(proposal))
                    )
                except InfrastructureError:
                    raise
                except (
                    CandidateExecutionError,
                    TimeoutError,
                    ValueError,
                    RuntimeError,
                ) as error:
                    algorithm_seconds = time.perf_counter() - started
                    algorithm_counters = planner.stats()
                    checked = {
                        "feasible": False,
                        "metrics": None,
                        "diagnostics": [
                            {"kind": "candidate_error", "message": str(error)[:1500]}
                        ],
                    }
            records.append(
                {
                    "candidate": str(candidate),
                    "dataset": dataset["name"],
                    "seconds": time.perf_counter() - started,
                    "algorithm_seconds": algorithm_seconds,
                    "algorithm_oracle_counters": algorithm_counters,
                    "telemetry": telemetry,
                    "source_sha256": hashlib.sha256(candidate.read_bytes()).hexdigest(),
                    "provenance": source_provenance(candidate),
                    "evaluation": checked,
                }
            )
            artifacts.log(
                f"{candidate.name} {dataset['name']} feasible={checked['feasible']} metrics={checked['metrics']}",
                event="benchmark_end",
            )
            artifacts.write_json(
                "benchmark.json",
                {
                    "origin": "explicit_program_benchmark",
                    "records": records,
                    "oracle": planner.metadata,
                    "cache_policy": "cold_candidate_instance; model retained",
                    "data_fingerprint": bundle["fingerprint"],
                    "work_limits": {
                        k: getattr(joint.search, k)
                        for k in (
                            "seed",
                            "soft_seconds",
                            "max_region_evaluations",
                            "max_move_attempts",
                            "candidate_timeout_seconds",
                        )
                    },
                    "candidate_summaries": [
                        {
                            "candidate": str(c),
                            "evaluated_instances": len(rows),
                            "expected_instances": len(datasets),
                            "aggregate": (
                                aggregate_metrics([r["evaluation"] for r in rows])
                                if len(rows) == len(datasets)
                                and all(r["evaluation"]["feasible"] for r in rows)
                                else None
                            ),
                        }
                        for c in candidates
                        for rows in [[r for r in records if r["candidate"] == str(c)]]
                    ],
                },
            )
    return artifacts.root
