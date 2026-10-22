"""Explicit offline-program comparison, never labelled as a new LLM search."""

from dataclasses import asdict
import time

from .artifacts import RunArtifacts, run_path
from .data import prepare_data, turbines_from_record
from .joint_tools import JointTools
from .sandbox import run_candidate


def benchmark(candidates, joint, outputs, run_id, planner):
    if run_id and run_path(outputs, run_id).exists():
        raise FileExistsError("Benchmark output exists")
    artifacts = RunArtifacts(outputs, run_id)
    records = []
    for candidate in candidates:
        for dataset in prepare_data(joint.search)["datasets"]:
            points = turbines_from_record(dataset)
            started = time.perf_counter()
            with artifacts.heartbeat(
                f"benchmark {candidate.name} {dataset['name']}",
                joint.search.heartbeat_seconds,
            ):
                proposal = run_candidate(
                    candidate.resolve(),
                    points,
                    joint.search.deployment,
                    joint.search.seed,
                    joint.search.candidate_timeout_seconds,
                    planner=planner,
                    limits={
                        "soft_seconds": joint.search.soft_seconds,
                        "max_region_evaluations": joint.search.max_region_evaluations,
                        "max_move_attempts": joint.search.max_move_attempts,
                    },
                )
            checked = dict(
                JointTools(
                    points, joint.search.deployment, planner=planner, soft_seconds=300
                ).deployment(asdict(proposal))
            )
            records.append(
                {
                    "candidate": str(candidate),
                    "dataset": dataset["name"],
                    "seconds": time.perf_counter() - started,
                    "evaluation": checked,
                }
            )
            artifacts.log(
                f"{candidate.name} {dataset['name']} {checked['metrics']}",
                event="benchmark_end",
            )
            artifacts.write_json(
                "benchmark.json",
                {
                    "origin": "explicit_program_benchmark",
                    "records": records,
                    "oracle": planner.metadata,
                },
            )
    return artifacts.root
