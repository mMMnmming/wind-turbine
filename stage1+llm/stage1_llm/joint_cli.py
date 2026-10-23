"""V5 public commands. Validate the frozen oracle before constructing a paid client."""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path

from .artifacts import RunArtifacts, run_path, validate_run_id
from .data import prepare_data
from .domain import load_turbines
from .joint_artifacts import (
    export_deployment,
    export_joint_candidate,
    list_joint_candidates,
    verify_selection,
)
from .joint_planner import InfrastructureError, PlannerService
from .joint_search import JointSearch
from .joint_settings import DEFAULT_CONFIG, load_joint_settings
from .joint_tools import JointTools
from .sandbox import run_candidate
from .settings import load_deepseek_settings

LOCAL_ROOT = Path(__file__).resolve().parents[1]


def parser():
    result = argparse.ArgumentParser(
        description="V5: LLM deployment evolution with frozen wind-aware Stage2 greedy feedback"
    )
    subs = result.add_subparsers(dest="command", required=True)
    for command in (
        "prepare-data",
        "optimize",
        "resume",
        "evaluate",
        "benchmark",
        "smoke-test",
        "list-candidates",
        "export-best",
        "check-model",
    ):
        sub = subs.add_parser(command)
        sub.add_argument("--search-config", type=Path, default=DEFAULT_CONFIG)
        sub.add_argument("--outputs", type=Path, default=LOCAL_ROOT / "outputs")
        sub.add_argument("--run-id", type=validate_run_id)
        sub.add_argument("--stage2-checkpoint", type=Path)
        sub.add_argument("--stage2-device", choices=("cpu", "cuda"))
    for command in ("optimize", "resume"):
        sub = subs.choices[command]
        sub.add_argument(
            "--deepseek-config",
            type=Path,
            default=LOCAL_ROOT / "config/deepseek.local.json",
        )
        sub.add_argument("--evaluation-budget", type=int)
        sub.add_argument("--no-improvement-limit", type=int)
        sub.add_argument("--heartbeat-seconds", type=float)
        sub.add_argument(
            "--show-code", action=argparse.BooleanOptionalAction, default=None
        )
        sub.add_argument(
            "--stream", action=argparse.BooleanOptionalAction, default=None
        )
        sub.add_argument("--thinking", choices=("enabled", "disabled"))
        sub.add_argument("--reasoning-effort", choices=("low", "high", "max"))
        sub.add_argument("--max-tokens", type=int)
        sub.add_argument("--api-retries", type=int)
        sub.add_argument("--search-mode", choices=("standard", "focused_six"))
        sub.add_argument("--seed-candidate", type=Path)
    subs.choices["resume"].add_argument(
        "--resume-run-id", type=validate_run_id, required=True
    )
    subs.choices["resume"].add_argument("--reset-stall", action="store_true")
    subs.choices["evaluate"].add_argument("--candidate", type=Path, required=True)
    subs.choices["evaluate"].add_argument("--input", type=Path, required=True)
    subs.choices["evaluate"].add_argument("--timeout", type=float)
    subs.choices["check-model"].add_argument("--input", type=Path)
    subs.choices["benchmark"].add_argument(
        "--candidate", type=Path, action="append", required=True
    )
    subs.choices["export-best"].add_argument("--candidate-id")
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        if args.command == "smoke-test":
            from .joint_smoke import smoke_test

            smoke_test(args.outputs, args.run_id)
            return 0
        if args.command in ("list-candidates", "export-best"):
            if not args.run_id:
                raise ValueError("--run-id required")
            root = run_path(args.outputs, args.run_id)
            if args.command == "list-candidates":
                list_joint_candidates(root)
            else:
                print(export_joint_candidate(root, args.candidate_id), flush=True)
            return 0
        joint = load_joint_settings(args.search_config)
        saved_run = None
        if args.command == "resume":
            saved_run = json.loads(
                (
                    run_path(args.outputs, args.resume_run_id) / "search_state.json"
                ).read_text(encoding="utf-8")
            )
        overrides = {}
        for name in (
            "evaluation_budget",
            "no_improvement_limit",
            "heartbeat_seconds",
            "show_code",
        ):
            value = getattr(args, name, None)
            if value is not None:
                overrides[name] = value
        joint_overrides = {}
        if (
            saved_run is not None
            and saved_run["identity"].get("search_mode") == "focused_six"
        ):
            joint_overrides.update(
                search_mode="focused_six",
                seed_candidate=run_path(args.outputs, args.resume_run_id)
                / "seeds/0000.py",
            )
            if args.evaluation_budget is None:
                overrides["evaluation_budget"] = saved_run["evaluation_budget"]
            if args.no_improvement_limit is None:
                overrides["no_improvement_limit"] = saved_run["identity"]["settings"][
                    "no_improvement_limit"
                ]
        if getattr(args, "search_mode", None) is not None:
            joint_overrides["search_mode"] = args.search_mode
        if getattr(args, "seed_candidate", None) is not None:
            joint_overrides["seed_candidate"] = args.seed_candidate.resolve()
        joint = replace(
            joint, search=replace(joint.search, **overrides), **joint_overrides
        )
        if args.stage2_checkpoint:
            joint = replace(joint, checkpoint=args.stage2_checkpoint.resolve())
        if args.stage2_device:
            joint = replace(joint, device=args.stage2_device)
        if args.command == "prepare-data":
            bundle = prepare_data(joint.search)
            print(
                f"Frozen {len(bundle['datasets'])} instances: {bundle['fingerprint']}",
                flush=True,
            )
            return 0
        print(
            f"[stage2_start] Loading frozen greedy actor: {joint.checkpoint}; device={joint.device}; no API request yet",
            flush=True,
        )
        planner = PlannerService(
            joint.checkpoint,
            joint.device,
            joint.search.deployment,
            joint.gust_speed_mps,
        )
        try:
            print(
                f"[stage2_ready] {json.dumps(planner.metadata,ensure_ascii=False)}",
                flush=True,
            )
            if args.command == "check-model":
                from .joint_planner import CheckedPlanner, GreedyDecoder

                points = load_turbines(args.input or joint.search.target_input)[:4]
                expected = planner.region(points, points[0].identifier)
                if "route_ids" not in expected:
                    raise ValueError(
                        f"check-model input subset fails structural/radius checks: {expected.get('diagnostics')}; "
                        "use --input with a small nearby point set. This is not a whole-farm performance test."
                    )
                metadata = planner.metadata
                planner.close()
                decoder = GreedyDecoder(
                    joint.checkpoint,
                    joint.device,
                    joint.search.deployment,
                    joint.gust_speed_mps,
                )
                direct = CheckedPlanner(
                    decoder, joint.search.deployment, joint.gust_speed_mps
                ).region(points, points[0].identifier)
                if (
                    expected["route_ids"] != direct["route_ids"]
                    or expected["flight_seconds"] != direct["flight_seconds"]
                ):
                    raise InfrastructureError(
                        "Direct greedy and worker route/time disagree."
                    )
                print(
                    json.dumps(
                        {
                            "status": "verified_direct_greedy_equals_worker",
                            "oracle": metadata,
                            "checked_nodes": len(points),
                            "performance_claim": False,
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
                return 0
            if args.command in ("optimize", "resume"):
                llm = load_deepseek_settings(args.deepseek_config)
                llm_overrides = {}
                for option, field in (
                    ("thinking", "thinking"),
                    ("reasoning_effort", "reasoning_effort"),
                    ("max_tokens", "max_tokens"),
                    ("api_retries", "retries"),
                    ("stream", "stream"),
                ):
                    value = getattr(args, option)
                    if value is not None:
                        llm_overrides[field] = value
                llm = replace(llm, **llm_overrides)
                search = JointSearch(joint, llm, args.outputs, planner)
                try:
                    root = search.run(
                        args.resume_run_id if args.command == "resume" else args.run_id,
                        resume=args.command == "resume",
                        reset_stall=getattr(args, "reset_stall", False),
                    )
                    print(f"Saved v5 search: {root}", flush=True)
                finally:
                    search.client.close()
                return 0
            if args.command == "benchmark":
                from .joint_benchmark import benchmark

                benchmark(args.candidate, joint, args.outputs, args.run_id, planner)
                return 0
            candidate = args.candidate.resolve()
            verify_selection(candidate, planner.metadata)
            source = candidate.read_text(encoding="utf-8")
            origin = {"source": "unverified_external", "candidate": str(candidate)}
            selection = candidate.parent / "selection.json"
            if selection.is_file():
                selected = json.loads(selection.read_text(encoding="utf-8"))
                if selected["run_identity"]["settings"]["deployment"] != asdict(
                    joint.search.deployment
                ):
                    raise ValueError("Selected deployment config differs.")
                origin = {
                    "source": selected["origin"],
                    "candidate_id": selected["candidate_id"],
                }
                for name in (
                    "seed",
                    "soft_seconds",
                    "max_region_evaluations",
                    "max_move_attempts",
                    "candidate_timeout_seconds",
                ):
                    if selected["run_identity"]["settings"][name] != getattr(
                        joint.search, name
                    ):
                        raise ValueError(
                            f"Selected algorithm work configuration differs: {name}"
                        )
                if (
                    args.timeout is not None
                    and args.timeout != joint.search.candidate_timeout_seconds
                ):
                    raise ValueError(
                        "Selected algorithm timeout must match its saved configuration."
                    )
            if args.run_id and run_path(args.outputs, args.run_id).exists():
                raise FileExistsError("Evaluation output exists; use a new run ID.")
            artifacts = RunArtifacts(args.outputs, args.run_id)
            artifacts.write_text("candidates/evaluated.py", source)
            points = load_turbines(args.input)
            telemetry = {}
            timeout = (
                args.timeout
                if args.timeout is not None
                else joint.search.candidate_timeout_seconds
            )
            limits = {
                "soft_seconds": joint.search.soft_seconds,
                "max_region_evaluations": joint.search.max_region_evaluations,
                "max_move_attempts": joint.search.max_move_attempts,
            }
            started = time.perf_counter()
            with artifacts.heartbeat(
                "Selected deployment/network evaluation active",
                joint.search.heartbeat_seconds,
            ):
                # Keep the selected source path so sandbox verifies/uses its saved runtime.
                proposal = run_candidate(
                    candidate,
                    points,
                    joint.search.deployment,
                    joint.search.seed,
                    timeout,
                    limits=limits,
                    telemetry=telemetry,
                    planner=planner,
                    observer=lambda event: artifacts.log(
                        str(event), event=event.get("kind", "tool_progress")
                    ),
                )
            algorithm_seconds = time.perf_counter() - started
            checked = dict(
                JointTools(
                    points, joint.search.deployment, planner=planner, soft_seconds=300
                ).deployment(asdict(proposal))
            )
            artifacts.write_json(
                "evaluation.json",
                {
                    "origin": origin,
                    "proposal": asdict(proposal),
                    "evaluation": checked,
                    "telemetry": telemetry,
                    "algorithm_seconds": algorithm_seconds,
                    "oracle": planner.metadata,
                    "oracle_counters": planner.stats(),
                },
            )
            export_deployment(
                artifacts.root,
                points,
                proposal,
                checked,
                planner.metadata,
                joint.search.deployment,
            )
            artifacts.log(
                f"feasible={checked['feasible']} metrics={checked['metrics']} output={artifacts.root}",
                event="evaluation_end",
            )
            return 0 if checked["feasible"] else 2
        finally:
            planner.close()
    except (
        ValueError,
        FileNotFoundError,
        FileExistsError,
        InfrastructureError,
        KeyError,
    ) as error:
        print(f"ERROR: {type(error).__name__}: {error}", file=sys.stderr, flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
