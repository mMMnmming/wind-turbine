"""Small v2 command surface. No paid request is made by prepare/check/evaluate."""
from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from dataclasses import replace
from datetime import datetime
from pathlib import Path

from .artifacts import Records, validate_run_id
from .domain import load_turbines
from .settings import LOCAL, load_deepseek_settings, load_settings


def parser():
    result = argparse.ArgumentParser(description="V2: bounded from-scratch deployment algorithm evolution")
    subparsers = result.add_subparsers(dest="command", required=True)
    for command in ("prepare-data", "check-model", "optimize", "resume", "evaluate"):
        sub = subparsers.add_parser(command)
        sub.add_argument("--search-config", type=Path, default=LOCAL / "config/search.json")
        sub.add_argument("--stage2-checkpoint", type=Path)
        sub.add_argument("--stage2-device", choices=("cpu", "cuda"))
        sub.add_argument("--outputs", type=Path, default=LOCAL / "outputs")
        sub.add_argument("--run-id", type=validate_run_id)
        if command in ("optimize", "resume", "evaluate"):
            sub.add_argument("--callback-timeout-seconds", type=float,
                             help="Single strategy call watchdog; best audited solution remains eligible (default 30)")
        if command in ("optimize", "resume"):
            sub.add_argument("--deepseek-config", type=Path, default=LOCAL / "config/deepseek.local.json")
            sub.add_argument("--evaluation-budget", type=int, help="Target number of fully qualified algorithm cases")
            sub.add_argument("--max-attempts-per-round", type=int, choices=(1, 2, 3),
                             help="Requests per case including initial generation (default 3)")
            sub.add_argument("--thinking", choices=("enabled", "disabled"))
            sub.add_argument("--max-tokens", type=int)
            sub.add_argument("--stream", action=argparse.BooleanOptionalAction, default=None)
            sub.add_argument("--api-retries", type=int, choices=(0,))
        if command in ("evaluate", "check-model"):
            sub.add_argument("--input", type=Path)
        if command == "evaluate":
            sub.add_argument("--candidate", type=Path, required=True)
            sub.add_argument("--final-test", action="store_true")
    return result


def initialize_planner(settings, records):
    from .planner import PlannerService

    records.progress("[准备] 加载冻结Stage2模型，验证权重和风况环境｜尚未请求LLM")
    stop = threading.Event()
    start = time.perf_counter()

    def pulse():
        while not stop.wait(settings.heartbeat_seconds):
            records.progress(f"[准备中] Stage2加载/环境验证｜耗时{time.perf_counter()-start:.0f}s")

    thread = threading.Thread(target=pulse, daemon=True)
    thread.start()
    try:
        planner = PlannerService(settings.stage2_checkpoint, settings.stage2_device,
                                 settings.deployment, settings.gust_speed_mps,
                                 timeout=settings.stage2_timeout_seconds)
    finally:
        stop.set()
        thread.join(timeout=2)
    records.json("diagnostics/runtime/oracle.json", planner.metadata)
    records.progress(f"[Stage2就绪] {planner.metadata['architecture']}｜{planner.metadata['device']}"
                     f"｜greedy×1｜风速{settings.deployment.wind_speed_mps:g}m/s"
                     f"｜风向{settings.deployment.wind_direction_degrees:g}°")
    planner.observer = lambda text: records.event("stage2_worker_heartbeat", message=text)
    return planner


def main(argv=None):
    args = parser().parse_args(argv)
    records, planner = None, None
    try:
        settings = load_settings(args.search_config)
        overrides = {}
        if args.stage2_checkpoint:
            overrides["stage2_checkpoint"] = args.stage2_checkpoint.resolve()
        if args.stage2_device:
            overrides["stage2_device"] = args.stage2_device
        if getattr(args, "evaluation_budget", None) is not None:
            overrides["evaluation_budget"] = args.evaluation_budget
        if getattr(args, "max_attempts_per_round", None) is not None:
            overrides["max_attempts_per_round"] = args.max_attempts_per_round
        if getattr(args, "callback_timeout_seconds", None) is not None:
            overrides["callback_timeout_seconds"] = args.callback_timeout_seconds
        settings = replace(settings, **overrides)
        if args.command == "prepare-data":
            from .data import prepare_data
            bundle = prepare_data(settings)
            print(f"数据准备完成：{len(bundle['datasets'])}实例；87仅作隔离检查，不进入搜索。", flush=True)
            return 0
        if args.command == "resume" and not args.run_id:
            raise ValueError("resume requires --run-id")
        run_id = args.run_id or ("v2-" + args.command + "-" + datetime.now().strftime("%Y%m%d_%H%M%S"))
        root = args.outputs.resolve() / run_id
        if args.command != "resume" and root.exists():
            raise FileExistsError("Run directory already exists; choose a new --run-id")
        if args.command == "resume" and not (root / "diagnostics/runtime/search_state.json").exists():
            raise FileNotFoundError("No v2 search_state.json for this run")
        llm = None
        if args.command in ("optimize", "resume"):
            llm = load_deepseek_settings(args.deepseek_config)
            llm_overrides = {}
            for name in ("thinking", "max_tokens", "stream"):
                if getattr(args, name, None) is not None:
                    llm_overrides[name] = getattr(args, name)
            if args.thinking == "disabled":
                llm_overrides["reasoning_effort"] = None
            llm = replace(llm, **llm_overrides)
            if not llm.api_key:
                raise ValueError("Configure an API key in the explicitly supplied private config")
        records = Records(root, (llm.api_key,) if llm else ())
        from .evolution import exclusive_run
        with exclusive_run(root):
            planner = initialize_planner(settings, records)
            if args.command == "check-model":
                from .planner import CheckedPlanner, GreedyDecoder
                points = load_turbines(args.input or settings.target_input)[:4]
                service = planner.region(points, points[0].identifier)
                metadata = planner.metadata
                planner.close()
                planner = None
                direct = CheckedPlanner(GreedyDecoder(settings.stage2_checkpoint, settings.stage2_device,
                                                      settings.deployment, settings.gust_speed_mps),
                                        settings.deployment, settings.gust_speed_mps).region(points, points[0].identifier)
                fields = ("route_ids", "flight_seconds", "energy_seconds", "feasible", "wind_audit")
                if any(direct[k] != service[k] for k in fields):
                    raise RuntimeError("Direct and service greedy results disagree")
                records.json("summary.json", {"status": "verified_direct_greedy_equals_worker",
                    "checked_nodes": len(points), "oracle": metadata, "performance_claim": False})
                records.progress("[验收通过] 正式权重与环境已加载，直接/桥接贪心路线及含风审核一致")
                return 0
            if args.command == "evaluate":
                from .runtime import run_instance
                points_path = (args.input or settings.target_input).resolve()
                if points_path == settings.protected_input.resolve() and not args.final_test:
                    raise ValueError("87 is final-test only; explicitly use --final-test")
                points = load_turbines(points_path)
                if not args.final_test:
                    from .data import _reject_protected
                    _reject_protected(points, load_turbines(settings.protected_input))
                from dataclasses import asdict
                dataset = {"name": points_path.stem, "split": "final_test" if args.final_test else "evaluation",
                           "turbines": [asdict(p) for p in points]}
                source = args.candidate.read_text(encoding="utf-8")
                from .candidate import validate_candidate_source
                validate_candidate_source(source)
                records.text("candidates/external.py", source)
                result = run_instance(source, str(args.candidate.resolve()), dataset, settings,
                                      planner, records, "external", 1, 1)
                from .evolution import compact_instance
                records.json("instance_metrics.json", [compact_instance(result)])
                usable = result["solution_usable"] and not result["fatal_error"]
                records.json("summary.json", {"status": result["execution_status"],
                    "solution_usable": result["solution_usable"], "fatal_error": result["fatal_error"],
                    "interruptions": result["interruptions"], "origin": "external_code",
                    "oracle": planner.metadata, "objectives": result["objectives"]})
                return 0 if usable else 1
            from .data import prepare_data
            from .evolution import Evolution
            bundle = prepare_data(settings)
            records.progress(f"[运行] 目标{settings.evaluation_budget}个有效案例｜每轮最多{settings.max_attempts_per_round}次尝试"
                             f"｜{len(bundle['datasets'])}实例/案例"
                             f"｜按搜索进展停止，无总时间截止｜单次调用保护{settings.callback_timeout_seconds:g}s"
                             "｜每代从单风机单巢起步")
            if settings.search_seconds is not None:
                records.progress("[兼容配置] search_seconds已弃用，不作为搜索停止条件")
            status = Evolution(settings, llm, bundle, planner, records).run(resume=args.command == "resume")
            return 0 if status == "target_completed" else 2
    except KeyboardInterrupt:
        if records:
            records.progress("[中断] 已保存进度；使用相同参数resume，不自动重发未知API请求")
        return 130
    except Exception as error:
        if records:
            import traceback
            records.json("diagnostics/errors/run.json", {"error": str(error),
                "error_type": type(error).__name__, "traceback": traceback.format_exc()})
            records.progress(f"[运行停止] {type(error).__name__}: {error}")
        else:
            print(f"[运行停止] {type(error).__name__}: {error}", file=sys.stderr)
        return 1
    finally:
        if planner is not None:
            planner.close()
