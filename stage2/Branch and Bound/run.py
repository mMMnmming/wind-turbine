"""Independently solve a saved Stage1 deployment, then reuse its frozen optimum."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
from datetime import datetime
import hashlib
import importlib
import io
import json
import math
from pathlib import Path
import platform
import signal
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from stage2.benchmark import load_references
from stage2.data import instance_fingerprint, load_region_instances, split_fingerprint
from stage1.deployment import load_config


solver = importlib.import_module("stage2.Branch and Bound.solver")
inputs = importlib.import_module("stage2.Branch and Bound.inputs")
HERE = Path(__file__).resolve().parent
REPOSITORY = HERE.parents[1]
DEFAULT_STAGE1_SUMMARY = REPOSITORY / "stage1/results87/summary.json"
DEFAULT_TURBINES = REPOSITORY / "turbine87.tsv"


def _log(message: str) -> None:
    print(f"{datetime.now().astimezone().isoformat(timespec='seconds')} | {message}", flush=True)


def _save_reports(output: Path, report: dict) -> None:
    results = report["results"]
    flights = [row["flight_seconds"] for row in results if row.get("route") is not None]
    count = len(report["settings"]["instances"])
    all_optimal = len(results) == count and all(row["proven_optimal"] for row in results)
    flight_complete = len(flights) == count
    runtime = math.fsum(row["runtime_seconds"] for row in results)
    summary = {
        "region_count": count, "reported_count": len(results), "route_region_count": len(flights),
        "proven_optimal_count": sum(bool(row["proven_optimal"]) for row in results),
        "infeasible_count": sum(row["termination"] == "INFEASIBLE" for row in results),
        "runtime_seconds": runtime, "total_runtime_seconds": runtime,
        "total_flight_seconds": math.fsum(flights) if flight_complete else None,
        "available_flight_seconds": math.fsum(flights), "flight_total_complete": flight_complete,
        "all_regions_proven_optimal": all_optimal,
        "flight_total_status": "optimal" if all_optimal else "best_known" if flight_complete else "partial",
        "runtime_region_count": len(results), "runtime_total_complete": len(results) == count,
        "timing_scope": solver.TIMING_SCOPE,
        "scope": "Fixed-deployment regional routes, not joint deployment/routing optimality.",
    }
    summary["unfinished_count"] = summary["region_count"] - summary["proven_optimal_count"] - summary["infeasible_count"]
    report["summary"] = summary
    solver.write_json_atomic(output / "reference_results.json", report)
    solver.write_json_atomic(output / "summary.json", summary)
    stream = io.StringIO(newline="")
    fields = ["region", "nest_id", "node_count", "flight_seconds", "equivalent_energy_seconds",
              "runtime_seconds", "objective_bound_seconds", "mip_gap", "proven_optimal", "termination",
              "expanded_nodes", "instance_fingerprint"]
    writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    for record in results:
        writer.writerow({key: "'" + value if isinstance(value, str) and value.startswith(("=", "+", "-", "@"))
                         else value for key, value in record.items()})
    temporary = output / "results.csv.tmp"
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        handle.write(stream.getvalue())
    temporary.replace(output / "results.csv")


def _request_stop(signum: int, frame: object) -> None:
    raise KeyboardInterrupt


def _load_input(args: argparse.Namespace) -> tuple[list, dict]:
    if args.regions_json is None:
        summary = args.stage1_summary or DEFAULT_STAGE1_SUMMARY
        if not summary.is_file():
            raise FileNotFoundError(f"Missing Stage1 87 deployment: {summary}. Run stage1/run.py on turbine87.tsv first, or supply --stage1-summary. No network or deployment was run.")
        return inputs.load_stage1_input(summary, args.turbines or DEFAULT_TURBINES)
    if args.turbines is not None:
        raise ValueError("--turbines pairs with --stage1-summary, not --regions-json.")
    if not args.regions_json.is_file():
        raise FileNotFoundError(f"Missing {args.regions_json}; supply a saved Stage1 deployment or a region snapshot.")
    instances = load_region_instances(args.regions_json, args.stage1_config)
    payload = json.loads(args.regions_json.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and payload.get("frozen_reference") is True:
        source = payload["source"]
        inputs.freeze_regions(args.regions_json, instances, source, resume=True)
    else:
        config = load_config(args.stage1_config)
        source = {"kind": "regions_json", "regions_sha256": inputs.source_hash(args.regions_json),
                  "stage1_config": asdict(config), "communication_radius_km": config.communication_radius_km}
    return instances, source


def _region_figure(output: Path, index: int, instance: object, result: dict, dpi: int) -> None:
    plots = importlib.import_module("stage2.Branch and Bound.plots")
    # A resumed best_known route can improve without changing the filename.
    plots.write_region_route(output, index, instance, result, dpi=dpi)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    source_group = parser.add_mutually_exclusive_group()
    source_group.add_argument("--regions-json", type=Path)
    source_group.add_argument("--stage1-summary", type=Path)
    parser.add_argument("--turbines", type=Path, help="TSV paired with a native Stage1 summary; defaults to turbine87.tsv.")
    parser.add_argument("--stage1-config", type=Path, default=REPOSITORY / "stage1/config.json")
    parser.add_argument("--output", type=Path, default=HERE / "outputs/optimal87")
    parser.add_argument("--plot-dpi", type=int, default=150)
    parser.add_argument("--time-limit", type=float, default=0, help="Seconds per region this invocation; 0 means unlimited.")
    parser.add_argument("--log-interval", type=float, default=10)
    parser.add_argument("--checkpoint-interval", type=float, default=60)
    parser.add_argument("--tolerance", type=float, default=1e-8)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    if args.plot_dpi < 1:
        parser.error("--plot-dpi must be positive.")
    options = solver.Options(time_limit=args.time_limit, log_interval=args.log_interval,
                             checkpoint_interval=args.checkpoint_interval, tolerance=args.tolerance)
    instances, source = _load_input(args)
    keys = [instance_fingerprint(instance) for instance in instances]
    if not instances or len(set(keys)) != len(keys):
        raise ValueError("Input must contain nonempty, unique region geometries/environments.")
    report_path = args.output / "reference_results.json"
    if report_path.exists() and not args.resume:
        raise FileExistsError("Results already exist; use --resume or a new --output. Nothing was overwritten.")
    if not report_path.exists():
        if args.resume:
            raise FileNotFoundError("--resume requires an existing reference_results.json.")
        if any((args.output / name).exists() for name in ("regions.json", "checkpoints", "summary.json", "results.csv", "routes_overview.png")):
            raise FileExistsError("Orphan solver artifacts exist. Use a fresh output directory; nothing was overwritten.")
    frozen_path = args.output / "regions.json"
    inputs.freeze_regions(frozen_path, instances, source, resume=args.resume)
    settings = {"workflow_version": 2, "algorithm": solver.ALGORITHM_VERSION, "tolerance": options.tolerance,
                "input_sha256": hashlib.sha256(frozen_path.read_bytes()).hexdigest(),
                "split_fingerprint": split_fingerprint(instances), "instances": keys}
    if report_path.exists():
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if report.get("settings") != settings:
            raise ValueError("Resume input, ordered regions, algorithm or tolerance differs. Use a new output.")
        load_references(report_path, instances)
    else:
        report = {"schema_version": 1, "settings": settings, "results": [],
                  "algorithm": solver.ALGORITHM_VERSION, "timing_scope": solver.TIMING_SCOPE,
                  "input_source": source, "frozen_regions": "regions.json",
                  "platform": platform.platform(), "python": platform.python_version(),
                  "scope": "Fixed-deployment regional directed ATSP, not joint deployment/routing optimality."}
    args.output.mkdir(parents=True, exist_ok=True)
    _save_reports(args.output, report)
    previous_handler = signal.signal(signal.SIGTERM, _request_stop)
    try:
        records = {record["instance_fingerprint"]: record for record in report["results"]}
        for index, instance in enumerate(instances, 1):
            key = instance_fingerprint(instance)
            label = f"Region {index}/{len(instances)} | nest={instance.identifiers[instance.depot_index]}"
            previous = records.get(key)
            if previous and previous["termination"] in ("OPTIMAL", "INFEASIBLE"):
                _log(f"{label} | cached={previous['termination']} | runtime={previous['runtime_seconds']:.6f}s")
                _region_figure(args.output, index, instance, previous, args.plot_dpi)
                continue
            checkpoint = args.output / "checkpoints" / f"{key}.json"
            result = solver.solve_instance(instance, options=options, checkpoint_path=checkpoint,
                                           resume=args.resume and checkpoint.exists(), progress=_log, label=label)
            result["region"] = index
            records[key] = result
            report["results"] = [records[item] for item in keys if item in records]
            _save_reports(args.output, report)
            _region_figure(args.output, index, instance, result, args.plot_dpi)
            if result["termination"] in ("INTERRUPTED", "MEMORY_LIMIT"):
                importlib.import_module("stage2.Branch and Bound.plots").write_overview(
                    args.output, instances, report["results"], dpi=args.plot_dpi)
                _log(f"Saved unfinished search. Resume with the same command plus --resume: {args.output}")
                return 130 if result["termination"] == "INTERRUPTED" else 2
    finally:
        signal.signal(signal.SIGTERM, previous_handler)
    importlib.import_module("stage2.Branch and Bound.plots").write_overview(
        args.output, instances, report["results"], dpi=args.plot_dpi)
    unfinished = any(record["termination"] not in ("OPTIMAL", "INFEASIBLE") for record in report["results"])
    totals = report["summary"]
    _log(f"B&B totals | regions={totals['region_count']} | flight={totals['total_flight_seconds']}s | "
         f"solve={totals['total_runtime_seconds']:.6f}s | proven={totals['proven_optimal_count']} | "
         f"flight_status={totals['flight_total_status']}")
    _log(f"Saved all regional results: {report_path}; unfinished={unfinished}")
    return 2 if unfinished else 0


if __name__ == "__main__":
    raise SystemExit(main())
