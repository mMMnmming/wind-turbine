"""Read saved model/B&B measurements and report regional and aggregate gaps."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from stage2.benchmark import _closed_bound
from stage2.data import instance_fingerprint, instance_from_record


COLUMNS = {
    "region": "区域", "node_count": "节点数", "model_flight_seconds": "模型飞行时间/s",
    "bnb_flight_seconds": "分支定界飞行时间/s", "model_runtime_seconds": "模型运行时间/s",
    "bnb_runtime_seconds": "分支定界运行时间/s", "flight_gap_percent": "飞行时间 gap/%", "status": "状态",
}


def _number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise ValueError(f"Missing or invalid {name}; rerun network evaluation if runtime was not recorded.")
    return float(value)


def flight_gap_percent(model: float, optimum: float) -> float | None:
    """Only a numerical-roundoff-sized negative gap may be suppressed."""
    model, optimum = _number(model, "model flight"), _number(optimum, "optimal flight")
    if optimum == 0:
        return None
    if model < optimum - max(1e-8, abs(optimum) * 1e-9):
        raise ValueError("Model beats the claimed optimum; check instance/environment and reference proof.")
    return max(0.0, (model - optimum) / optimum * 100)


def _all_region_totals(rows: list[dict], prefix: str, expected_count: int) -> dict:
    timed = [row for row in rows if row.get(f"{prefix}_runtime_seconds") is not None]
    flights = [row[f"{prefix}_flight_seconds"] for row in rows if row.get(f"{prefix}_flight_seconds") is not None]
    complete = len(flights) == expected_count and expected_count > 0
    return {
        "region_count": expected_count, "reported_count": len(timed), "route_region_count": len(flights),
        "missing_count": expected_count - len(timed), "flight_total_complete": complete,
        "total_flight_seconds": math.fsum(flights) if complete else None,
        "available_flight_seconds": math.fsum(flights),
        "total_runtime_seconds": math.fsum(row[f"{prefix}_runtime_seconds"] for row in timed) if timed else None,
        "runtime_total_complete": len(timed) == expected_count,
        "infeasible_count": (sum(row.get("bnb_termination") == "INFEASIBLE" for row in timed) if prefix == "bnb"
                             else sum(row.get("model_feasible") is False for row in timed)),
        "unproven_count": sum(not row.get("bnb_proven_optimal", False) and row.get("bnb_termination") != "INFEASIBLE"
                              for row in timed) if prefix == "bnb" else 0,
        "no_route_count": expected_count - len(flights),
        "note": "Raw measured totals, not the feasible/proven gap subset; unavailable full totals stay null.",
    }


def summarize_rows(rows: list[dict], expected_count: int | None = None) -> dict:
    """Aggregate matched timings; flight gap uses only valid proven comparisons."""
    matched = [row for row in rows if row.get("model_flight_seconds") is not None
               and row.get("bnb_flight_seconds") is not None]
    timed = [row for row in rows if row.get("model_runtime_seconds") is not None
             and row.get("bnb_runtime_seconds") is not None]
    valid = [row for row in rows if row["flight_gap_percent"] is not None]
    model_flight = math.fsum(row["model_flight_seconds"] for row in valid)
    bnb_flight = math.fsum(row["bnb_flight_seconds"] for row in valid)
    expected_count = len(rows) if expected_count is None else expected_count
    return {
        "row_count": expected_count, "displayed_row_count": len(rows), "matched_cost_count": len(matched), "valid_gap_count": len(valid),
        "coverage": "complete" if len(valid) == expected_count and rows else "partial",
        "model_all_regions": _all_region_totals(rows, "model", expected_count),
        "bnb_all_regions": _all_region_totals(rows, "bnb", expected_count),
        "model_flight_seconds": model_flight if valid else None,
        "bnb_flight_seconds": bnb_flight if valid else None,
        "flight_gap_percent": flight_gap_percent(model_flight, bnb_flight) if valid else None,
        "model_runtime_seconds": math.fsum(row["model_runtime_seconds"] for row in timed),
        "bnb_runtime_seconds": math.fsum(row["bnb_runtime_seconds"] for row in timed),
        "runtime_count": len(timed),
        "model_valid_runtime_seconds": math.fsum(row["model_runtime_seconds"] for row in valid),
        "bnb_valid_runtime_seconds": math.fsum(row["bnb_runtime_seconds"] for row in valid),
        "flight_total_scope": "Only feasible model routes with a proven positive-cost optimum.",
        "runtime_total_scope": "Matched instance rows with both measured runtimes, including no-solution searches.",
    }


def _records(payload: dict, name: str) -> dict[str, dict]:
    if payload.get("schema_version") != 1 or not isinstance(payload.get("results"), list):
        raise ValueError(f"Unsupported {name} results format; use the new saved results JSON.")
    records = {}
    for record in payload["results"]:
        key = record["instance_fingerprint"]
        if not isinstance(key, str) or len(key) != 64:
            raise ValueError(f"Invalid {name} instance fingerprint.")
        if key in records:
            raise ValueError(f"Duplicate {name} region fingerprint: {key}")
        _number(record.get("runtime_seconds"), f"{name} runtime_seconds")
        if record.get("flight_seconds") is not None:
            _number(record["flight_seconds"], f"{name} flight_seconds")
        records[key] = record
    return records


def _validate_measurement(record: dict, instance: object, *, reference: bool = False) -> None:
    if record["node_count"] != instance.node_count:
        raise ValueError("Node count differs from the saved geometry.")
    route = record.get("route")
    if route is not None:
        measured = instance.evaluate_route(tuple(route))
        if (bool(record["feasible"]) != measured.feasible
                or not math.isclose(record["flight_seconds"], measured.flight_seconds, rel_tol=1e-9, abs_tol=1e-8)):
            raise ValueError("Saved route, feasibility or flight time differs from the original environment.")
    elif record.get("feasible") or record.get("flight_seconds") is not None:
        raise ValueError("A saved feasible cost has no route.")
    if reference and record.get("proven_optimal"):
        if (record.get("status") != 2 or not record.get("feasible") or route is None
                or not _closed_bound(record["flight_seconds"], record.get("objective_bound_seconds"), record.get("mip_gap"))):
            raise ValueError("Reference claims optimality without a feasible closed-bound proof.")


def compare_results(model_payload: dict, reference_payload: dict) -> dict:
    """Match exact ordered geometry/environment fingerprints, not row numbers."""
    models, references = _records(model_payload, "model"), _records(reference_payload, "B&B")
    if not models and not references:
        raise ValueError("No regions to compare.")
    instances = {}
    for key in dict.fromkeys([*models, *references]):
        for record in (models.get(key), references.get(key)):
            if record is None:
                continue
            if "instance" not in record:
                raise ValueError("Missing full saved instance; rerun the new evaluator/solver.")
            instance = instance_from_record(record["instance"])
            if instance_fingerprint(instance) != key:
                raise ValueError("Instance fingerprint differs from coordinates, depot or environment.")
            _validate_measurement(record, instance, reference=record is references.get(key))
            instances[key] = instance
    rows, warnings = [], []
    for key in dict.fromkeys([*models, *references]):
        model, reference = models.get(key), references.get(key)
        instance = instances[key]
        source = model if model is not None else reference
        row = {"region": source.get("region"), "nest_id": instance.identifiers[instance.depot_index],
               "node_count": instance.node_count, "instance_fingerprint": key,
               "model_flight_seconds": model.get("flight_seconds") if model else None,
               "bnb_flight_seconds": reference.get("flight_seconds") if reference else None,
               "model_runtime_seconds": model["runtime_seconds"] if model else None,
               "bnb_runtime_seconds": reference["runtime_seconds"] if reference else None,
               "model_feasible": model.get("feasible") if model else None,
               "bnb_feasible": reference.get("feasible") if reference else None,
               "bnb_termination": reference.get("termination") if reference else None,
               "bnb_proven_optimal": reference.get("proven_optimal", False) if reference else False,
               "flight_gap_percent": None, "status": ""}
        states = []
        if model is None or reference is None:
            states.append("缺少模型区域" if model is None else "缺少分支定界区域")
            warnings.append(f"{states[-1]}: {key}; check missing regions or different geometry/environment.")
        if model is not None and not model["feasible"]:
            states.append("模型不可行")
        if reference is not None:
            if not reference.get("proven_optimal"):
                states.append("无可行解" if reference.get("termination") == "INFEASIBLE" else "未证明最优")
            elif reference["flight_seconds"] == 0:
                states.append("零参考成本，gap不定义")
            elif model is not None and model["feasible"]:
                row["flight_gap_percent"] = flight_gap_percent(model["flight_seconds"], reference["flight_seconds"])
        row["status"] = "；".join(states) or "已证明最优，模型可行"
        rows.append(row)
    declared_counts = [len(rows), len(reference_payload.get("settings", {}).get("instances", []))]
    for payload in (model_payload, reference_payload):
        count = payload.get("summary", {}).get("region_count", 0)
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ValueError("Invalid declared total region count.")
        declared_counts.append(count)
    expected_count = max(declared_counts)
    if expected_count > len(rows):
        warnings.append(f"Both files lack measurements for {expected_count - len(rows)} declared regions; totals are partial.")
    return {
        "schema_version": 1, "rows": rows, "summary": summarize_rows(rows, expected_count), "warnings": warnings,
        "model_device": model_payload.get("device", "not recorded"),
        "model_method": model_payload.get("method", "model_greedy (legacy metadata)"),
        "model_candidate_count": model_payload.get("candidate_count", 1),
        "model_checkpoint_sha256": model_payload.get("checkpoint_sha256"),
        "model_timing_scope": model_payload.get("timing_scope", "not recorded"),
        "bnb_timing_scope": reference_payload.get("timing_scope", "B&B search to proof/explicit stop; accumulated active time."),
        "timing_note": "Network inference and exact search/proof have different scopes; not a controlled same-hardware speed benchmark.",
    }


def _cell(value: object) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return f"{value:.9f}"
    return str(value).replace("|", "\\|").replace("\n", " ")


def write_comparison(report: dict, output: Path) -> str:
    """Write full-precision JSON/CSV and a readable Markdown table, no overwrite."""
    paths = [output / f"comparison.{suffix}" for suffix in ("csv", "md", "json")]
    if any(path.exists() for path in paths):
        raise FileExistsError("Comparison files already exist. Choose a new --output directory.")
    table = ["| " + " | ".join(COLUMNS.values()) + " |", "| " + " | ".join("---" for _ in COLUMNS) + " |"]
    table.insert(0, f"模型方法：{report.get('model_method', 'not recorded')}；候选数：{report.get('model_candidate_count', 1)}。\n")
    for row in report["rows"]:
        table.append("| " + " | ".join(_cell(row[key]) for key in COLUMNS) + " |")
    summary = report["summary"]
    scope = "全部区域" if summary["coverage"] == "complete" else "部分区域"
    table.extend(["", "## 全部区域原始合计", "",
                  "| 方法 | 全场飞行合计/s | 已有路线飞行合计/s | 运行合计/s | 已有路线/总区域 | 不可行 | 未证明 | 缺失 |",
                  "| --- | --- | --- | --- | --- | --- | --- | --- |"])
    for prefix, label in (("model", "网络模型"), ("bnb", "分支定界")):
        raw = summary[f"{prefix}_all_regions"]
        table.append(f"| {label} | {_cell(raw['total_flight_seconds'])} | {_cell(raw['available_flight_seconds'])} | "
                     f"{_cell(raw['total_runtime_seconds'])} | {raw['route_region_count']}/{raw['region_count']} | "
                     f"{raw['infeasible_count']} | {raw['unproven_count']} | {raw['missing_count']} |")
    table.extend(["", "原始合计包含模型不可行路线和未证明最优的现有路线，不能当作完整可行最优成本。",
                  "", "## 有效对比区域合计", "",
                  f"有效gap区域：{summary['valid_gap_count']}/{summary['row_count']}（{scope}）",
                  f"有效gap区域飞行时间合计：模型 {_cell(summary['model_flight_seconds'])} s；分支定界 {_cell(summary['bnb_flight_seconds'])} s。",
                  f"有效区域总飞行时间gap：{_cell(summary['flight_gap_percent'])}%（按飞行时间合计计算，不是区域百分比平均）。",
                  f"有效区域运行时间合计：模型 {_cell(summary['model_valid_runtime_seconds'])} s；分支定界 {_cell(summary['bnb_valid_runtime_seconds'])} s。",
                  f"双方有计时的匹配区域运行时间合计（{summary['runtime_count']}个区域）：网络推理 {_cell(summary['model_runtime_seconds'])} s；分支定界 {_cell(summary['bnb_runtime_seconds'])} s。",
                  f"模型设备：{report['model_device']}。网络推理与分支定界求解至证明的计时范围不同，不是同硬件受控速度实验。"])
    table.extend([f"警告：{warning}" for warning in report["warnings"]])
    markdown = "\n".join(table) + "\n"
    output.mkdir(parents=True, exist_ok=True)
    # UTF-8 BOM allows Excel to recognize the Chinese column headings.
    with paths[0].open("w", encoding="utf-8-sig", newline="") as handle:
        fields = [*COLUMNS, "row_type", "nest_id", "instance_fingerprint", "model_feasible", "bnb_feasible", "bnb_termination", "bnb_proven_optimal", "model_method", "model_candidate_count"]
        writer = csv.DictWriter(handle, fieldnames=[COLUMNS.get(key, key) for key in fields])
        writer.writeheader()
        for row in report["rows"]:
            row = {**row, "row_type": "region", "model_method": report.get("model_method"),
                   "model_candidate_count": report.get("model_candidate_count", 1)}
            # Prevent spreadsheet formulas if external node identifiers start with = etc.
            writer.writerow({COLUMNS.get(key, key): "'" + value if isinstance(value, str) and value.startswith(("=", "+", "-", "@"))
                             else value for key, value in row.items()})
        raw_model, raw_bnb = summary["model_all_regions"], summary["bnb_all_regions"]
        for kind, title, model_flight, bnb_flight, model_time, bnb_time, gap, status in (
            ("all_regions_total", "全部区域原始合计", raw_model["total_flight_seconds"], raw_bnb["total_flight_seconds"],
             raw_model["total_runtime_seconds"], raw_bnb["total_runtime_seconds"], None,
             f"模型不可行{raw_model['infeasible_count']}；参考未证明{raw_bnb['unproven_count']}；缺失{raw_model['missing_count']}/{raw_bnb['missing_count']}"),
            ("valid_regions_total", "有效对比区域合计", summary["model_flight_seconds"], summary["bnb_flight_seconds"],
             summary["model_valid_runtime_seconds"], summary["bnb_valid_runtime_seconds"], summary["flight_gap_percent"],
             f"有效{summary['valid_gap_count']}/{summary['row_count']}"),
        ):
            writer.writerow({"row_type": kind, COLUMNS["region"]: title,
                             COLUMNS["model_flight_seconds"]: model_flight, COLUMNS["bnb_flight_seconds"]: bnb_flight,
                             COLUMNS["model_runtime_seconds"]: model_time, COLUMNS["bnb_runtime_seconds"]: bnb_time,
                             COLUMNS["flight_gap_percent"]: gap, COLUMNS["status"]: status})
    paths[1].write_text(markdown, encoding="utf-8")
    paths[2].write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return markdown


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-results", type=Path, required=True)
    parser.add_argument("--reference-results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    model = json.loads(args.model_results.read_text(encoding="utf-8"))
    reference = json.loads(args.reference_results.read_text(encoding="utf-8"))
    report = compare_results(model, reference)
    print(write_comparison(report, args.output), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
