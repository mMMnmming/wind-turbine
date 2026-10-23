"""Read-only result loading plus PNG/PDF rendering, independent of torch/LLM."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from functools import lru_cache
from pathlib import Path

from .artifacts import atomic_json


def read_json(path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError(f"Cannot read {path}: {exc}") from exc


def child(root, relative):
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError(f"Saved path escapes run directory: {relative}")
    return path


def resolve_candidate(root, selected):
    requested_best = selected == "best"
    if selected == "best":
        summary = read_json(root / "summary.json")
        selected = summary.get("best_candidate")
        if not selected:
            raise ValueError("No qualified best candidate. Explicitly select an attempt Axxxx, old Cxxxx, or external.")
    if not isinstance(selected, str) or not re.fullmatch(r"(?:[CA][0-9]{4,}|external)", selected):
        raise ValueError("candidate must be best, Cxxxx, Axxxx or external")
    index_path = root / "diagnostics/runtime/candidate_index.json"
    if index_path.exists():
        index = read_json(index_path)
        entry = index.get("cases", {}).get(selected) or index.get("attempts", {}).get(selected)
        if not entry:
            raise ValueError(f"{selected} not found in {index_path}")
        if requested_best and not entry.get("eligible"):
            raise ValueError("Summary best does not point to a qualified case")
        return selected, child(root, entry["evaluation_directory"]), child(root, entry["problem_directory"]), {
            "status": entry["status"], "eligible": entry["eligible"],
            "source_attempt": entry["attempt_id"], "source_path": entry.get("source_path"),
            "expected_instances": index.get("expected_instances", [])}
    # Earlier v2 and standalone evaluate have direct result directories.
    meta_path = root / "diagnostics/evaluations" / (selected + ".json")
    meta = read_json(meta_path) if meta_path.exists() else {}
    if requested_best and meta.get("eligible") is False:
        raise ValueError("Summary best does not point to a qualified candidate")
    expected = []
    identity_path = root / "diagnostics/runtime/identity.json"
    if selected != "external" and identity_path.exists():
        settings = read_json(identity_path).get("settings", {})
        expected = ["real102"]
        expected += [f"external-validation-{i}" for i in range(1, len(settings.get("validation_inputs", []))+1)]
        expected += [f"training-kde-{i:03d}" for i in range(settings.get("training_virtual_fields", 0))]
        expected += [f"validation-kde-{i:03d}" for i in range(settings.get("validation_virtual_fields", 0))]
    return selected, root / "diagnostics/evaluations" / selected, root / "diagnostics/runtime" / selected, {
        "status": meta.get("status", "external_single_instance" if selected == "external" else "qualification_unknown"),
        "eligible": meta.get("eligible"), "source_attempt": None,
        "source_path": meta.get("source_path"), "expected_instances": expected}


def validate_result(problem, result, source):
    points = {}
    for turbine in problem["turbines"]:
        identifier = turbine["identifier"]
        x, y = turbine["x_km"], turbine["y_km"]
        if not isinstance(identifier, str) or identifier in points or not all(math.isfinite(v) for v in (x, y)):
            raise ValueError(f"{source}: duplicate/invalid turbine IDs or coordinates")
        points[identifier] = (x, y)
    solution = result["solution"]
    assignments = solution["assignments"]
    nests = solution["nest_ids"]
    if (not points or set(assignments) != set(points) or len(nests) != len(set(nests)) or
            set(assignments.values()) != set(nests) or not set(nests) <= set(points) or
            any(assignments[n] != n for n in nests)):
        raise ValueError(f"{source}: incomplete coverage, unknown nest IDs or invalid self assignment")
    audit = result["final_audit"]
    if not audit.get("feasible"):
        raise ValueError(f"{source}: saved final audit is not feasible")
    regions = audit["regions"]
    if len(regions) != len(nests) or {r["nest_id"] for r in regions} != set(nests):
        raise ValueError(f"{source}: final-audit regions do not match nests")
    for region in regions:
        nest, route = region["nest_id"], region["route_ids"]
        members = {t for t, n in assignments.items() if n == nest}
        if (not region.get("feasible") or len(route) != len(members)+1 or
                route[0] != nest or route[-1] != nest or set(route) != members or
                len(set(route[:-1])) != len(members)):
            raise ValueError(f"{source}: invalid/unknown IDs or nonclosed stored route for nest {nest}")
    for key in ("nest_count", "total_flight_seconds", "maximum_flight_seconds", "minimum_endurance_margin_seconds"):
        if not math.isfinite(audit["metrics"][key]):
            raise ValueError(f"{source}: missing/nonfinite metric {key}")
    return points, assignments, sorted(nests), regions, audit["metrics"]


def execution_marker(result, metadata):
    if result.get("execution_status") == "interrupted_timeout" and result.get("solution_usable") and not result.get("fatal_error"):
        return "INTERRUPTED"
    if not result.get("code_success", False):
        return "FAILED"
    return "NOT QUALIFIED" if metadata.get("eligible") is False else None


def render_diagnostic(problem_path, result_path, output, candidate, instance, metadata,
                      show_radius, show_routes=True):
    # Import only plotting dependencies; no evaluator, candidate source or Stage2.
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Circle

    problem, result = read_json(problem_path), read_json(result_path)
    points, assignments, nests, regions, metrics = validate_result(problem, result, result_path)
    environment = problem["environment"]
    wind_speed = float(environment["wind_speed_mps"])
    wind_bearing = float(environment["wind_direction_degrees"])
    radius = float(problem["config"]["communication_radius_km"])
    if not all(math.isfinite(v) for v in (wind_speed, wind_bearing, radius)):
        raise ValueError(f"{problem_path}: invalid wind/radius metadata")
    output.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(11, 9), layout="constrained")
    try:
        colors = {n: plt.get_cmap("tab20")(i % 20) for i, n in enumerate(nests)}
        xs, ys = [p[0] for p in points.values()], [p[1] for p in points.values()]
        for nest in nests:
            members = [t for t, n in assignments.items() if n == nest]
            ax.scatter([points[t][0] for t in members], [points[t][1] for t in members],
                       color=colors[nest], s=23, edgecolors="white", linewidths=.35, zorder=3)
            x, y = points[nest]
            ax.scatter([x], [y], marker="*", s=230, color=colors[nest], edgecolors="black", linewidths=.9, zorder=5)
            if len(nests) <= 12:
                ax.annotate(nest, (x, y), xytext=(6, 7), textcoords="offset points", fontsize=9, fontweight="bold")
            if show_radius:
                ax.add_patch(Circle((x, y), radius, fill=False, color=colors[nest], linestyle="--", alpha=.3))
                xs.extend((x-radius, x+radius))
                ys.extend((y-radius, y+radius))
        for region in regions if show_routes else []:
            color = colors[region["nest_id"]]
            route = region["route_ids"]
            for a, b in zip(route, route[1:]):
                if a == b:
                    continue
                ax.annotate("", xy=points[b], xytext=points[a], zorder=2,
                            arrowprops={"arrowstyle": "-|>", "color": color, "lw": .9,
                                        "alpha": .8, "shrinkA": 3, "shrinkB": 3,
                                        "mutation_scale": 9})
        pad = max(max(xs)-min(xs), max(ys)-min(ys), 1.0)*.08
        ax.set_xlim(min(xs)-pad, max(xs)+pad)
        ax.set_ylim(min(ys)-pad, max(ys)+pad)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("x (km)")
        ax.set_ylabel("y (km)")
        ax.ticklabel_format(style="plain", useOffset=False)
        ax.grid(alpha=.2)
        # Match Stage2's compass-to-Cartesian conversion; do not claim a wind-from bearing.
        angle = math.radians(90.0 - wind_bearing)
        xlo, xhi = ax.get_xlim()
        ylo, yhi = ax.get_ylim()
        wind_start = (xlo+.83*(xhi-xlo), ylo+.83*(yhi-ylo))
        wind_length = .08*min(xhi-xlo, yhi-ylo)
        wind_end = (wind_start[0]+wind_length*math.cos(angle), wind_start[1]+wind_length*math.sin(angle))
        ax.annotate("", xy=wind_end, xytext=wind_start,
                    arrowprops={"arrowstyle": "-|>", "color": "#263238", "lw": 2, "mutation_scale": 15})
        ax.text(.98, .98, f"Wind: {wind_speed:g} m/s, {wind_bearing:g} deg\nStage2 convention",
                transform=ax.transAxes, ha="right", va="top", fontsize=9,
                bbox={"facecolor": "white", "alpha": .85, "edgecolor": "none"})
        candidate_status = metadata["status"]
        if execution_marker(result, metadata):
            candidate_status = execution_marker(result, metadata) + ": " + candidate_status
        ax.set_title(f"{candidate} / {instance}\n{candidate_status}; stop: {result['stop_reason']}", fontsize=12)
        fig.text(.02, .01,
                 f"Nests: {metrics['nest_count']}  |  Total flight: {metrics['total_flight_seconds']:.2f} s  |  "
                 f"Max region: {metrics['maximum_flight_seconds']:.2f} s\n"
                 f"Min endurance margin: {metrics['minimum_endurance_margin_seconds']:.2f} s  |  "
                 f"Service radius: {radius:g} km  |  Stored Stage2 routes, no replanning",
                 fontsize=9)
        fig.get_layout_engine().set(rect=(0, .06, 1, .94))
        handles = [Line2D([], [], marker="o", linestyle="", color="gray", label="Turbine"),
                   Line2D([], [], marker="*", linestyle="", color="gray", markersize=12, label="Nest")]
        if len(nests) <= 12:
            handles += [Line2D([], [], color=colors[n], label="Region " + n) for n in nests]
        ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.01, 1),
                  fontsize=8, framealpha=.85)
        paths = [output / (instance + ".png"), output / (instance + ".pdf")]
        for path in paths:
            fig.savefig(path, dpi=300)
        return {"instance": instance, "status": "plotted", "files": [str(p) for p in paths],
                "metrics": metrics, "candidate_status": candidate_status,
                "code_success": result.get("code_success"), "stop_reason": result["stop_reason"],
                "execution_status": result.get("execution_status"),
                "solution_usable": result.get("solution_usable"),
                "problem_file": str(problem_path), "result_file": str(result_path),
                "nest_ids": nests, "assignment_count": len(assignments),
                "route_ids": {r["nest_id"]: r["route_ids"] for r in regions}}
    finally:
        plt.close(fig)


@lru_cache(maxsize=1)
def paper_font():
    from matplotlib.font_manager import FontProperties, findfont
    for family in ("Times New Roman", "Liberation Serif", "DejaVu Serif"):
        try:
            findfont(FontProperties(family=family), fallback_to_default=False)
        except ValueError:
            continue
        if family != "Times New Roman":
            print(f"[绘图字体] Times New Roman不可用，使用{family}", flush=True)
        return family
    raise ValueError("No supported serif font is available for paper style")


def paper_figure(points, nests, regions, radius, show_radius=True, show_routes=False,
                 status_marker=None):
    """Match stage1/run.py's figure parameters without importing its algorithm."""
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle

    with plt.rc_context({"font.family": paper_font(), "mathtext.fontset": "stix", "font.size": 18}):
        fig, ax = plt.subplots(figsize=(6, 7.5))
        try:
            xs, ys = [p[0] for p in points.values()], [p[1] for p in points.values()]
            ax.scatter(xs, ys, c="darkblue", label="Wind Turbines", s=30, zorder=3)
            ax.scatter([points[n][0] for n in nests], [points[n][1] for n in nests],
                       c="red", s=30, marker="s", edgecolors="black", linewidths=1.5,
                       label="UAV Nests", zorder=4)
            if show_radius:
                for nest in nests:
                    ax.add_patch(Circle(points[nest], radius, color="blue", alpha=.5, fill=False))
            if show_routes:
                for region in regions:
                    for a, b in zip(region["route_ids"], region["route_ids"][1:]):
                        if a != b:
                            ax.annotate("", xy=points[b], xytext=points[a], zorder=2,
                                        arrowprops={"arrowstyle": "-|>", "color": "blue", "lw": .8,
                                                    "alpha": .6, "shrinkA": 3, "shrinkB": 3})
            ax.set_xlim(min(xs)-.5, max(xs)+.5)
            ax.set_ylim(min(ys)-.5, max(ys)+.5)
            ax.set_xlabel("X Coordinate (km)", labelpad=10)
            ax.set_ylabel("Y Coordinate (km)", labelpad=10)
            ax.legend(fontsize=16, loc="upper right", bbox_to_anchor=(1., 1.), borderaxespad=.5)
            ax.grid(True, linestyle="--", alpha=.6)
            ax.axis("equal")
            if status_marker:
                ax.text(.02, .02, status_marker, transform=ax.transAxes, color="darkred", fontsize=10,
                        bbox={"facecolor": "white", "alpha": .85, "edgecolor": "none"})
            fig.tight_layout(pad=1.5)
            fig.subplots_adjust(left=.15, bottom=.15, right=.9, top=.9)
            return fig, ax
        except BaseException:
            plt.close(fig)
            raise


def render(problem_path, result_path, output, candidate, instance, metadata,
           show_radius=None, style="paper", show_routes=None):
    radius_visible = style == "paper" if show_radius is None else show_radius
    routes_visible = style == "diagnostic" if show_routes is None else show_routes
    if style == "diagnostic":
        row = render_diagnostic(problem_path, result_path, output, candidate, instance, metadata,
                                radius_visible, routes_visible)
    else:
        import matplotlib
        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
        problem, result = read_json(problem_path), read_json(result_path)
        points, assignments, nests, regions, metrics = validate_result(problem, result, result_path)
        radius = float(problem["config"]["communication_radius_km"])
        if not math.isfinite(radius) or radius <= 0:
            raise ValueError(f"{problem_path}: invalid service radius")
        marker = execution_marker(result, metadata)
        fig, _ = paper_figure(points, nests, regions, radius, radius_visible, routes_visible, marker)
        try:
            output.mkdir(parents=True, exist_ok=True)
            paths = [output/(instance+".png"), output/(instance+".pdf")]
            for path in paths:
                fig.savefig(path, dpi=300, bbox_inches="tight")
            row = {"instance": instance, "status": "plotted", "files": [str(p) for p in paths],
                   "metrics": metrics, "candidate_status": ((marker+": ") if marker else "")+metadata["status"],
                   "code_success": result.get("code_success"), "stop_reason": result["stop_reason"],
                   "execution_status": result.get("execution_status"),
                   "solution_usable": result.get("solution_usable"),
                   "problem_file": str(problem_path), "result_file": str(result_path),
                   "nest_ids": nests, "assignment_count": len(assignments), "font_family": paper_font(),
                   "environment": problem.get("environment"), "service_radius_km": radius,
                   "route_ids": {r["nest_id"]: r["route_ids"] for r in regions}}
        finally:
            plt.close(fig)
    return {**row, "style": style, "show_radius": radius_visible, "show_routes": routes_visible}


def source_fingerprint(root, candidate, metadata):
    relative = metadata.get("source_path") or ("candidates/"+candidate+".py")
    path = child(root, relative)
    # read_text normalizes CRLF; leave all other source differences significant.
    source = path.read_text(encoding="utf-8").replace("\r\n", "\n").replace("\r", "\n")
    return hashlib.sha256(source.encode("utf-8")).hexdigest()


def extra_tasks(extra_root, selected_fingerprint):
    """Only load saved standalone 87/189 evaluations; never execute or rescore."""
    _, results, problems, metadata = resolve_candidate(extra_root, "external")
    names = [name for name in ("turbine87", "turbine189")
             if (results/(name+".json")).exists() or (problems/(name+"-problem.json")).exists()]
    if not names:
        raise FileNotFoundError(f"No saved turbine87/turbine189 inputs or results in {extra_root}")
    provenance_error = None
    try:
        if source_fingerprint(extra_root, "external", metadata) != selected_fingerprint:
            provenance_error = "Extra evaluation source differs from selected candidate source"
    except (OSError, ValueError) as exc:
        provenance_error = "Cannot verify extra evaluation source: " + str(exc)
    return [{"instance": name, "problem_path": problems/(name+"-problem.json"),
             "result_path": results/(name+".json"), "metadata": metadata,
             "source_run": str(extra_root), "source_group": "extra_independent_evaluation",
             "expected_turbines": int(name.removeprefix("turbine")),
             "provenance_error": provenance_error} for name in names]


def parser():
    result = argparse.ArgumentParser(description="Plot saved UAV deployments on a headless server; no model/API calls")
    result.add_argument("--run-dir", type=Path, required=True)
    result.add_argument("--candidate", default="best", help="best, qualified Cxxxx, attempt Axxxx or external")
    group = result.add_mutually_exclusive_group()
    group.add_argument("--instance", default=None)
    group.add_argument("--all-instances", action="store_true")
    result.add_argument("--output-dir", type=Path)
    result.add_argument("--style", choices=("paper", "diagnostic"), default="paper")
    result.add_argument("--show-radius", action=argparse.BooleanOptionalAction, default=None)
    result.add_argument("--show-routes", action=argparse.BooleanOptionalAction, default=None)
    result.add_argument("--extra-run-dir", action="append", type=Path, default=[],
                        help="Saved external 87/189 evaluation; repeat for both wind farms")
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        root = args.run_dir.resolve()
        candidate, results, problems, metadata = resolve_candidate(root, args.candidate)
        output = args.output_dir.resolve() if args.output_dir else root / "plots" / candidate / args.style
        if args.all_instances:
            names = list(dict.fromkeys(metadata["expected_instances"] +
                         sorted(p.stem for p in results.glob("*.json"))))
            if not names and not args.extra_run_dir:
                raise ValueError(f"No stored instance results found in {results}")
        else:
            names = [args.instance or "real102"]
        for name in names:
            if not name or name in (".", "..") or "/" in name or "\\" in name:
                raise ValueError("Invalid instance name")
        tasks = [{"instance": name, "problem_path": problems/(name+"-problem.json"),
                  "result_path": results/(name+".json"), "metadata": metadata,
                  "source_run": str(root), "source_group": "selected_run",
                  "expected_turbines": int(name.removeprefix("turbine"))
                    if name in ("turbine87", "turbine189") else None} for name in names]
        for extra in args.extra_run_dir:
            extra_root = extra.resolve()
            try:
                fingerprint = source_fingerprint(root, candidate, metadata)
                tasks.extend(extra_tasks(extra_root, fingerprint))
            except (OSError, ValueError, KeyError) as exc:
                tasks.append({"instance": "extra:"+extra_root.name, "source_run": str(extra_root),
                              "source_group": "extra_independent_evaluation", "setup_error": str(exc),
                              "missing_source": isinstance(exc, FileNotFoundError)})
        summary = {"candidate": candidate, "source_attempt": metadata["source_attempt"],
                   "run_dir": str(root), "style": args.style, "requested": len(tasks),
                   "plotted": 0, "missing": 0, "errors": 0, "results": [],
                   "note": "Extra evaluations are displayed separately, not added to algorithm selection metrics"}
        scheduled = set()
        for task in tasks:
            name = task["instance"]
            origin = {k: task[k] for k in ("instance", "source_run", "source_group")}
            if task.get("setup_error"):
                category = "missing" if task.get("missing_source") else "errors"
                summary[category] += 1
                summary["results"].append({**origin, "status": "missing" if category == "missing" else "error",
                                           "error": task["setup_error"]})
                print(f"[额外结果不可用 {task['source_run']}] {task['setup_error']}", flush=True)
                continue
            problem_path, result_path = task["problem_path"], task["result_path"]
            missing = [str(p) for p in (problem_path, result_path) if not p.is_file()]
            if missing:
                summary["missing"] += 1
                summary["results"].append({**origin, "status": "missing", "files": missing})
                print(f"[缺失 {name}] " + "; ".join(missing), flush=True)
                continue
            try:
                if name in scheduled:
                    raise ValueError("Duplicate instance output name; refusing to overwrite an earlier result")
                if task.get("provenance_error"):
                    raise ValueError(task["provenance_error"])
                if task.get("expected_turbines") is not None:
                    actual = len(read_json(problem_path)["turbines"])
                    if actual != task["expected_turbines"]:
                        raise ValueError(f"{problem_path}: expected {task['expected_turbines']} turbines, found {actual}")
                row = render(problem_path, result_path, output, candidate, name, task["metadata"],
                             args.show_radius, args.style, args.show_routes)
                scheduled.add(name)
                summary["plotted"] += 1
                summary["results"].append({**row, **origin})
                print(f"[绘图完成 {name}] {output / (name+'.png')}", flush=True)
            except Exception as exc:
                summary["errors"] += 1
                summary["results"].append({**origin, "status": "error", "error": str(exc),
                                           "problem_file": str(problem_path), "result_file": str(result_path)})
                print(f"[绘图失败 {name}] {exc}｜数据:{result_path}｜坐标:{problem_path}", flush=True)
        atomic_json(output / "plot_summary.json", summary)
        print(f"[绘图结束] 请求{summary['requested']}｜完成{summary['plotted']}｜缺失{summary['missing']}"
              f"｜异常{summary['errors']}｜输出:{output}", flush=True)
        return 0 if not summary["missing"] and not summary["errors"] else 1
    except Exception as exc:
        print(f"[绘图停止] {type(exc).__name__}: {exc}", flush=True)
        return 1
