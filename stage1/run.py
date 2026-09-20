"""Command-line entry point for the Walney-189 stage-1 reproduction."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np

from deployment import (
    DeploymentResult,
    Turbine,
    load_config,
    load_turbines,
    result_as_dict,
    run_deployment,
)


ROOT = Path(__file__).resolve().parent


def _csv_safe(value: object) -> object:
    """Prevent spreadsheet applications from evaluating untrusted ID fields as formulas."""
    if isinstance(value, str) and value.startswith(("=", "+", "-", "@")):
        return "'" + value
    return value


def _write_assignments(path: Path, result: DeploymentResult) -> None:
    by_nest = {nest.identifier: nest for nest in result.nests}
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["turbine_id", "nest_id", "nest_x_km", "nest_y_km"])
        for turbine_id, nest_id in sorted(result.assignments.items()):
            nest = by_nest[nest_id]
            writer.writerow([_csv_safe(turbine_id), _csv_safe(nest_id), nest.x_km, nest.y_km])


def _write_nests(path: Path, result: DeploymentResult) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "nest_id",
                "x_km",
                "y_km",
                "assigned_turbine_count",
                "route_distance_km",
                "flight_seconds",
                "equivalent_energy_seconds",
                "service_radius_margin_km",
                "endurance_margin_seconds",
                "route_ids",
            ]
        )
        for nest in result.nests:
            writer.writerow(
                [
                    _csv_safe(nest.identifier),
                    nest.x_km,
                    nest.y_km,
                    len(nest.turbine_ids),
                    nest.route_distance_km,
                    nest.flight_seconds,
                    nest.equivalent_energy_seconds,
                    nest.service_radius_margin_km,
                    nest.endurance_margin_seconds,
                    ";".join(str(_csv_safe(identifier)) for identifier in nest.route_ids),
                ]
            )


def _write_iteration_log(path: Path, result: DeploymentResult) -> None:
    """Write every candidate-pruning step in a spreadsheet-friendly audit log."""
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "iteration",
                "event",
                "removed_nest_id",
                "active_nest_count",
                "covered_turbine_count",
                "minimum_cover_count",
                "maximum_cover_count",
                "active_nest_ids",
            ]
        )
        for record in result.iterations:
            writer.writerow(
                [
                    record.iteration,
                    record.event,
                    _csv_safe(record.removed_nest_id or ""),
                    record.active_nest_count,
                    record.covered_turbine_count,
                    record.minimum_cover_count,
                    record.maximum_cover_count,
                    ";".join(_csv_safe(nest_id) for nest_id in record.active_nest_ids),
                ]
            )


def _configure_ga_plot_style() -> None:
    """Apply the visual defaults used by ``遗传对比试验.py``'s ``plot_solution``."""
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["mathtext.fontset"] = "stix"
    plt.rcParams["font.size"] = 18


def _write_solution_figure(
    path: Path,
    turbines: list[Turbine],
    result: DeploymentResult,
    nest_ids: tuple[str, ...],
    show: bool,
) -> None:
    """Draw one deployment state in the exact GA reference-figure style."""
    _configure_ga_plot_style()
    by_identifier = {turbine.identifier: turbine for turbine in turbines}
    turbine_coordinates = np.array(
        [(turbine.x_km, turbine.y_km) for turbine in turbines], dtype=float
    )
    nest_coordinates = np.array(
        [(by_identifier[nest_id].x_km, by_identifier[nest_id].y_km) for nest_id in nest_ids],
        dtype=float,
    )
    fig = plt.figure(figsize=(6, 7.5))
    ax = fig.add_subplot(111)

    ax.scatter(
        turbine_coordinates[:, 0],
        turbine_coordinates[:, 1],
        c="darkblue",
        label="Wind Turbines",
        s=30,
    )
    ax.scatter(
        nest_coordinates[:, 0],
        nest_coordinates[:, 1],
        c="red",
        s=30,
        marker="s",
        edgecolors="black",
        linewidths=1.5,
        label="UAV Nests",
    )
    for nest_id in nest_ids:
        nest = by_identifier[nest_id]
        circle = Circle(
            (nest.x_km, nest.y_km),
            result.effective_service_radius_km,
            color="blue",
            alpha=0.5,
            fill=0,
        )
        ax.add_patch(circle)

    # The reference's fixed 3748--3755 X range belongs to its 100-turbine
    # demonstration subset.  Walney-189 spans 3734.67--3749.18, so preserve
    # the same view style while deriving bounds that retain every input point.
    x_padding = 0.5
    y_padding = 0.5
    ax.set_xlim(turbine_coordinates[:, 0].min() - x_padding, turbine_coordinates[:, 0].max() + x_padding)
    ax.set_ylim(turbine_coordinates[:, 1].min() - y_padding, turbine_coordinates[:, 1].max() + y_padding)
    ax.set_xlabel("X Coordinate (km)", labelpad=10)
    ax.set_ylabel("Y Coordinate (km)", labelpad=10)
    ax.legend(
        fontsize=16,
        loc="upper right",
        bbox_to_anchor=(1.0, 1.0),
        borderaxespad=0.5,
    )
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.axis("equal")
    plt.tight_layout(pad=1.5)
    plt.subplots_adjust(left=0.15, bottom=0.15, right=0.9, top=0.9)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()


def _write_iteration_process_figures(
    output: Path, turbines: list[Turbine], result: DeploymentResult
) -> None:
    """Write individual GA-style placement snapshots for the pruning process."""
    final_record = result.iterations[-1]
    _write_solution_figure(
        output / "iteration_process.png",
        turbines,
        result,
        final_record.active_nest_ids,
        show=False,
    )
    for requested_step in (0, 45, 90, 181):
        record = min(
            result.iterations,
            key=lambda candidate: (abs(candidate.iteration - requested_step), candidate.iteration),
        )
        _write_solution_figure(
            output / f"iteration_step_{requested_step:03d}.png",
            turbines,
            result,
            record.active_nest_ids,
            show=False,
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the Walney-189 UAV deployment reproduction.")
    parser.add_argument("--input", type=Path, default=ROOT.parent / "turbine189.tsv")
    parser.add_argument("--config", type=Path, default=ROOT / "config.json")
    parser.add_argument("--output", type=Path, default=ROOT / "results")
    parser.add_argument(
        "--no-show",
        action="store_false",
        dest="show",
        default=True,
        help="Write PNG files without opening the Matplotlib windows.",
    )
    args = parser.parse_args()

    turbines = load_turbines(args.input)
    result = run_deployment(turbines, load_config(args.config))
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "summary.json").write_text(
        json.dumps(result_as_dict(result), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_assignments(args.output / "assignments.csv", result)
    _write_nests(args.output / "nests.csv", result)
    _write_iteration_log(args.output / "iteration_log.csv", result)
    _write_solution_figure(
        args.output / "deployment.png",
        turbines,
        result,
        tuple(nest.identifier for nest in result.nests),
        args.show,
    )
    _write_iteration_process_figures(args.output, turbines, result)
    for record in result.iterations:
        removed = record.removed_nest_id or "-"
        print(
            f"Iteration {record.iteration:03d} | {record.event} | removed={removed} | active nests={record.active_nest_count}"
        )
    print(f"Completed: {len(result.nests)} nests, {len(result.assignments)} turbines assigned.")
    return 0 if result.all_constraints_satisfied else 2


if __name__ == "__main__":
    raise SystemExit(main())
