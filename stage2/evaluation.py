"""Auditable baselines, reports, and figures for wind-aware ATSP evaluation."""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from statistics import mean, median
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch
import numpy as np

from .atsp import WindATSPInstance


@dataclass(frozen=True, slots=True)
class RouteReport:
    method: str
    route_indices: tuple[int, ...]
    route_identifiers: tuple[str, ...]
    flight_seconds: float
    equivalent_energy_seconds: float
    endurance_margin_seconds: float
    route_distance_km: float
    feasible: bool
    legs: tuple[dict[str, float | int | str], ...]


@dataclass(frozen=True, slots=True)
class RandomRouteSummary:
    sample_count: int
    feasible_count: int
    infeasible_count: int
    feasible_rate: float
    mean_equivalent_seconds: float | None
    best_equivalent_seconds: float | None
    median_equivalent_seconds: float | None
    representative: RouteReport | None


def nearest_neighbor_route(instance: WindATSPInstance) -> tuple[int, ...]:
    """Return a deterministic directed nearest-neighbour depot tour."""
    times = instance.travel_time_seconds
    remaining = set(range(instance.node_count)) - {instance.depot_index}
    route = [instance.depot_index]
    current = instance.depot_index
    while remaining:
        next_node = min(remaining, key=lambda node: (times[current, node], node))
        route.append(next_node)
        remaining.remove(next_node)
        current = next_node
    route.append(instance.depot_index)
    return tuple(route)


def evaluate_route(instance: WindATSPInstance, method: str, route: tuple[int, ...]) -> RouteReport:
    """Validate a closed tour and expose every directed edge for audit."""
    metrics = instance.evaluate_route(route)
    coordinates = instance.coordinates_km
    times = instance.travel_time_seconds
    legs: list[dict[str, float | int | str]] = []
    total_distance = 0.0
    for sequence, (first, second) in enumerate(zip(route, route[1:]), start=1):
        distance = float(np.linalg.norm(coordinates[second] - coordinates[first]))
        total_distance += distance
        legs.append(
            {
                "sequence": sequence,
                "from_index": first,
                "to_index": second,
                "from_id": instance.identifiers[first],
                "to_id": instance.identifiers[second],
                "distance_km": distance,
                "flight_seconds": float(times[first, second]),
            }
        )
    return RouteReport(
        method=method,
        route_indices=route,
        route_identifiers=tuple(instance.identifiers[index] for index in route),
        flight_seconds=metrics.flight_seconds,
        equivalent_energy_seconds=metrics.equivalent_energy_seconds,
        endurance_margin_seconds=instance.config.max_endurance_seconds - metrics.equivalent_energy_seconds,
        route_distance_km=total_distance,
        feasible=metrics.feasible,
        legs=tuple(legs),
    )


def random_route_summary(instance: WindATSPInstance, samples: int, seed: int) -> RandomRouteSummary:
    """Sample reproducible random tours and select a median feasible example."""
    if samples < 1:
        raise ValueError("samples must be positive.")
    generator = np.random.default_rng(seed)
    clients = np.array([node for node in range(instance.node_count) if node != instance.depot_index])
    reports = [
        evaluate_route(
            instance,
            "random_seeded",
            (instance.depot_index, *(int(item) for item in generator.permutation(clients)), instance.depot_index),
        )
        for _ in range(samples)
    ]
    feasible = [report for report in reports if report.feasible]
    if not feasible:
        return RandomRouteSummary(samples, 0, samples, 0.0, None, None, None, None)
    feasible.sort(key=lambda report: report.equivalent_energy_seconds)
    costs = [report.equivalent_energy_seconds for report in feasible]
    median_cost = float(median(costs))
    representative = min(feasible, key=lambda report: abs(report.equivalent_energy_seconds - median_cost))
    return RandomRouteSummary(
        samples,
        len(feasible),
        samples - len(feasible),
        len(feasible) / samples,
        float(mean(costs)),
        float(min(costs)),
        median_cost,
        representative,
    )


def _safe_csv(value: object) -> object:
    if isinstance(value, str) and value.startswith(("=", "+", "-", "@")):
        return "'" + value
    return value


def _route_dict(report: RouteReport | None) -> dict[str, object] | None:
    return None if report is None else asdict(report)


def _comparison_rows(
    evaluations: Iterable[tuple[int, WindATSPInstance, RouteReport, RouteReport, RandomRouteSummary]],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for region, instance, model, nearest, random_summary in evaluations:
        for report in (model, nearest):
            rows.append(
                {
                    "region": region,
                    "nest_id": instance.identifiers[instance.depot_index],
                    "node_count": instance.node_count,
                    "method": report.method,
                    "feasible": report.feasible,
                    "flight_seconds": report.flight_seconds,
                    "equivalent_energy_seconds": report.equivalent_energy_seconds,
                    "endurance_margin_seconds": report.endurance_margin_seconds,
                    "route_distance_km": report.route_distance_km,
                    "route_ids": ";".join(report.route_identifiers),
                    "random_samples": "",
                    "random_feasible_rate": "",
                    "random_best_equivalent_seconds": "",
                    "random_median_equivalent_seconds": "",
                    "random_mean_equivalent_seconds": "",
                }
            )
        representative = random_summary.representative
        rows.append(
            {
                "region": region,
                "nest_id": instance.identifiers[instance.depot_index],
                "node_count": instance.node_count,
                "method": "random_seeded_median_feasible",
                "feasible": representative.feasible if representative else False,
                "flight_seconds": representative.flight_seconds if representative else "",
                "equivalent_energy_seconds": representative.equivalent_energy_seconds if representative else "",
                "endurance_margin_seconds": representative.endurance_margin_seconds if representative else "",
                "route_distance_km": representative.route_distance_km if representative else "",
                "route_ids": ";".join(representative.route_identifiers) if representative else "",
                "random_samples": random_summary.sample_count,
                "random_feasible_rate": random_summary.feasible_rate,
                "random_best_equivalent_seconds": random_summary.best_equivalent_seconds or "",
                "random_median_equivalent_seconds": random_summary.median_equivalent_seconds or "",
                "random_mean_equivalent_seconds": random_summary.mean_equivalent_seconds or "",
            }
        )
    return rows


def _configure_plot_style() -> None:
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["mathtext.fontset"] = "stix"
    plt.rcParams["font.size"] = 18


def _draw_route(ax: plt.Axes, instance: WindATSPInstance, report: RouteReport, service_radius_km: float) -> None:
    coordinates = instance.coordinates_km
    depot = instance.depot_index
    ax.scatter(coordinates[:, 0], coordinates[:, 1], c="darkblue", s=30, label="Wind Turbines", zorder=3)
    ax.scatter(
        coordinates[depot, 0], coordinates[depot, 1], c="red", s=30, marker="s", edgecolors="black",
        linewidths=1.5, label="UAV Nest", zorder=5,
    )
    ax.add_patch(Circle(tuple(coordinates[depot]), service_radius_km, color="blue", alpha=0.5, fill=False))
    for first, second in zip(report.route_indices, report.route_indices[1:]):
        ax.add_patch(
            FancyArrowPatch(
                coordinates[first], coordinates[second], arrowstyle="->", mutation_scale=9,
                linewidth=1.15, color="dimgray", alpha=0.9, zorder=2,
            )
        )
    ax.plot([], [], color="dimgray", linewidth=1.15, label="Greedy Route")
    padding = 0.5
    ax.set_xlim(coordinates[:, 0].min() - padding, coordinates[:, 0].max() + padding)
    ax.set_ylim(coordinates[:, 1].min() - padding, coordinates[:, 1].max() + padding)
    ax.set_xlabel("X Coordinate (km)", labelpad=10)
    ax.set_ylabel("Y Coordinate (km)", labelpad=10)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.axis("equal")


def _write_region_figure(path: Path, instance: WindATSPInstance, report: RouteReport, service_radius_km: float, dpi: int) -> None:
    _configure_plot_style()
    fig, ax = plt.subplots(figsize=(6, 7.5))
    _draw_route(ax, instance, report, service_radius_km)
    ax.legend(fontsize=16, loc="upper right", bbox_to_anchor=(1.0, 1.0), borderaxespad=0.5)
    fig.tight_layout(pad=1.5)
    fig.subplots_adjust(left=0.15, bottom=0.15, right=0.9, top=0.9)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _write_overview(path: Path, evaluations: list[tuple[int, WindATSPInstance, RouteReport, RouteReport, RandomRouteSummary]], service_radius_km: float, dpi: int) -> None:
    _configure_plot_style()
    rows = math.ceil(len(evaluations) / 2)
    fig, axes = plt.subplots(rows, 2, figsize=(12, 7.5 * rows), squeeze=False)
    for ax, (region, instance, model, _, _) in zip(axes.flat, evaluations):
        _draw_route(ax, instance, model, service_radius_km)
        ax.text(0.02, 0.98, f"Region {region:02d}", transform=ax.transAxes, va="top", fontsize=14)
    for ax in axes.flat[len(evaluations):]:
        ax.set_visible(False)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=16, loc="upper right", bbox_to_anchor=(0.98, 0.98))
    fig.tight_layout(pad=2.0)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _write_cost_comparison(path: Path, rows: list[dict[str, object]], dpi: int) -> None:
    _configure_plot_style()
    regions = sorted({int(row["region"]) for row in rows})
    methods = ("model_greedy", "nearest_neighbor", "random_seeded_median_feasible")
    labels = ("Model", "Nearest neighbour", "Random median")
    colors = ("dimgray", "#E69F00", "#56B4E9")
    fig, ax = plt.subplots(figsize=(10, 6.5))
    x = np.arange(len(regions), dtype=float)
    width = 0.23
    for offset, method, label, color in zip((-width, 0.0, width), methods, labels, colors):
        selected = [next(row for row in rows if row["region"] == region and row["method"] == method) for region in regions]
        values = [float(row["equivalent_energy_seconds"]) if row["equivalent_energy_seconds"] != "" else 0.0 for row in selected]
        bars = ax.bar(x + offset, values, width, label=label, color=color, edgecolor="black", linewidth=0.6)
        for bar, row in zip(bars, selected):
            if not bool(row["feasible"]):
                bar.set_hatch("//")
    ax.set_xticks(x, [f"Region {region:02d}" for region in regions])
    ax.set_ylabel("Equivalent Energy Time (s)")
    ax.grid(axis="y", linestyle="--", alpha=0.6)
    ax.legend(fontsize=14)
    fig.tight_layout(pad=1.5)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def write_evaluation_artifacts(
    output: Path,
    evaluations: list[tuple[int, WindATSPInstance, RouteReport, RouteReport, RandomRouteSummary]],
    checkpoint_path: Path,
    input_hashes: dict[str, str],
    service_radius_km: float,
    plot_dpi: int,
) -> None:
    """Write all human-readable, machine-readable, and graphical artifacts."""
    output.mkdir(parents=True, exist_ok=True)
    rows = _comparison_rows(evaluations)
    fieldnames = list(rows[0]) if rows else []
    with (output / "evaluation_regions.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _safe_csv(value) for key, value in row.items()})
    regions = []
    for region, instance, model, nearest, random_summary in evaluations:
        regions.append(
            {
                "region": region,
                "nest_id": instance.identifiers[instance.depot_index],
                "node_count": instance.node_count,
                "model_greedy": _route_dict(model),
                "nearest_neighbor": _route_dict(nearest),
                "random_seeded": {**asdict(random_summary), "representative": _route_dict(random_summary.representative)},
            }
        )
    (output / "evaluation_routes.json").write_text(json.dumps(regions, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary = {
        "checkpoint": str(checkpoint_path),
        "input_hashes": input_hashes,
        "region_count": len(evaluations),
        "method_summary": {
            method: {
                "mean_equivalent_energy_seconds": float(mean(float(row["equivalent_energy_seconds"]) for row in rows if row["method"] == method and row["equivalent_energy_seconds"] != "")),
                "feasible_route_rate": float(mean(bool(row["feasible"]) for row in rows if row["method"] == method)),
            }
            for method in ("model_greedy", "nearest_neighbor", "random_seeded_median_feasible")
            if any(row["method"] == method and row["equivalent_energy_seconds"] != "" for row in rows)
        },
    }
    (output / "evaluation_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    log_lines = [f"Checkpoint: {checkpoint_path}", f"Regions: {len(evaluations)}"]
    for region, instance, model, nearest, random_summary in evaluations:
        log_lines.append(f"Region {region:02d} | nest={instance.identifiers[0]} | nodes={instance.node_count}")
        for report in (model, nearest):
            log_lines.append(
                f"  {report.method}: feasible={report.feasible} | flight={report.flight_seconds:.2f}s | "
                f"energy={report.equivalent_energy_seconds:.2f}s | margin={report.endurance_margin_seconds:.2f}s | "
                f"distance={report.route_distance_km:.3f}km"
            )
        log_lines.append(
            f"  random_seeded: feasible={random_summary.feasible_count}/{random_summary.sample_count} "
            f"({random_summary.feasible_rate:.1%}) | mean={random_summary.mean_equivalent_seconds} | "
            f"median={random_summary.median_equivalent_seconds} | best={random_summary.best_equivalent_seconds}"
        )
    (output / "evaluation.log").write_text("\n".join(log_lines) + "\n", encoding="utf-8")
    _write_cost_comparison(output / "cost_comparison.png", rows, plot_dpi)
    _write_overview(output / "model_routes_overview.png", evaluations, service_radius_km, plot_dpi)
    for region, instance, model, _, _ in evaluations:
        _write_region_figure(output / f"region_{region:02d}_routes.png", instance, model, service_radius_km, plot_dpi)
