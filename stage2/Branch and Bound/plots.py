"""Route diagrams for independent B&B results; never labelled greedy routes."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

from stage2.atsp import WindATSPInstance


def route_status(record: dict) -> str:
    if record.get("proven_optimal"):
        return "optimal"
    return "best_known" if record.get("route") is not None else "no_route"


def _draw(ax: plt.Axes, region: int, instance: WindATSPInstance, record: dict) -> None:
    coordinates = instance.coordinates_km
    depot = instance.depot_index
    clients = [i for i in range(instance.node_count) if i != depot]
    ax.scatter(coordinates[clients, 0], coordinates[clients, 1], s=28, color="tab:blue", label="Inspection turbines")
    ax.scatter(*coordinates[depot], s=65, marker="s", color="tab:red", label="UAV nest", zorder=5)
    for index, identifier in enumerate(instance.identifiers):
        ax.annotate(f"{index}: {identifier}", coordinates[index], xytext=(4, 4), textcoords="offset points", fontsize=7)
    status = route_status(record)
    route = record.get("route")
    if route is not None:
        instance.evaluate_route(tuple(route))
        for first, second in zip(route, route[1:]):
            if first != second:
                ax.add_patch(FancyArrowPatch(coordinates[first], coordinates[second], arrowstyle="->",
                                            mutation_scale=10, color="dimgray", linewidth=1.1,
                                            linestyle="-" if status == "optimal" else "--"))
        label = "Branch and Bound optimum" if status == "optimal" else "Best known (not proven optimal)"
        ax.plot([], [], color="dimgray", linestyle="-" if status == "optimal" else "--", label=label)
    else:
        ax.text(0.5, 0.5, "No feasible route obtained", transform=ax.transAxes, ha="center")
    flight = record.get("flight_seconds")
    cost = f"{flight:.3f} s" if flight is not None else "N/A"
    ax.set_title(f"Region {region:02d} | {record.get('termination', 'PENDING')}\nFlight: {cost}")
    ax.set_xlabel("X coordinate (km)")
    ax.set_ylabel("Y coordinate (km)")
    ax.set_aspect("equal", adjustable="datalim")
    ax.margins(0.12)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=7, loc="best")


def write_region_route(output: Path, region: int, instance: WindATSPInstance, record: dict, *, dpi: int = 150) -> Path:
    """Write a correctly labelled per-region figure outside solver timing."""
    output.mkdir(parents=True, exist_ok=True)
    path = output / f"region_{region:02d}_{route_status(record)}_route.png"
    figure, ax = plt.subplots(figsize=(7, 7))
    try:
        _draw(ax, region, instance, record)
        figure.tight_layout()
        figure.savefig(path, dpi=dpi)
    finally:
        plt.close(figure)
    return path


def write_overview(output: Path, instances: list[WindATSPInstance], records: list[dict], *, dpi: int = 150) -> Path:
    """Include unsolved regions explicitly instead of silently dropping them."""
    from stage2.data import instance_fingerprint

    lookup = {record["instance_fingerprint"]: record for record in records}
    columns = min(2, len(instances))
    rows = (len(instances) + columns - 1) // columns
    figure, axes = plt.subplots(rows, columns, figsize=(7 * columns, 6 * rows), squeeze=False)
    path = output / "routes_overview.png"
    try:
        for index, instance in enumerate(instances):
            record = lookup.get(instance_fingerprint(instance), {"termination": "PENDING", "route": None})
            _draw(axes.flat[index], index + 1, instance, record)
        for ax in list(axes.flat)[len(instances):]:
            ax.set_visible(False)
        figure.tight_layout()
        figure.savefig(path, dpi=dpi)
    finally:
        plt.close(figure)
    return path
