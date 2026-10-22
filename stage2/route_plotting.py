"""Shared route artwork; colours show the environment's time-equivalent energy.

This is not a measured battery model. The fixed 0--100% scale is shared by every
region, so routes returning with plenty of energy are not artificially depleted.
No Torch import: the independent exact solver can use these diagrams unchanged.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from matplotlib.patches import FancyArrowPatch
import numpy as np

from .atsp import WindATSPInstance


# Match the reference: 100% deep purple, decreasing through blue/green to 0% yellow.
ENERGY_CMAP = plt.get_cmap("viridis_r")
ENERGY_NORM = Normalize(0, 100, clip=True)
ENERGY_LABEL = "SoC"


def route_energy_segments(
    instance: WindATSPInstance, route: tuple[int, ...], subdivisions: int = 24
) -> tuple[np.ndarray, np.ndarray]:
    """Colour each flight subsegment by remaining energy at its midpoint.

    Inspection hover is charged on arrival at each non-depot node; return to the
    depot charges no extra hover. Negative values are retained for unsafe tours.
    """
    instance.evaluate_route(tuple(route))
    if subdivisions < 1:
        raise ValueError("subdivisions must be positive.")
    segments, remaining = [], []
    elapsed = 0.0
    times = instance.travel_time_seconds
    limit = instance.config.max_endurance_seconds
    if limit <= 0:
        raise ValueError("Endurance must be positive for energy colours.")
    for first, second in zip(route, route[1:]):
        duration = times[first, second]
        start, end = instance.coordinates_km[[first, second]]
        if np.linalg.norm(end - start) > 0:
            fractions = np.linspace(0.0, 1.0, subdivisions + 1)
            points = start + fractions[:, None] * (end - start)
            segments.extend(np.stack((points[:-1], points[1:]), axis=1))
            middle = (fractions[:-1] + fractions[1:]) / 2
            remaining.extend(100 * (1 - (elapsed + middle * duration) / limit))
        elapsed += duration
        if second != instance.depot_index:
            elapsed += instance.config.hover_equivalent_seconds
    return np.asarray(segments, dtype=float).reshape(-1, 2, 2), np.asarray(
        remaining, dtype=float
    )


def draw_energy_route(
    ax: plt.Axes,
    instance: WindATSPInstance,
    route: tuple[int, ...],
    label: str,
    *,
    linestyle: str = "-",
) -> None:
    """Draw a directed route, node IDs, depot and fixed remaining-energy scale."""
    metrics = instance.evaluate_route(tuple(route))
    segments, remaining = route_energy_segments(instance, route)
    collection = LineCollection(
        segments.tolist(),
        cmap=ENERGY_CMAP,
        norm=ENERGY_NORM,
        linewidths=1.8,
        linestyles=linestyle,
        zorder=2,
    )
    collection.set_array(remaining)
    ax.add_collection(collection)
    coordinates = instance.coordinates_km
    clients = [
        index for index in range(instance.node_count) if index != instance.depot_index
    ]
    ax.scatter(
        coordinates[clients, 0],
        coordinates[clients, 1],
        s=23,
        color="#1677b8",
        edgecolors="white",
        linewidths=0.4,
        label="Wind turbines",
        zorder=4,
    )
    ax.scatter(
        *coordinates[instance.depot_index],
        s=65,
        marker="s",
        color="#d62728",
        edgecolors="black",
        linewidths=0.7,
        label="UAV nest",
        zorder=5,
    )
    elapsed = 0.0
    times = instance.travel_time_seconds
    for first, second in zip(route, route[1:]):
        a, b = coordinates[[first, second]]
        duration = times[first, second]
        if np.linalg.norm(b - a) > 0:
            colour = ENERGY_CMAP(
                ENERGY_NORM(
                    100
                    * (
                        1
                        - (elapsed + 0.5 * duration)
                        / instance.config.max_endurance_seconds
                    )
                )
            )
            ax.add_patch(
                FancyArrowPatch(
                    a + 0.40 * (b - a),
                    a + 0.62 * (b - a),
                    arrowstyle="-|>",
                    mutation_scale=10,
                    linewidth=1.2,
                    color=colour,
                    zorder=3,
                )
            )
        elapsed += duration
        if second != instance.depot_index:
            elapsed += instance.config.hover_equivalent_seconds
    for index, identifier in enumerate(instance.identifiers):
        # KDE IDs repeat a long field prefix at every node; retain the unique
        # turbine suffix in artwork, while saved routes keep the complete IDs.
        display_id = (
            identifier.rsplit("-", 1)[-1]
            if identifier.startswith("virtual-")
            else identifier
        )
        ax.annotate(
            display_id,
            coordinates[index],
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=7,
            zorder=6,
            clip_on=True,
        )
    ax.plot(
        [], [], color=ENERGY_CMAP(0.5), linewidth=1.8, linestyle=linestyle, label=label
    )
    colorbar = ax.figure.colorbar(
        collection, ax=ax, fraction=0.035, pad=0.04, ticks=[0, 100]
    )
    colorbar.ax.set_title(ENERGY_LABEL, fontsize=11, pad=8)
    colorbar.outline.set_visible(False)
    colorbar.ax.tick_params(labelsize=8)
    remaining_at_return = 100 * (
        1 - metrics.equivalent_energy_seconds / instance.config.max_endurance_seconds
    )
    text = f"Flight: {metrics.flight_seconds / 60:.2f} min | Estimated return SoC: {remaining_at_return:.1f}%"
    if not metrics.feasible:
        text += " (INFEASIBLE)"
    ax.set_aspect("equal", adjustable="box")
    span = np.ptp(coordinates, axis=0)
    padding = max(float(span.max()) * 0.12, 0.08)
    ax.set_xlim(coordinates[:, 0].min() - padding, coordinates[:, 0].max() + padding)
    ax.set_ylim(coordinates[:, 1].min() - padding, coordinates[:, 1].max() + padding)
    ax.set_xlabel("X coordinate (km)\n" + text, fontsize=10)
    ax.set_ylabel("Y coordinate (km)", fontsize=11)
    ax.tick_params(labelsize=10)
    ax.grid(True, linestyle=":", alpha=0.18)
