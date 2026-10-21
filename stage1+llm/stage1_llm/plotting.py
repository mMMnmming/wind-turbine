"""Self-contained deployment figures using the agreed Stage-1 visual convention."""
from __future__ import annotations

from pathlib import Path

from .domain import DeploymentResult, Turbine


def write_deployment_figure(path: Path, turbines: list[Turbine], result: DeploymentResult) -> None:
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.patches import Circle

    plt.rcParams.update({"font.family": "Times New Roman", "mathtext.fontset": "stix", "font.size": 18})
    by_id = {item.identifier: item for item in turbines}
    points = np.array([(item.x_km, item.y_km) for item in turbines], dtype=float)
    nests = np.array([(by_id[item.identifier].x_km, by_id[item.identifier].y_km) for item in result.nests], dtype=float)
    fig, ax = plt.subplots(figsize=(6, 7.5))
    ax.scatter(points[:, 0], points[:, 1], c="darkblue", s=30, label="Wind Turbines")
    ax.scatter(nests[:, 0], nests[:, 1], c="red", marker="s", s=30, edgecolors="black", linewidths=1.5, label="UAV Nests")
    for nest in result.nests:
        ax.add_patch(Circle((nest.x_km, nest.y_km), result.effective_service_radius_km, color="blue", alpha=0.5, fill=False))
    ax.set_xlim(points[:, 0].min() - 0.5, points[:, 0].max() + 0.5)
    ax.set_ylim(points[:, 1].min() - 0.5, points[:, 1].max() + 0.5)
    ax.set_xlabel("X Coordinate (km)", labelpad=10)
    ax.set_ylabel("Y Coordinate (km)", labelpad=10)
    ax.legend(fontsize=16, loc="upper right", bbox_to_anchor=(1.0, 1.0), borderaxespad=0.5)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.axis("equal")
    plt.tight_layout(pad=1.5)
    plt.subplots_adjust(left=0.15, bottom=0.15, right=0.9, top=0.9)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    plt.close(fig)
