"""Tests for reproducible, auditable routing evaluation artifacts."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from stage2.atsp import WindATSPConfig, WindATSPInstance
from stage2.evaluation import (
    evaluate_route,
    nearest_neighbor_route,
    random_route_summary,
    write_evaluation_artifacts,
)


class EvaluationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.instance = WindATSPInstance(
            identifiers=("NEST", "A", "B", "C"),
            coordinates_km=np.array([[0.0, 0.0], [0.5, 0.0], [1.0, 0.0], [0.5, 0.5]]),
            depot_index=0,
            config=WindATSPConfig(max_endurance_seconds=1_000.0),
        )

    def test_nearest_neighbor_is_deterministic_and_closed(self) -> None:
        route = nearest_neighbor_route(self.instance)
        self.assertEqual(route, (0, 1, 3, 2, 0))

    def test_seeded_random_summary_is_stable(self) -> None:
        first = random_route_summary(self.instance, samples=25, seed=1234)
        second = random_route_summary(self.instance, samples=25, seed=1234)
        self.assertEqual(first, second)
        self.assertEqual(first.sample_count, 25)

    def test_infeasible_route_is_reported_not_rewritten(self) -> None:
        constrained = WindATSPInstance(
            self.instance.identifiers,
            self.instance.coordinates_km,
            0,
            WindATSPConfig(max_endurance_seconds=1.0),
        )
        report = evaluate_route(constrained, "manual", (0, 1, 2, 3, 0))
        self.assertFalse(report.feasible)
        self.assertLess(report.endurance_margin_seconds, 0.0)

    def test_artifacts_include_logs_tables_json_and_figures(self) -> None:
        model = evaluate_route(self.instance, "model_greedy", (0, 1, 2, 3, 0))
        nearest = evaluate_route(self.instance, "nearest_neighbor", nearest_neighbor_route(self.instance))
        random = random_route_summary(self.instance, samples=10, seed=1234)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            write_evaluation_artifacts(
                output=output,
                evaluations=[(1, self.instance, model, nearest, random)],
                checkpoint_path=Path("best.pt"),
                input_hashes={"turbine_tsv_sha256": "abc"},
                service_radius_km=7.0,
                plot_dpi=100,
            )
            for name in (
                "evaluation_summary.json", "evaluation_regions.csv", "evaluation_routes.json",
                "evaluation.log", "cost_comparison.png", "model_routes_overview.png", "region_01_routes.png",
            ):
                self.assertTrue((output / name).is_file(), name)


if __name__ == "__main__":
    unittest.main()

