"""Behavioral tests for the wind-aware ATSP environment."""

from __future__ import annotations

import unittest

import numpy as np

try:
    import torch
except ImportError:  # The project intentionally does not install a CPU torch wheel.
    torch = None

from stage2.atsp import RouteValidationError, WindATSPConfig, WindATSPInstance


class WindATSPTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = WindATSPConfig(
            wind_speed_mps=6.0,
            wind_direction_degrees=90.0,
            max_horizontal_speed_mps=23.0,
            max_endurance_seconds=2_460.0,
            hover_seconds_per_turbine=10.0,
            hover_to_flight_energy_ratio=1.14,
        )
        self.instance = WindATSPInstance(
            identifiers=("NEST", "EAST", "NORTH"),
            coordinates_km=np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]),
            depot_index=0,
            config=self.config,
        )

    def test_paper_compass_east_wind_maps_to_positive_x(self) -> None:
        eastbound = self.instance.travel_time_seconds[0, 1]
        westbound = self.instance.travel_time_seconds[1, 0]

        self.assertLess(eastbound, westbound)

    def test_valid_route_starts_and_ends_at_depot_once_per_turbine(self) -> None:
        metrics = self.instance.evaluate_route((0, 1, 2, 0))

        self.assertTrue(metrics.feasible)
        self.assertEqual(metrics.visited_identifiers, ("EAST", "NORTH"))

    def test_repeated_or_missing_turbine_is_rejected(self) -> None:
        with self.assertRaises(RouteValidationError):
            self.instance.evaluate_route((0, 1, 1, 0))

    def test_safe_action_mask_rejects_visited_nodes_and_keeps_return_safe_nodes(self) -> None:
        mask = self.instance.safe_action_mask(current_index=0, visited={1}, elapsed_seconds=0.0)

        self.assertFalse(mask[0])
        self.assertFalse(mask[1])
        self.assertTrue(mask[2])

    def test_pointer_decoder_visits_only_valid_clients_without_nearest_neighbor_mask(self) -> None:
        if torch is None:
            self.skipTest("PyTorch is not usable in this interpreter.")
        from stage2.model import GraphPointerNetwork

        model = GraphPointerNetwork(hidden_dim=8, encoder_layers=1, heads=2)
        features = torch.zeros(2, 4, 7)
        valid = torch.tensor([[True, True, True, False], [True, True, False, False]])
        route, _ = model(
            features, decode_type="greedy", valid_node_mask=valid,
        )
        self.assertEqual(set(route[0].tolist()), {1, 2})
        self.assertEqual(route[1, 0].item(), 1)

    def test_node_features_include_fixed_design_gust(self) -> None:
        if torch is None:
            self.skipTest("PyTorch is not usable in this interpreter.")
        from stage2.model import make_node_features

        coordinates = torch.tensor([[[0.0, 0.0], [1.0, 0.0]]])
        features = make_node_features(coordinates, 6.0, 90.0, 12.0)

        self.assertEqual(features.shape, (1, 2, 7))
        self.assertAlmostEqual(features[0, 0, 4].item(), 1.0, places=6)
        self.assertAlmostEqual(features[0, 0, 6].item(), 12.0 / 23.0, places=6)


if __name__ == "__main__":
    unittest.main()
