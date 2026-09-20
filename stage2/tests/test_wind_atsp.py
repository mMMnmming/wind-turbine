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

    def test_east_wind_creates_an_asymmetric_time_matrix(self) -> None:
        self.assertNotEqual(
            self.instance.travel_time_seconds[0, 2],
            self.instance.travel_time_seconds[2, 0],
        )

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

    def test_pointer_decoder_applies_dynamic_endurance_mask(self) -> None:
        if torch is None:
            self.skipTest("PyTorch is not usable in this interpreter.")
        from stage2.model import GraphPointerNetwork

        model = GraphPointerNetwork(hidden_dim=8, encoder_layers=1, heads=2)
        features = torch.zeros(1, 3, 6)
        times = torch.tensor([[[0.0, 8.0, 1.0], [8.0, 0.0, 1.0], [1.0, 1.0, 0.0]]])
        route, _ = model(
            features, decode_type="greedy", travel_time_seconds=times,
            hover_equivalent_seconds=0.0, max_endurance_seconds=10.0,
        )
        self.assertEqual(route[0, 0].item(), 2)


if __name__ == "__main__":
    unittest.main()
