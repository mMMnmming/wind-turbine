"""Tests for normalized rewards, validation checkpoint ordering, and mixed batches."""

from __future__ import annotations

import unittest

import torch

from stage2.atsp import WindATSPConfig, WindATSPInstance
from stage2.train import (
    TrainingConfig,
    _pad_instance_batch,
    _reward,
    _route_metrics_batch,
    validate_resume_environment,
    validation_is_better,
)


class TrainingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = TrainingConfig(device="cpu")

    def test_feasible_completion_reward_exceeds_safety_failure(self) -> None:
        endurance = torch.tensor([100.0, 100.0])
        reward = _reward(
            flight=torch.tensor([50.0, 50.0]),
            energy=torch.tensor([80.0, 110.0]),
            feasible=torch.tensor([True, False]),
            endurance=endurance,
            config=self.config,
        )

        self.assertGreater(reward[0].item(), reward[1].item())
        self.assertLess(reward[1].item(), -20.0)

    def test_validation_prefers_feasibility_before_shorter_cost(self) -> None:
        infeasible_but_short = {"feasible_rate": 0.5, "mean_normalized_flight_cost": 0.2}
        feasible_but_long = {"feasible_rate": 1.0, "mean_normalized_flight_cost": 0.8}

        self.assertTrue(validation_is_better(feasible_but_long, infeasible_but_short))
        self.assertFalse(validation_is_better(infeasible_but_short, feasible_but_long))

    def test_resume_rejects_changed_wind_environment(self) -> None:
        environment = {"wind_speed_mps": 6.0, "gust_speed_mps": 12.0}

        with self.assertRaisesRegex(ValueError, "differs from the checkpoint"):
            validate_resume_environment(environment, {**environment, "gust_speed_mps": 15.0})

    def test_mixed_node_count_batch_is_padded_and_routes_are_measured_per_graph(self) -> None:
        config = WindATSPConfig(max_endurance_seconds=10_000.0)
        instances = [
            WindATSPInstance(("D", "A", "B"), torch.tensor([[0.0, 0.0], [0.2, 0.0], [0.4, 0.0]]).numpy(), 0, config),
            WindATSPInstance(("D", "A"), torch.tensor([[0.0, 0.0], [0.3, 0.0]]).numpy(), 0, config),
        ]
        coordinates, valid, times, hover, endurance = _pad_instance_batch(instances, torch.device("cpu"))
        route = torch.tensor([[1, 2], [1, 0]])
        flight, energy, distance, feasible = _route_metrics_batch(coordinates, valid, times, route, hover, endurance)

        self.assertEqual(valid.tolist(), [[True, True, True], [True, True, False]])
        self.assertTrue(torch.all(feasible))
        self.assertGreater(flight[0].item(), flight[1].item())
        self.assertGreater(distance[0].item(), distance[1].item())
        self.assertGreater(energy[0].item(), flight[0].item())


if __name__ == "__main__":
    unittest.main()
