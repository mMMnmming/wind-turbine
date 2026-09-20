"""Tests for reproducible Stage-1-backed training data preparation."""

from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from stage2.data import build_deployment_instances, sample_kde_topology


ROOT = Path(__file__).resolve().parents[2]


class DataPreparationTests(unittest.TestCase):
    def test_walney_102_generates_three_training_regions(self) -> None:
        instances = build_deployment_instances(ROOT / "turbine102.tsv", ROOT / "stage1" / "config.json")

        self.assertEqual(len(instances), 3)
        self.assertEqual(sorted(instance.node_count for instance in instances), [30, 33, 39])

    def test_kde_sampling_is_seed_reproducible(self) -> None:
        points = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 0.0]])

        first = sample_kde_topology(points, sample_size=6, seed=1234)
        second = sample_kde_topology(points, sample_size=6, seed=1234)

        np.testing.assert_allclose(first, second)


if __name__ == "__main__":
    unittest.main()
