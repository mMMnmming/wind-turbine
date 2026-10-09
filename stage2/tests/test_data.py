"""Tests for reproducible Stage-1-backed training data preparation."""

from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from stage2.data import build_deployment_instances, build_training_validation_instances, sample_kde_topology


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

    def test_training_and_validation_seed_streams_generate_different_topologies(self) -> None:
        points = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 0.0], [1.0, 2.0]])

        training = sample_kde_topology(points, sample_size=8, seed=1234)
        validation = sample_kde_topology(points, sample_size=8, seed=1_001_234)

        self.assertFalse(np.array_equal(training, validation))

    def test_split_builder_keeps_real_102_regions_and_disjoint_virtual_ids(self) -> None:
        training, validation = build_training_validation_instances(
            ROOT / "turbine102.tsv", ROOT / "stage1" / "config.json",
            virtual_fields=2, seed=321, validation_fields=2,
        )

        self.assertEqual(sum(not instance.identifiers[0].startswith("virtual-") for instance in training), 3)
        train_virtual_ids = {identifier for instance in training for identifier in instance.identifiers if identifier.startswith("virtual-")}
        validation_virtual_ids = {identifier for instance in validation for identifier in instance.identifiers if identifier.startswith("virtual-")}
        self.assertTrue(train_virtual_ids)
        self.assertTrue(validation_virtual_ids)
        self.assertTrue(train_virtual_ids.isdisjoint(validation_virtual_ids))


if __name__ == "__main__":
    unittest.main()
