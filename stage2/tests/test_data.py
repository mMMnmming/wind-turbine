"""Tests for reproducible Stage-1-backed training data preparation."""

from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from stage1.deployment import InfeasibleDeploymentError, Turbine, run_deployment
from stage2.data import (
    build_deployment_instances,
    build_kde_instances,
    build_training_validation_instances,
    sample_kde_topology,
)


ROOT = Path(__file__).resolve().parents[2]


class DataPreparationTests(unittest.TestCase):
    def _small_source(self) -> list[Turbine]:
        return [Turbine("A", 0.0, 0.0), Turbine("B", 0.1, 0.0),
                Turbine("C", 0.0, 0.1), Turbine("D", 0.1, 0.1)]

    def test_rejected_topology_is_replaced_reproducibly_without_reducing_field_count(self) -> None:
        def generate():
            records = []
            calls = 0

            def reject_first(turbines, config):
                nonlocal calls
                calls += 1
                if calls == 1:
                    raise InfeasibleDeploymentError("forced infeasible final assignment")
                return run_deployment(turbines, config)

            with patch("stage2.data.load_turbines", return_value=self._small_source()), patch(
                "stage2.data.run_deployment", side_effect=reject_first
            ):
                instances = build_kde_instances(
                    ROOT / "turbine102.tsv", ROOT / "stage1" / "config.json", 2, 1234,
                    generation_records=records,
                )
            return instances, records

        first, first_records = generate()
        second, second_records = generate()
        self.assertEqual(first_records, second_records)
        self.assertFalse(first_records[0]["accepted"])
        accepted = [record for record in first_records if record["accepted"]]
        self.assertEqual([record["field_id"] for record in accepted], [0, 1])
        self.assertEqual(accepted[1]["seed"], 1235)
        self.assertNotEqual(accepted[0]["seed"], 1234)
        self.assertEqual(len(first), len(second))
        for left, right in zip(first, second):
            np.testing.assert_array_equal(left.coordinates_km, right.coordinates_km)

    def test_repeated_deployment_failure_stops_at_attempt_limit(self) -> None:
        with patch("stage2.data.load_turbines", return_value=self._small_source()), patch(
            "stage2.data.run_deployment", side_effect=InfeasibleDeploymentError("infeasible")
        ) as deployment:
            with self.assertRaisesRegex(InfeasibleDeploymentError, "after 2 attempts"):
                build_kde_instances(
                    ROOT / "turbine102.tsv", ROOT / "stage1" / "config.json", 1, 1234,
                    max_attempts_per_field=2,
                )
        self.assertEqual(deployment.call_count, 2)

    def test_unexpected_deployment_error_is_not_hidden_by_retries(self) -> None:
        with patch("stage2.data.load_turbines", return_value=self._small_source()), patch(
            "stage2.data.run_deployment", side_effect=ValueError("unexpected bug")
        ) as deployment:
            with self.assertRaisesRegex(ValueError, "unexpected bug"):
                build_kde_instances(ROOT / "turbine102.tsv", ROOT / "stage1" / "config.json", 1, 1234)
        self.assertEqual(deployment.call_count, 1)

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
