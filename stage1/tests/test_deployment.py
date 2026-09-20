"""Behavioral tests for the Walney-189 deployment reproduction."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
import json
import subprocess
import sys

from stage1.deployment import (
    DeploymentConfig,
    InputValidationError,
    Turbine,
    assign_turbines,
    effective_service_radius_km,
    load_config,
    load_turbines,
    run_deployment,
)


def turbine(identifier: str, x: float, y: float) -> Turbine:
    return Turbine(identifier=identifier, x_km=x, y_km=y)


class InputTests(unittest.TestCase):
    def test_load_turbines_preserves_identifiers_and_coordinates(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "wind_farm.tsv"
            source.write_text("A01\t1.25\t2.5\nB02\t3\t4\n", encoding="utf-8")

            turbines = load_turbines(source)

        self.assertEqual(turbines, [turbine("A01", 1.25, 2.5), turbine("B02", 3.0, 4.0)])

    def test_load_turbines_rejects_duplicate_identifiers(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "duplicate.tsv"
            source.write_text("A01\t1\t2\nA01\t3\t4\n", encoding="utf-8")

            with self.assertRaises(InputValidationError):
                load_turbines(source)

    def test_load_turbines_rejects_non_finite_coordinates(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "invalid.tsv"
            source.write_text("A01\tnan\t2\n", encoding="utf-8")

            with self.assertRaises(InputValidationError):
                load_turbines(source)

    def test_load_turbines_rejects_wrong_column_count(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "invalid.tsv"
            source.write_text("A01\t1\n", encoding="utf-8")

            with self.assertRaises(InputValidationError):
                load_turbines(source)

    def test_load_config_rejects_non_kilometre_coordinates(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "config.json"
            source.write_text(json.dumps({"coordinate_unit": "m"}), encoding="utf-8")

            with self.assertRaises(InputValidationError):
                load_config(source)


class ConstraintTests(unittest.TestCase):
    def test_effective_radius_uses_the_stricter_communication_limit(self) -> None:
        config = DeploymentConfig()

        self.assertEqual(effective_service_radius_km(config), 7.0)

    def test_turbine_exactly_on_radius_boundary_is_covered(self) -> None:
        result = run_deployment(
            [turbine("NEST", 0, 0), turbine("EDGE", 7, 0)],
            DeploymentConfig(max_endurance_seconds=10_000),
        )

        self.assertEqual(set(result.assignments), {"NEST", "EDGE"})

    def test_short_endurance_prevents_an_infeasible_group(self) -> None:
        result = run_deployment(
            [turbine("A", 0, 0), turbine("B", 1, 0)],
            DeploymentConfig(max_endurance_seconds=100, communication_radius_km=7),
        )

        self.assertEqual(len(result.nests), 2)

    def test_energy_exactly_on_endurance_boundary_is_feasible(self) -> None:
        config = DeploymentConfig(
            communication_radius_km=1,
            max_endurance_seconds=11.4,
            hover_seconds_per_turbine=10,
            hover_to_flight_energy_ratio=1.14,
        )

        result = run_deployment([turbine("A", 0, 0)], config)

        self.assertTrue(result.all_constraints_satisfied)

    def test_deployment_does_not_impose_a_hidden_turbine_count_cap(self) -> None:
        result = run_deployment(
            [turbine(f"T{index}", 0, 0) for index in range(6)],
            DeploymentConfig(max_endurance_seconds=10_000),
        )

        self.assertEqual(len(result.nests), 1)
        self.assertEqual(len(result.nests[0].turbine_ids), 6)


class HeuristicTests(unittest.TestCase):
    def test_redundant_candidate_nest_is_pruned(self) -> None:
        result = run_deployment(
            [turbine("A", 0, 0), turbine("B", 0.1, 0)],
            DeploymentConfig(max_endurance_seconds=10_000),
        )

        self.assertEqual(len(result.nests), 1)
        self.assertEqual(set(result.assignments), {"A", "B"})

    def test_assignment_prefers_nearest_active_nest_then_identifier(self) -> None:
        turbines = [turbine("B", -1, 0), turbine("A", 1, 0), turbine("TARGET", 0, 0)]
        assignments = assign_turbines(
            turbines,
            {"A": ("A", "TARGET"), "B": ("B", "TARGET")},
        )

        self.assertEqual(assignments["TARGET"], "A")

    def test_same_input_and_config_produce_identical_result(self) -> None:
        turbines = [turbine("C", 2, 0), turbine("A", 0, 0), turbine("B", 1, 0)]
        config = DeploymentConfig(max_endurance_seconds=10_000)

        first = run_deployment(turbines, config)
        second = run_deployment(turbines, config)

        self.assertEqual(first, second)

    def test_pruning_exposes_a_replayable_iteration_log(self) -> None:
        result = run_deployment(
            [turbine("A", 0, 0), turbine("B", 0.1, 0)],
            DeploymentConfig(max_endurance_seconds=10_000),
        )

        self.assertEqual(result.iterations[0].iteration, 0)
        self.assertEqual(result.iterations[0].active_nest_count, 2)
        self.assertEqual(result.iterations[-1].active_nest_count, 1)
        self.assertEqual(result.iterations[-1].event, "remove_redundant_nest")


class WalneyIntegrationTests(unittest.TestCase):
    def test_walney_189_is_fully_and_feasibly_assigned(self) -> None:
        source = Path(__file__).resolve().parents[2] / "turbine189.tsv"
        result = run_deployment(load_turbines(source), DeploymentConfig())

        self.assertEqual(len(result.assignments), 189)
        self.assertEqual(len(set(result.assignments)), 189)
        self.assertTrue(result.all_constraints_satisfied)
        self.assertTrue(all(nest.identifier in result.assignments.values() for nest in result.nests))

    def test_cli_writes_stable_auditable_outputs(self) -> None:
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "results"
            command = [
                sys.executable,
                str(root / "run.py"),
                "--output",
                str(output),
                "--no-show",
            ]
            subprocess.run(command, cwd=root.parent, check=True, capture_output=True, text=True)
            first = {path.name: path.read_bytes() for path in output.iterdir()}
            subprocess.run(command, cwd=root.parent, check=True, capture_output=True, text=True)
            second = {path.name: path.read_bytes() for path in output.iterdir()}

        self.assertEqual(first, second)
        self.assertEqual(
            set(first),
            {
                "assignments.csv",
                "deployment.png",
                "iteration_log.csv",
                "iteration_process.png",
                "iteration_step_000.png",
                "iteration_step_030.png",
                "iteration_step_060.png",
                "iteration_step_090.png",
                "nests.csv",
                "summary.json",
            },
        )
        self.assertTrue(first["deployment.png"].startswith(b"\x89PNG\r\n\x1a\n"))
        self.assertTrue(first["iteration_process.png"].startswith(b"\x89PNG\r\n\x1a\n"))
        self.assertTrue(first["iteration_step_000.png"].startswith(b"\x89PNG\r\n\x1a\n"))


if __name__ == "__main__":
    unittest.main()
