"""Directed exact-reference formulation and honest solver-status reporting."""

from __future__ import annotations

import unittest
from contextlib import redirect_stderr
import io
import json
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

from stage2.benchmark import (build_atsp_lp, load_references, main, parse_gurobi_solution,
                             reference_summary, select_benchmark_instances, solve_with_gurobi)
from stage2.data import instance_fingerprint, instance_to_record
from stage2.tests.test_pool import small_instance


class BenchmarkTests(unittest.TestCase):
    def setUp(self) -> None:
        self.instance = small_instance("depot", 0.0)

    def solution(self, status: int = 2) -> dict:
        metrics = self.instance.evaluate_route((0, 1, 2, 0))
        return {
            "SolutionInfo": {"Status": status, "Runtime": 0.1, "ObjVal": metrics.flight_seconds,
                             "ObjBound": metrics.flight_seconds, "MIPGap": 0.0},
            "Vars": [{"VarName": name, "X": 1.0} for name in ("x_0_1", "x_1_2", "x_2_0")],
        }

    def test_formulation_is_directed_and_contains_subtour_and_endurance_constraints(self) -> None:
        lp = build_atsp_lp(self.instance)
        for expected in ("x_0_1", "x_1_0", "mtz_1_2", "endurance:", "Binary", "u_1"):
            self.assertIn(expected, lp)
        self.assertNotIn("x_0_0", lp)

    def test_proven_optimum_has_a_valid_closed_route(self) -> None:
        result = parse_gurobi_solution(self.instance, self.solution())
        self.assertTrue(result["proven_optimal"])
        self.assertEqual(result["route"], [0, 1, 2, 0])
        self.assertEqual(result["instance_fingerprint"], instance_fingerprint(self.instance))

    def test_time_limit_with_incumbent_is_not_called_optimal(self) -> None:
        result = parse_gurobi_solution(self.instance, self.solution(status=9))
        self.assertFalse(result["proven_optimal"])
        self.assertIsNotNone(result["flight_seconds"])

    def test_optimal_status_with_an_open_bound_is_not_a_closed_bound_optimum(self) -> None:
        solution = self.solution()
        solution["SolutionInfo"]["ObjBound"] *= 0.999
        solution["SolutionInfo"]["MIPGap"] = 0.001
        result = parse_gurobi_solution(self.instance, solution)
        self.assertTrue(result["optimal_within_solver_tolerance"])
        self.assertFalse(result["proven_optimal"])

    def test_no_solution_has_no_invented_objective_or_route(self) -> None:
        result = parse_gurobi_solution(self.instance, {"SolutionInfo": {"Status": 3, "Runtime": 0.1}})
        self.assertFalse(result["proven_optimal"])
        self.assertIsNone(result["flight_seconds"])
        self.assertIsNone(result["route"])

    def test_disconnected_or_incomplete_solver_route_is_rejected(self) -> None:
        solution = self.solution()
        solution["Vars"] = [{"VarName": "x_0_1", "X": 1.0}, {"VarName": "x_1_0", "X": 1.0}]
        with self.assertRaisesRegex(ValueError, "tour"):
            parse_gurobi_solution(self.instance, solution)

    def test_reference_gaps_exclude_infeasible_predictions_and_unproven_optima(self) -> None:
        reference = parse_gurobi_solution(self.instance, self.solution())
        key = instance_fingerprint(self.instance)
        report = reference_summary([(self.instance, reference["flight_seconds"] * 1.1, True)], {key: reference})
        self.assertAlmostEqual(report["mean_optimality_gap"], 0.1)
        self.assertEqual(report["optimality_gap_count"], 1.0)
        self.assertAlmostEqual(report["weighted_optimality_gap"], 0.1)
        excluded = reference_summary([(self.instance, 1.0, False)], {key: reference})
        self.assertEqual(excluded["optimality_gap_count"], 0.0)
        self.assertNotIn("mean_optimality_gap", excluded)
        reference["proven_optimal"] = False
        unproven = reference_summary([(self.instance, 1.0, True)], {key: reference})
        self.assertNotIn("mean_optimality_gap", unproven)

    def test_weighted_gap_uses_sum_not_mean_of_percentages(self) -> None:
        first = small_instance("first", 0, nodes=3)
        second = small_instance("second", 10, nodes=4)
        references = {
            instance_fingerprint(first): {"flight_seconds": 100., "feasible": True, "proven_optimal": True},
            instance_fingerprint(second): {"flight_seconds": 900., "feasible": True, "proven_optimal": True},
        }
        result = reference_summary([(first, 110., True), (second, 900., True)], references)
        self.assertAlmostEqual(result["mean_optimality_gap"], .05)
        self.assertAlmostEqual(result["weighted_optimality_gap"], .01)

    def test_subset_selection_covers_node_sizes_deterministically(self) -> None:
        instances = [small_instance(str(index), float(index), nodes=3 + index) for index in range(8)]
        selected = select_benchmark_instances(instances, 3)
        self.assertEqual([item.node_count for item in selected], [3, 6, 10])

    def test_solver_json_objective_must_match_measured_route(self) -> None:
        solution = self.solution()
        solution["SolutionInfo"]["ObjVal"] += 10.0
        with self.assertRaisesRegex(ValueError, "objective differs"):
            parse_gurobi_solution(self.instance, solution)
        with self.assertRaisesRegex(ValueError, "no objective"):
            parse_gurobi_solution(self.instance, {"SolutionInfo": {"Status": 2}})

    @patch("stage2.benchmark.subprocess.run")
    def test_cli_solution_is_parsed_and_existing_artifacts_are_not_overwritten(self, runner) -> None:
        def finish(invocation, **kwargs):
            result_path = next(item.split("=", 1)[1] for item in invocation if item.startswith("ResultFile="))
            Path(result_path).write_text(json.dumps(self.solution()), encoding="utf-8")
            return SimpleNamespace(returncode=0, stdout=b"test Gurobi output", stderr=b"")

        runner.side_effect = finish
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            result = solve_with_gurobi(self.instance, output, "gurobi_cl", 10.0, 1, 12, 0.0)
            self.assertTrue(result["proven_optimal"])
            content = (output / "model.lp").read_bytes()
            with self.assertRaisesRegex(ValueError, "Existing solver artifacts"):
                solve_with_gurobi(self.instance, output, "gurobi_cl", 10.0, 1, 12, 0.0)
            self.assertEqual(content, (output / "model.lp").read_bytes())
            self.assertEqual(runner.call_count, 1)
            self.assertIn("JSONSolDetail=1", runner.call_args.args[0])

    @patch("stage2.benchmark.subprocess.run")
    def test_failed_license_artifacts_are_preserved_and_timeout_is_reported(self, runner) -> None:
        runner.return_value = SimpleNamespace(returncode=1, stdout=b"Error 10009: License expired", stderr=b"")
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            with self.assertRaisesRegex(RuntimeError, "license expired"):
                solve_with_gurobi(self.instance, output, "gurobi_cl", 1.0, 1, 12, 0.0)
            before = (output / "solver_console.log").read_bytes()
            with self.assertRaisesRegex(ValueError, "Existing solver artifacts"):
                solve_with_gurobi(self.instance, output, "gurobi_cl", 1.0, 1, 12, 0.0)
            self.assertEqual(before, (output / "solver_console.log").read_bytes())
        runner.side_effect = subprocess.TimeoutExpired("gurobi_cl", 121)
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(RuntimeError, "process timeout"):
                solve_with_gurobi(self.instance, Path(directory), "gurobi_cl", 1.0, 1, 12, 0.0)

    def test_reference_file_rejects_wrong_split_or_false_optimality(self) -> None:
        result = parse_gurobi_solution(self.instance, self.solution())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "reference_results.json"
            path.write_text(json.dumps({"schema_version": 1, "results": [result]}), encoding="utf-8")
            self.assertEqual(len(load_references(path, [self.instance])), 1)
            with self.assertRaisesRegex(ValueError, "outside"):
                load_references(path, [small_instance("different", 100.0)])
            result["mip_gap"] = 0.1
            path.write_text(json.dumps({"schema_version": 1, "results": [result]}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "closed objective bound"):
                load_references(path, [self.instance])

    @patch("stage2.benchmark.shutil.which", return_value="gurobi_cl")
    @patch("stage2.benchmark.solve_with_gurobi")
    def test_benchmark_main_resumes_completed_records_and_checks_settings(self, solver, which) -> None:
        solver.return_value = parse_gurobi_solution(self.instance, self.solution())
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "regions.json"
            source.write_text(json.dumps({"regions": [instance_to_record(self.instance)]}), encoding="utf-8")
            arguments = ["stage2.benchmark", "--regions-json", str(source), "--output", str(root / "reference")]
            with patch("sys.argv", arguments):
                self.assertEqual(main(), 0)
                self.assertEqual(main(), 0)
            self.assertEqual(solver.call_count, 1)
            with patch("sys.argv", [*arguments, "--time-limit", "121"]):
                with self.assertRaisesRegex(ValueError, "settings differ"):
                    main()

    def test_invalid_solver_options_fail_before_any_solving(self) -> None:
        base = ["stage2.benchmark", "--regions-json", "unused.json", "--output", "unused"]
        for option in (["--time-limit", "0"], ["--threads", "0"], ["--mip-gap", "nan"], ["--seed", "-1"]):
            with self.subTest(option=option), patch("sys.argv", base + option), redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    main()
        with patch("stage2.benchmark.shutil.which", return_value=None), patch("sys.argv", base), redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                main()


if __name__ == "__main__":
    unittest.main()
