"""Stage1-native input freezes and independently labelled optimal route plots."""

from __future__ import annotations

import copy
from dataclasses import asdict
import importlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from stage1.deployment import DeploymentConfig, load_turbines, result_as_dict, run_deployment


inputs = importlib.import_module("stage2.Branch and Bound.inputs")
plots = importlib.import_module("stage2.Branch and Bound.plots")
runner = importlib.import_module("stage2.Branch and Bound.run")


def native_fixture(root: Path):
    turbines = root / "turbines.tsv"
    turbines.write_text("D\t0\t0\nA\t0.1\t0.1\nB\t0.2\t0.1\nC\t0.3\t0\n", encoding="utf-8")
    deployment = run_deployment(load_turbines(turbines), DeploymentConfig())
    summary = root / "summary.json"
    summary.write_text(json.dumps(result_as_dict(deployment)), encoding="utf-8")
    return summary, turbines


class NativeInputTests(unittest.TestCase):
    def test_native_summary_needs_no_network_or_redeployment(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            summary, turbines = native_fixture(root)
            with patch("stage1.deployment.run_deployment", side_effect=AssertionError("Must not redeploy")):
                instances, source = inputs.load_stage1_input(summary, turbines)
            saved = json.loads(summary.read_text())
            self.assertEqual(len(instances), saved["nest_count"])
            self.assertEqual(sum(item.node_count for item in instances), 4)
            self.assertEqual(instances[0].config.gust_speed_mps, 12)
            self.assertIn("stage1_summary_sha256", source)
            self.assertIn("turbines_sha256", source)
            self.assertEqual(instances[0].identifiers[0], saved["nests"][0]["identifier"])
            for name in asdict(instances[0].config):
                if name != "gust_speed_mps":
                    self.assertEqual(getattr(instances[0].config, name), saved["config"][name])

    def test_wrong_tsv_memberships_coordinates_and_config_fail(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            summary, turbines = native_fixture(root)
            original = json.loads(summary.read_text())
            cases = []
            for change in ("assignment", "duplicate", "coordinate", "config", "constraints", "route"):
                bad = copy.deepcopy(original)
                if change == "assignment":
                    bad["assignments"].pop("A")
                elif change == "duplicate":
                    bad["nests"][0]["turbine_ids"].append("A")
                elif change == "coordinate":
                    bad["nests"][0]["x_km"] += 1
                elif change == "config":
                    bad["config"]["wind_speed_mps"] = 100
                elif change == "route":
                    bad["nests"][0]["route_distance_km"] += 1
                else:
                    bad["all_constraints_satisfied"] = False
                cases.append(bad)
            for bad in cases:
                summary.write_text(json.dumps(bad))
                with self.assertRaises(ValueError):
                    inputs.load_stage1_input(summary, turbines)
            summary.write_text(json.dumps(original))
            turbines.write_text(turbines.read_text().replace("A\t", "OTHER\t"))
            with self.assertRaises(ValueError):
                inputs.load_stage1_input(summary, turbines)

    def test_snapshot_is_immutable_and_resume_checks_source(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            summary, turbines = native_fixture(root)
            instances, source = inputs.load_stage1_input(summary, turbines)
            target = root / "frozen/regions.json"
            inputs.freeze_regions(target, instances, source)
            before = target.read_bytes()
            inputs.freeze_regions(target, instances, source, resume=True)
            self.assertEqual(before, target.read_bytes())
            with self.assertRaises(FileExistsError):
                inputs.freeze_regions(target, instances, source)
            with self.assertRaises(ValueError):
                inputs.freeze_regions(target, instances, {**source, "turbines_sha256": "wrong"}, resume=True)
            with self.assertRaises(FileNotFoundError):
                inputs.freeze_regions(root / "absent.json", instances, source, resume=True)

    def test_native_cli_exports_routes_totals_and_preserves_cache(self):
        from contextlib import redirect_stdout
        import io

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            summary, turbines = native_fixture(root)
            output = root / "optimal"
            args = ["--stage1-summary", str(summary), "--turbines", str(turbines),
                    "--output", str(output), "--plot-dpi", "40"]
            with redirect_stdout(io.StringIO()):
                self.assertEqual(runner.main(args), 0)
            frozen = (output / "regions.json").read_bytes()
            report = json.loads((output / "reference_results.json").read_text())
            totals = json.loads((output / "summary.json").read_text())
            self.assertAlmostEqual(totals["total_flight_seconds"], sum(r["flight_seconds"] for r in report["results"]))
            self.assertAlmostEqual(totals["total_runtime_seconds"], sum(r["runtime_seconds"] for r in report["results"]))
            self.assertTrue(totals["all_regions_proven_optimal"])
            self.assertTrue((output / "region_01_optimal_route.png").is_file())
            self.assertTrue((output / "routes_overview.png").is_file())
            with redirect_stdout(io.StringIO()):
                self.assertEqual(runner.main([*args, "--resume"]), 0)
            resumed = json.loads((output / "summary.json").read_text())
            self.assertEqual(totals["total_runtime_seconds"], resumed["total_runtime_seconds"])
            self.assertEqual(frozen, (output / "regions.json").read_bytes())
            # A portable frozen input can resume without opening the native files.
            with redirect_stdout(io.StringIO()):
                self.assertEqual(runner.main(["--regions-json", str(output / "regions.json"), "--output",
                                              str(output), "--resume", "--plot-dpi", "40"]), 0)

    def test_default_input_is_stage1_and_solver_does_not_import_torch(self):
        import subprocess
        import sys

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch.object(runner, "DEFAULT_STAGE1_SUMMARY", root / "missing-summary.json"):
                with self.assertRaises(FileNotFoundError):
                    runner.main(["--output", str(root / "missing")])
            self.assertFalse((root / "missing").exists())
            summary, turbines = native_fixture(root)
            script = Path(runner.__file__)
            invocation = [sys.executable, "-c", "import runpy,sys; sys.modules['torch']=None; p=sys.argv.pop(1); sys.argv[0]=p; runpy.run_path(p,run_name='__main__')",
                          str(script), "--stage1-summary", str(summary), "--turbines", str(turbines),
                          "--output", str(root / "no-torch"), "--plot-dpi", "40"]
            result = subprocess.run(invocation, capture_output=True, timeout=30, check=False)
            self.assertEqual(result.returncode, 0, result.stderr.decode(errors="replace"))

    def test_plot_status_never_labels_unproven_route_optimal(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            summary, turbines = native_fixture(root)
            instance = inputs.load_stage1_input(summary, turbines)[0][0]
            route = (0, *range(1, instance.node_count), 0)
            for termination, proof, expected in (("OPTIMAL", True, "optimal"),
                                                  ("TIME_LIMIT", False, "best_known"),
                                                  ("INFEASIBLE", False, "no_route")):
                record = {"region": 1, "termination": termination, "proven_optimal": proof,
                          "route": None if expected == "no_route" else route,
                          "flight_seconds": 0.0, "runtime_seconds": 0.1}
                figure = plots.write_region_route(root / termination, 1, instance, record, dpi=40)
                self.assertIn(expected, figure.name)
                self.assertGreater(figure.stat().st_size, 100)

    def test_resume_refreshes_existing_best_known_figure(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            target = output / "region_01_best_known_route.png"
            target.write_bytes(b"outdated route")
            result = {"proven_optimal": False, "route": (0, 1, 0)}
            with patch.object(plots, "write_region_route") as draw:
                runner._region_figure(output, 1, object(), result, 40)
            draw.assert_called_once()


if __name__ == "__main__":
    unittest.main()
