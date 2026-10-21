"""Entry points with spaces, resume protection, and saved-result comparison."""

from __future__ import annotations

from contextlib import redirect_stdout
import importlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from stage2.data import instance_to_record
from stage2.tests.test_pool import small_instance


runner = importlib.import_module("stage2.Branch and Bound.run")
comparison = importlib.import_module("stage2.Branch and Bound.compare")
HERE = Path(__file__).resolve().parents[1]


class CliTests(unittest.TestCase):
    def test_solve_compare_resume_and_changed_input(self):
        instance = small_instance("D", 0, nodes=6)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            regions = root / "test_regions.json"
            regions.write_text(json.dumps({"regions": [instance_to_record(instance)]}))
            output = root / "with spaces"
            arguments = ["--regions-json", str(regions), "--output", str(output)]
            with redirect_stdout(io.StringIO()) as log:
                self.assertEqual(runner.main(arguments), 0)
            self.assertIn("Region 1/1", log.getvalue())
            report_path = output / "reference_results.json"
            before = report_path.read_bytes()
            self.assertTrue(json.loads(before)["results"][0]["proven_optimal"])
            self.assertTrue((output / "summary.json").exists())
            self.assertTrue((output / "results.csv").exists())
            with self.assertRaises(FileExistsError):
                runner.main(arguments)
            with redirect_stdout(io.StringIO()):
                self.assertEqual(runner.main([*arguments, "--resume"]), 0)
            restored = json.loads(report_path.read_text())
            self.assertEqual(restored["results"], json.loads(before)["results"])
            model_record = {**restored["results"][0], "runtime_seconds": 0.001}
            model = root / "model.json"
            model.write_text(json.dumps({"schema_version": 1, "results": [model_record]}))
            with redirect_stdout(io.StringIO()) as table:
                self.assertEqual(comparison.main(["--model-results", str(model), "--reference-results",
                                                 str(report_path), "--output", str(root / "comparison")]), 0)
            self.assertIn("gap", table.getvalue())
            regions.write_text(regions.read_text() + "\n")
            with self.assertRaises(ValueError):
                runner.main([*arguments, "--resume"])

    def test_missing_input_does_not_create_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaises(FileNotFoundError):
                runner.main(["--regions-json", str(root / "missing.json"), "--output", str(root / "out")])
            self.assertFalse((root / "out").exists())

    def test_script_paths_and_cli_help_work_without_torch(self):
        for name in ("run.py", "compare.py"):
            result = subprocess.run([sys.executable, str(HERE / name), "--help"],
                                    capture_output=True, timeout=30, check=False)
            self.assertEqual(result.returncode, 0, result.stderr.decode(errors="replace"))


if __name__ == "__main__":
    unittest.main()
