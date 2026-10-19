"""CLI defaults and split-output isolation tests."""

from __future__ import annotations

import unittest
from contextlib import redirect_stderr
import io
import json
from pathlib import Path
import tempfile
from unittest.mock import patch

from stage2.run import ROOT, _resolve_input_path, _run_search, main


class RunCliTests(unittest.TestCase):
    def test_training_and_evaluation_default_to_separate_datasets(self) -> None:
        self.assertEqual(_resolve_input_path("train", None), ROOT.parent / "turbine102.tsv")
        self.assertEqual(_resolve_input_path("evaluate", None), ROOT.parent / "turbine87.tsv")

    def test_explicit_input_path_is_preserved(self) -> None:
        explicit = Path("custom-test.tsv")
        self.assertEqual(_resolve_input_path("evaluate", explicit), explicit)

    def test_search_rejects_existing_results_before_preparing_data(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "search").mkdir()
            marker = root / "search/existing.json"
            marker.write_text("original", encoding="utf-8")
            with patch("sys.argv", ["stage2.run", "search", "--output", str(root)]), \
                    patch("stage2.run._prepare_or_load_dataset") as prepare, redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    main()
            prepare.assert_not_called()
            self.assertEqual(marker.read_text(encoding="utf-8"), "original")

    def test_legacy_search_still_screens_twelve_and_repeats_three_seeds(self) -> None:
        import torch

        from stage2.tests.test_pool import small_instance
        from stage2.train import TrainingConfig

        def fake_train(instances, config, output, validation):
            output.mkdir(parents=True, exist_ok=True)
            checkpoint = output / "best.pt"
            torch.save({"best_validation": {"feasible_rate": 1.0,
                                           "mean_normalized_flight_cost": config.learning_rate}}, checkpoint)
            return checkpoint

        instances = [small_instance("real", 0.0), small_instance("virtual-00000-000", 10.0),
                     small_instance("virtual-00033-000", 20.0)]
        with tempfile.TemporaryDirectory() as directory, patch("stage2.train.train", side_effect=fake_train) as trainer:
            checkpoint = _run_search(instances, instances[:1], TrainingConfig(device="cpu"), Path(directory))
            self.assertTrue(checkpoint.is_file())
            self.assertEqual(trainer.call_count, 21)
            summary = json.loads((Path(directory) / "search/search_results.json").read_text())
            self.assertEqual(len(summary["screening"]), 12)
            self.assertEqual(len(summary["top_three_replicates"]), 9)
            self.assertEqual({row["seed"] for row in summary["top_three_replicates"]}, {1234, 1235, 1236})

    def test_cli_train_resume_and_validation_reference_evaluation(self) -> None:
        from itertools import permutations

        from stage2.benchmark import parse_gurobi_solution
        from stage2.tests.test_pool import small_instance

        training = [small_instance("real", 0.0), small_instance("virtual-00000-000", 1234.0)]
        validation = [small_instance("validation", 30.0, nodes=4)]
        instance = validation[0]
        optimum = min((instance.evaluate_route((0, *order, 0)) for order in permutations((1, 2, 3))),
                      key=lambda item: item.flight_seconds)
        # A synthetic solver response for interface testing, not a real Gurobi experiment.
        solution = {"SolutionInfo": {"Status": 2, "Runtime": 0.0, "ObjVal": optimum.flight_seconds,
                                     "ObjBound": optimum.flight_seconds, "MIPGap": 0.0},
                    "Vars": [{"VarName": f"x_{i}_{j}", "X": 1.0}
                             for i, j in zip(optimum.route, optimum.route[1:])]}
        reference = parse_gurobi_solution(instance, solution)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / "config.json"
            config.write_text(json.dumps({"device": "cpu", "seed": 1234, "hidden_dim": 8,
                                          "encoder_layers": 1, "heads": 2, "batch_size": 2,
                                          "epochs": 1, "batches_per_epoch": 1, "virtual_fields": 1,
                                          "validation_fields": 1, "refresh_every_epochs": 0}), encoding="utf-8")
            references = root / "reference_results.json"
            references.write_text(json.dumps({"schema_version": 1, "results": [reference]}), encoding="utf-8")
            common = ["--config", str(config), "--data-dir", str(root / "data"),
                      "--output", str(root / "training"), "--reference-results", str(references)]
            with patch("stage2.run.build_training_validation_instances", return_value=(training, validation)) as builder:
                with patch("sys.argv", ["stage2.run", "train", *common]):
                    self.assertEqual(main(), 0)
                with patch("sys.argv", ["stage2.run", "resume", *common, "--epochs", "2",
                                        "--checkpoint", str(root / "training/latest.pt")]):
                    self.assertEqual(main(), 0)
                self.assertEqual(builder.call_count, 1)
            source = root / "data/validation_regions.json"
            original = source.read_bytes()
            evaluate = ["stage2.run", "evaluate", "--config", str(config), "--regions-json", str(source),
                        "--checkpoint", str(root / "training/best.pt"), "--reference-results", str(references),
                        "--random-samples", "2", "--plot-dpi", "40"]
            with patch("sys.argv", [*evaluate, "--evaluation-output", str(root / "evaluation")]):
                self.assertEqual(main(), 0)
            summary = json.loads((root / "evaluation/evaluation_summary.json").read_text())
            self.assertEqual(summary["evaluation_split"], "validation")
            self.assertEqual(summary["reference_comparison"]["optimality_gap_count"], 1.0)
            self.assertEqual(original, source.read_bytes())
            with patch("sys.argv", [*evaluate, "--evaluation-output", str(root / "data")]), redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    main()
            self.assertEqual(original, source.read_bytes())


if __name__ == "__main__":
    unittest.main()
