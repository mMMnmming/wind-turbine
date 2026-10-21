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
    def test_explicit_regions_keep_their_environment_and_reject_checkpoint_override(self) -> None:
        from dataclasses import asdict
        from stage2.tests.test_pool import small_instance
        from stage2.run import _resolve_evaluation_environment

        instance = small_instance("D", 0)
        saved = asdict(instance.config)
        self.assertIs(_resolve_evaluation_environment([instance], saved, frozen_input=True)[0], instance)
        wrong = {**saved, "wind_speed_mps": 3.0}
        with self.assertRaises(ValueError):
            _resolve_evaluation_environment([instance], wrong, frozen_input=True)
        updated = _resolve_evaluation_environment([instance], wrong, frozen_input=False)
        self.assertEqual(updated[0].config.wind_speed_mps, 3)
        with self.assertLogs("stage2.run", level="WARNING"):
            self.assertIs(_resolve_evaluation_environment([instance], {}, frozen_input=True)[0], instance)

    def test_raw_model_totals_include_unsafe_routes(self) -> None:
        from stage2.run import _model_totals

        records = [{"flight_seconds": 100.0, "runtime_seconds": 0.01, "feasible": True},
                   {"flight_seconds": 3000.0, "runtime_seconds": 0.02, "feasible": False}]
        totals = _model_totals(records, 2)
        self.assertEqual(totals["total_flight_seconds"], 3100)
        self.assertAlmostEqual(totals["total_runtime_seconds"], 0.03)
        self.assertEqual(totals["infeasible_region_count"], 1)
        self.assertFalse(totals["all_routes_feasible"])
        self.assertIsNone(_model_totals(records[:1], 2)["total_flight_seconds"])

    def test_zero_task_frozen_region_needs_no_network_forward(self) -> None:
        from dataclasses import asdict
        import torch
        from stage2.atsp import WindATSPConfig, WindATSPInstance
        from stage2.data import instance_to_record
        from stage2.model import GraphPointerNetwork
        import numpy as np

        instance = WindATSPInstance(("D",), np.zeros((1, 2)), 0, WindATSPConfig())
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "regions.json"
            source.write_text(json.dumps({"frozen_reference": True, "source": {"communication_radius_km": 3},
                                          "regions": [instance_to_record(instance)]}))
            checkpoint = root / "best.pt"
            model = GraphPointerNetwork(8, 1, 2)
            torch.save({"training_config": {"hidden_dim": 8, "encoder_layers": 1, "heads": 2},
                        "model": model.state_dict(), "wind_environment": asdict(instance.config)}, checkpoint)
            from stage2.evaluation import write_evaluation_artifacts
            with patch("sys.argv", ["stage2.run", "evaluate", "--device", "cpu", "--regions-json", str(source),
                                    "--checkpoint", str(checkpoint), "--evaluation-output", str(root / "evaluation"),
                                    "--random-samples", "1", "--plot-dpi", "40"]), \
                    patch("stage2.run._timed_greedy_route", side_effect=AssertionError("No tasks")), \
                    patch("stage2.evaluation.write_evaluation_artifacts", wraps=write_evaluation_artifacts) as artifacts:
                self.assertEqual(main(), 0)
            metrics = json.loads((root / "evaluation/model_metrics.json").read_text())
            self.assertEqual(metrics["summary"]["total_flight_seconds"], 0)
            self.assertEqual(metrics["summary"]["total_runtime_seconds"], 0)
            self.assertEqual(artifacts.call_args.kwargs["service_radius_km"], 3)

    def test_inference_timer_synchronizes_cuda_and_includes_host_transfer(self) -> None:
        from types import SimpleNamespace
        from unittest.mock import Mock
        from stage2.run import _timed_greedy_route

        events = []
        features = SimpleNamespace(device=SimpleNamespace(type="cuda"))
        row = Mock()
        row.cpu.side_effect = lambda: events.append("transfer") or row
        row.tolist.return_value = [2, 1]
        model = Mock(side_effect=lambda *args, **kwargs: events.append("forward") or ([row], None))
        with patch("torch.cuda.synchronize", side_effect=lambda *args: events.append("sync")), \
                patch("stage2.run.time.perf_counter", side_effect=lambda: events.append("clock") or len(events)):
            route, runtime = _timed_greedy_route(model, features)
        self.assertEqual(route, (0, 2, 1, 0))
        self.assertGreater(runtime, 0)
        self.assertEqual(events, ["sync", "clock", "forward", "transfer", "sync", "clock"])

    def test_default_configuration_uses_selected_search_parameters(self) -> None:
        from stage2.train import TrainingConfig

        config = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
        self.assertEqual(
            {name: config[name] for name in ("learning_rate", "virtual_fields", "energy_weight")},
            {"learning_rate": 1e-4, "virtual_fields": 64, "energy_weight": 0.25},
        )
        self.assertEqual(TrainingConfig().learning_rate, config["learning_rate"])
        self.assertEqual(TrainingConfig().energy_weight, config["energy_weight"])
        self.assertEqual(config["refresh_fields"], config["virtual_fields"] // 4)
        self.assertEqual(config["refresh_every_epochs"], 5)
        self.assertEqual((config["seed"], config["data_seed"]), (1234, 1234))

    def test_missing_virtual_field_setting_falls_back_to_sixty_four(self) -> None:
        from stage2.tests.test_pool import small_instance

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / "config.json"
            config.write_text(json.dumps({"device": "cpu", "seed": 1234}), encoding="utf-8")
            manifest = {"input_hashes": {}, "kde_generation": {"attempts_by_split": {}}}
            with patch("stage2.run._prepare_or_load_dataset",
                       return_value=([small_instance("train", 0.0)], [small_instance("val", 10.0)], manifest)) as prepare, \
                    patch("sys.argv", ["stage2.run", "prepare_data", "--config", str(config), "--output", str(root)]):
                self.assertEqual(main(), 0)
            self.assertEqual(prepare.call_args.args[4], 64)

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
            timing = json.loads((root / "evaluation/model_metrics.json").read_text())
            self.assertEqual(timing["device"], "cpu")
            self.assertEqual(len(timing["results"]), 1)
            self.assertGreater(timing["results"][0]["runtime_seconds"], 0)
            self.assertAlmostEqual(timing["summary"]["total_flight_seconds"],
                                   sum(row["flight_seconds"] for row in timing["results"]))
            self.assertAlmostEqual(timing["summary"]["total_runtime_seconds"],
                                   sum(row["runtime_seconds"] for row in timing["results"]))
            self.assertEqual(timing["summary"]["region_count"], 1)
            self.assertIn("instance_fingerprint", timing["results"][0])
            self.assertAlmostEqual(timing["results"][0]["flight_seconds"],
                                   summary["method_summary"]["model_greedy"]["mean_flight_seconds"])
            self.assertEqual(original, source.read_bytes())
            with patch("sys.argv", [*evaluate, "--evaluation-output", str(root / "data")]), redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    main()
            self.assertEqual(original, source.read_bytes())


if __name__ == "__main__":
    unittest.main()
