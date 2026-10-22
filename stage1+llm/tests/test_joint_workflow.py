"""V5 CLI/export/resume acceptance with an explicitly mocked route generator."""

import contextlib
import dataclasses
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from test_joint import FakeDecoder
from stage1_llm.artifacts import atomic_json
from stage1_llm.client import DeepSeekSettings
from stage1_llm.domain import DeploymentConfig
from stage1_llm.joint_artifacts import export_joint_candidate, verify_selection
from stage1_llm.joint_cli import main
from stage1_llm.joint_planner import CheckedPlanner, InfrastructureError, REPO_ROOT
from stage1_llm.joint_search import JointSearch
from stage1_llm.joint_settings import load_joint_settings
from stage1_llm.joint_smoke import MockClient, GROUP


class JointWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.enterContext(contextlib.redirect_stdout(io.StringIO()))
        self.enterContext(contextlib.redirect_stderr(io.StringIO()))
        self.input = self.root / "tiny.tsv"
        self.input.write_text(
            "a\t0\t0\nb\t0.1\t0\nc\t0\t0.1\nd\t0.1\t0.1\n", encoding="utf-8"
        )
        self.config = self.root / "search-v5.json"
        raw = json.loads(
            (REPO_ROOT / "stage1+llm/config/search-v5.json").read_text(encoding="utf-8")
        )
        raw.update(
            target_input=str(self.input),
            protected_input=str(REPO_ROOT / "turbine87.tsv"),
            data_dir=str(self.root / "data"),
            training_virtual_fields=0,
            validation_virtual_fields=0,
            stage2_device="cpu",
            constraint_preflight=False,
        )
        atomic_json(self.config, raw)
        self.joint = load_joint_settings(self.config)
        self.planner = CheckedPlanner(FakeDecoder(), self.joint.search.deployment)
        self.planner.close = lambda: None
        self.outputs = self.root / "outputs"
        self.common = [
            "--search-config",
            str(self.config),
            "--outputs",
            str(self.outputs),
        ]
        self.llm = DeepSeekSettings("https://example.invalid", "offline_mock", "")

    def search(self, sources, budget=None):
        joint = dataclasses.replace(
            self.joint,
            search=dataclasses.replace(
                self.joint.search, evaluation_budget=budget or len(sources)
            ),
        )
        return JointSearch(
            joint, self.llm, self.outputs, self.planner, MockClient(sources)
        )

    def test_config_and_no_paid_initialization_on_model_failure(self):
        self.assertEqual(self.joint.device, "cpu")
        self.assertEqual(self.joint.search.archive_size, 6)
        with patch(
            "stage1_llm.joint_cli.PlannerService",
            side_effect=InfrastructureError("test loading fault"),
        ), patch(
            "stage1_llm.joint_cli.load_deepseek_settings",
            side_effect=AssertionError("no paid client before model"),
        ):
            self.assertEqual(main(["optimize", *self.common]), 2)
        raw = json.loads(self.config.read_text(encoding="utf-8"))
        raw["stage2_device"] = "invalid"
        atomic_json(self.config, raw)
        with self.assertRaises(ValueError):
            load_joint_settings(self.config)
        raw["stage2_device"] = "cpu"
        raw["search_version"] = 4
        atomic_json(self.config, raw)
        with self.assertRaises(ValueError):
            load_joint_settings(self.config)

    def test_cli_prepare_evaluate_benchmark_and_do_not_overwrite(self):
        self.assertEqual(main(["prepare-data", *self.common]), 0)
        candidate = self.root / "algorithm.py"
        candidate.write_text(GROUP, encoding="utf-8")
        with patch("stage1_llm.joint_cli.PlannerService", return_value=self.planner):
            args = [
                "evaluate",
                *self.common,
                "--candidate",
                str(candidate),
                "--input",
                str(self.input),
                "--run-id",
                "test",
            ]
            self.assertEqual(main(args), 0)
            self.assertEqual(main(args), 2)
            self.assertEqual(
                main(
                    [
                        "benchmark",
                        *self.common,
                        "--candidate",
                        str(candidate),
                        "--run-id",
                        "bench",
                    ]
                ),
                0,
            )
            self.assertEqual(
                main(
                    [
                        "benchmark",
                        *self.common,
                        "--candidate",
                        str(candidate),
                        "--run-id",
                        "bench",
                    ]
                ),
                2,
            )
        metrics = json.loads(
            (self.outputs / "test/model_metrics.json").read_text(encoding="utf-8")
        )
        self.assertEqual(metrics["candidate_count"], 1)
        self.assertEqual(metrics["results"][0]["route"][0], 0)

    def test_selected_export_runtime_model_hash_and_cli(self):
        root = self.search([GROUP]).run("select")
        with self.assertRaises(ValueError):
            export_joint_candidate(root)
        with self.assertRaises(ValueError):
            export_joint_candidate(root, "C9999", allow_mock=True)
        selected = export_joint_candidate(root, allow_mock=True)
        verify_selection(selected / "algorithm.py", self.planner.metadata)
        with patch("stage1_llm.joint_cli.PlannerService", return_value=self.planner):
            self.assertEqual(
                main(
                    [
                        "evaluate",
                        *self.common,
                        "--candidate",
                        str(selected / "algorithm.py"),
                        "--input",
                        str(self.input),
                        "--run-id",
                        "selected",
                    ]
                ),
                0,
            )
            self.assertEqual(
                main(
                    [
                        "evaluate",
                        *self.common,
                        "--candidate",
                        str(selected / "algorithm.py"),
                        "--input",
                        str(self.input),
                        "--timeout",
                        "3",
                        "--run-id",
                        "wrong-time",
                    ]
                ),
                2,
            )
        self.assertEqual(
            main(["list-candidates", *self.common, "--run-id", "select"]), 0
        )
        self.assertEqual(main(["export-best", *self.common, "--run-id", "select"]), 2)
        with self.assertRaises(ValueError):
            verify_selection(
                selected / "algorithm.py", {**self.planner.metadata, "changed": True}
            )
        runtime = selected / "runtime/stage1_llm/joint_tools.py"
        runtime.write_text(
            runtime.read_text(encoding="utf-8") + "\n# tampered\n", encoding="utf-8"
        )
        with self.assertRaises(ValueError):
            verify_selection(selected / "algorithm.py", self.planner.metadata)

    def test_resume_saved_source_after_infrastructure_failure(self):
        search = self.search([GROUP])
        with patch.object(
            search, "_evaluate", side_effect=InfrastructureError("test worker outage")
        ):
            with self.assertRaises(InfrastructureError):
                root = search.run("outage")
        root = self.outputs / "outage"
        state = json.loads((root / "search_state.json").read_text(encoding="utf-8"))
        self.assertEqual(state["pending"]["stage"], "evaluation")
        self.assertEqual(state["history"], [])
        resumed = self.search([], budget=1)
        resumed.run("outage", resume=True)
        self.assertEqual(resumed.client.calls, 0)
        self.assertEqual(len(resumed.history), 1)
        self.assertTrue(resumed.history[0]["evaluation"]["feasible"])
        changed = self.search([], budget=1)
        changed.planner = CheckedPlanner(
            FakeDecoder(),
            dataclasses.replace(DeploymentConfig(), wind_direction_degrees=0),
        )
        with self.assertRaises(ValueError):
            changed.run("outage", resume=True)

    def test_interrupted_unknown_provider_request_not_replayed(self):
        search = self.search([KeyboardInterrupt()])
        with patch.object(search.client, "generate", side_effect=KeyboardInterrupt):
            with self.assertRaises(KeyboardInterrupt):
                search.run("unknown")
        resumed = self.search([], budget=1)
        resumed.run("unknown", resume=True)
        self.assertEqual(resumed.client.calls, 0)
        self.assertEqual(resumed.history[0]["evaluation"]["status"], "generation_error")

    def test_cli_search_override_and_entrypoint_versions(self):
        import stage1_llm.entrypoint as entry

        search = self.search([GROUP])
        with patch(
            "stage1_llm.joint_cli.PlannerService", return_value=self.planner
        ), patch(
            "stage1_llm.joint_cli.load_deepseek_settings", return_value=self.llm
        ), patch(
            "stage1_llm.joint_cli.JointSearch", return_value=search
        ):
            self.assertEqual(
                main(
                    [
                        "optimize",
                        *self.common,
                        "--run-id",
                        "cli",
                        "--evaluation-budget",
                        "1",
                        "--stage2-device",
                        "cpu",
                        "--api-retries",
                        "0",
                        "--max-tokens",
                        "8192",
                    ]
                ),
                0,
            )
        with patch("stage1_llm.joint_cli.main", return_value=0) as current:
            self.assertEqual(
                entry.main(
                    [
                        "list-candidates",
                        "--outputs",
                        str(self.outputs),
                        "--run-id",
                        "cli",
                    ]
                ),
                0,
            )
            current.assert_called_once()
        with patch("stage1_llm.cli.main", return_value=0) as legacy:
            self.assertEqual(
                entry.main(
                    [
                        "optimize",
                        "--search-config",
                        str(REPO_ROOT / "stage1+llm/config/search-v4.json"),
                    ]
                ),
                0,
            )
            legacy.assert_called_once()
        legacy_dir = self.root / "legacy"
        legacy_dir.mkdir()
        atomic_json(legacy_dir / "selection.json", {"search_version": 4})
        with patch("stage1_llm.cli.main", return_value=0) as legacy:
            self.assertEqual(
                entry.main(
                    [
                        "evaluate",
                        "--candidate",
                        str(legacy_dir / "algorithm.py"),
                        "--input",
                        str(self.input),
                    ]
                ),
                0,
            )
            legacy.assert_called_once()

    def test_cli_check_model(self):
        from stage1_llm.joint_cli import main

        with patch(
            "stage1_llm.joint_cli.PlannerService", return_value=self.planner
        ), patch("stage1_llm.joint_planner.GreedyDecoder", return_value=FakeDecoder()):
            self.assertEqual(main(["check-model", *self.common]), 0)
        with patch(
            "stage1_llm.joint_cli.PlannerService", return_value=self.planner
        ), patch.object(
            self.planner,
            "region",
            return_value={
                "feasible": False,
                "diagnostics": [{"kind": "service_radius"}],
            },
        ):
            self.assertEqual(main(["check-model", *self.common]), 2)


if __name__ == "__main__":
    unittest.main()
