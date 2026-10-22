"""Offline contracts for budgeted, trusted v4 candidate tools."""

import contextlib
import dataclasses
import io
import json
import tempfile
import unittest
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

from test_v3 import SOLO, GROUP, controller, settings_at
from stage1_llm.candidate import CandidateProposal
from stage1_llm.domain import DeploymentConfig, Turbine
from stage1_llm.evaluator import evaluate_proposal
from stage1_llm.sandbox import run_candidate


class TrustedToolTests(unittest.TestCase):
    def tools(self, points=None, **limits):
        from stage1_llm.candidate_tools import CandidateTools

        return CandidateTools(
            points or [Turbine("a", 0, 0), Turbine("b", 1, 0)],
            DeploymentConfig(),
            **limits
        )

    def test_direction_cache_and_audit_match(self):
        tools = self.tools()
        self.assertAlmostEqual(tools.directed_time("a", "b"), 1000 / 29)
        self.assertAlmostEqual(tools.directed_time("b", "a"), 1000 / 17)
        first = tools.region("a", ["a", "b"])
        self.assertEqual(first, tools.region("a", ["b", "a"]))
        self.assertEqual(tools.stats()["cache_hits"], 1)
        checked = evaluate_proposal(
            list(tools.points),
            DeploymentConfig(),
            CandidateProposal(["a"], {"a": "a", "b": "a"}),
        )
        self.assertAlmostEqual(
            first["proxy_flight_seconds"], checked.metrics.total_proxy_flight_seconds
        )
        self.assertAlmostEqual(
            first["energy_seconds"],
            checked.deployment.nests[0].equivalent_energy_seconds,
        )
        first["feasible"] = False
        self.assertTrue(tools.region("a", ["a", "b"])["feasible"])

    def test_invalid_members_and_structured_numeric_failure(self):
        tools = self.tools([Turbine("a", 0, 0), Turbine("b", 20, 0)])
        row = tools.region("a", ["a", "b"])
        self.assertFalse(row["feasible"])
        self.assertGreater(row["diagnostics"][0]["excess"], 0)
        self.assertFalse(tools.region("a", ["b"])["feasible"])
        self.assertFalse(tools.region("a", ["a", "missing"])["feasible"])

    def test_budget_and_explicit_best(self):
        from stage1_llm.candidate_tools import BudgetExceeded

        tools = self.tools(max_region_evaluations=1)
        tools.region("a", ["a"])
        with self.assertRaises(BudgetExceeded):
            tools.region("b", ["b"])
        tools = self.tools(max_move_attempts=1)
        tools.tick()
        with self.assertRaises(BudgetExceeded):
            tools.tick()
        tools = self.tools()
        proposal = {"nest_ids": ["a"], "assignments": {"a": "a", "b": "a"}}
        self.assertTrue(tools.remember(proposal))
        self.assertEqual(tools.best(), proposal)
        proposal["assignments"]["b"] = "b"
        self.assertNotEqual(tools.best(), proposal)

    def test_split_can_add_nests_without_one_move_fully_fixing_region(self):
        tools = self.tools([Turbine(str(i), i * 10.0, 0) for i in range(5)])
        result = tools.split_repair({str(i): "0" for i in range(5)})
        self.assertGreater(len(result["nest_ids"]), 1)
        self.assertTrue(tools.deployment(result)["feasible"])

    def test_wind_time_cannot_replace_conservative_energy(self):
        from stage1_llm.candidate_tools import CandidateTools

        points = [Turbine("a", 0, 0), Turbine("b", 1, 0)]
        tools = CandidateTools(points, DeploymentConfig(max_endurance_seconds=125))
        directed_energy = (
            tools.directed_time("a", "b") + tools.directed_time("b", "a") + 22.8
        )
        self.assertLess(directed_energy, 125)
        row = tools.region("a", ["a", "b"])
        self.assertFalse(row["feasible"])
        self.assertEqual(row["diagnostics"][0]["kind"], "endurance")

    def test_edge_cases_validation_and_soft_time(self):
        from stage1_llm.candidate_tools import CandidateTools, BudgetExceeded

        for points in (
            [],
            [Turbine("a", 0, 0), Turbine("a", 1, 0)],
            [Turbine("a", float("nan"), 0)],
        ):
            with self.assertRaises(ValueError):
                CandidateTools(points, DeploymentConfig())
        for kwargs in (
            {"soft_seconds": 0},
            {"max_move_attempts": 0},
            {"max_region_evaluations": -1},
        ):
            with self.assertRaises(ValueError):
                self.tools(**kwargs)
        tools = self.tools()
        with patch(
            "stage1_llm.candidate_tools.time.perf_counter",
            return_value=tools._start + 200,
        ):
            with self.assertRaises(BudgetExceeded):
                tools.tick()
        with self.assertRaises(ValueError):
            tools.tick(-1)
        singleton = self.tools([Turbine("a", 0, 0)])
        row = singleton.region("a", ["a"])
        self.assertEqual(row["proxy_flight_seconds"], 0)
        self.assertEqual(row["proxy_route_ids"], ["a", "a"])
        self.assertFalse(
            singleton.deployment({"nest_ids": ["a"], "assignments": {}})["feasible"]
        )
        self.assertFalse(singleton.remember({"nest_ids": [], "assignments": {}}))
        with self.assertRaises(ValueError):
            singleton.split_repair({})
        impossible = CandidateTools(
            [Turbine("a", 0, 0)], DeploymentConfig(max_endurance_seconds=1)
        )
        with self.assertRaisesRegex(ValueError, "singleton"):
            impossible.split_repair({"a": "a"})

    def test_phases_heartbeat_and_remaining(self):
        events = []
        tools = self.tools(observer=events.append)
        with patch(
            "stage1_llm.candidate_tools.time.perf_counter",
            return_value=tools._start + 1,
        ):
            tools.phase("relocation")
        self.assertEqual(events[0]["phase"], "relocation")
        with patch(
            "stage1_llm.candidate_tools.time.perf_counter",
            return_value=tools._start + 12,
        ):
            tools.distance("a", "b")
        self.assertEqual(len(events), 2)
        self.assertEqual(tools.remaining()["move_attempts"], 20000)

    def test_subprocess_tools_and_budget_return_only_explicit_incumbent(self):
        code = """from candidate_tools import remember, tick, region
def solve(turbines, config, seed):
    ids = [t.identifier for t in turbines]
    remember({'nest_ids': ids, 'assignments': {i:i for i in ids}})
    while True:
        tick()
"""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "algorithm.py"
            path.write_text(code, encoding="utf-8")
            telemetry = {}
            result = run_candidate(
                path,
                list(self.tools().points),
                DeploymentConfig(),
                1,
                5,
                limits={"max_move_attempts": 2},
                telemetry=telemetry,
            )
            self.assertEqual(len(result.nest_ids), 2)
            self.assertEqual(telemetry["termination"], "budget_exhausted")
            path.write_text(
                code.replace(
                    "    remember({'nest_ids': ids, 'assignments': {i:i for i in ids}})\n",
                    "",
                ),
                encoding="utf-8",
            )
            with self.assertRaises(RuntimeError):
                run_candidate(
                    path,
                    list(self.tools().points),
                    DeploymentConfig(),
                    1,
                    5,
                    limits={"max_move_attempts": 2},
                )


class WindowTests(unittest.TestCase):
    def test_window_forgets_counts_masks_and_restores(self):
        from stage1_llm.selection import SlidingWindowUCB

        bandit = SlidingWindowUCB(["a", "b"], window=2, exploration=0, seed=1)
        bandit.update("a", 1)
        bandit.update("b", -1)
        bandit.update("b", 0.5)
        self.assertEqual(bandit.counts["a"], 0)
        self.assertEqual(bandit.select_with_trace(eligible=["b"])[0], "b")
        restored = SlidingWindowUCB(["a", "b"], window=2, exploration=0, seed=1)
        restored.load_state_dict(bandit.state_dict())
        self.assertEqual(restored.select_with_trace(), bandit.select_with_trace())
        with self.assertRaises(ValueError):
            bandit.select_with_trace(eligible=[])


class V4SearchTests(unittest.TestCase):
    def test_constraint_preflight_stops_single_nest_trap(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            c = controller(
                settings_at(root, evaluation_budget=1, constraint_preflight=True),
                root,
                [GROUP],
            )
            with contextlib.redirect_stdout(io.StringIO()):
                c.run("trap")
            row = c.history[0]
            self.assertFalse(c.archive)
            self.assertEqual(row["evaluation"]["failure_stage"], "constraint_preflight")
            self.assertTrue(row["evaluation"]["diagnostics"])
            self.assertEqual(row["evaluation"]["datasets"], [])

    def test_early_failure_and_generation_credit_separate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            settings = settings_at(
                root,
                evaluation_budget=2,
                selector="sliding_window",
                early_rejection=True,
            )
            c = controller(settings, root, [ValueError("provider failed"), SOLO])
            with contextlib.redirect_stdout(io.StringIO()):
                c.run("credit")
            self.assertEqual(
                c.bandit.total, 0
            )  # cold start and forced repair are not optimization evidence
            self.assertTrue(c.archive)
            self.assertEqual(c.history[0]["bandit_credit"], "excluded_generation_error")

    def test_partial_failure_resume_cannot_skip_into_feasible_archive(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            settings = settings_at(root, evaluation_budget=1, early_rejection=True)
            settings.target_input.write_text("a\t0\t0\nb\t20\t0\n", encoding="utf-8")
            c = controller(settings, root, [GROUP])
            with contextlib.redirect_stdout(io.StringIO()):
                c.run("failure")
            self.assertFalse(c.archive)
            self.assertTrue(c.history[0]["evaluation"]["diagnostics"])
            self.assertLess(c.history[0]["reward"], 0)

    def test_export_bundle_runs_without_repository_and_detects_tampering(self):
        from stage1_llm.exports import export_candidate
        from stage1_llm.sandbox import _runtime
        from stage1_llm.smoke import TOOL_GROUP

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            settings = settings_at(
                root,
                evaluation_budget=1,
                selector="sliding_window",
                constraint_preflight=True,
            )
            c = controller(settings, root, [TOOL_GROUP])
            with contextlib.redirect_stdout(io.StringIO()):
                out = c.run("export")
            target = export_candidate(out, "C0001", allow_mock=True)
            proc = subprocess.run(
                [
                    sys.executable,
                    "-I",
                    str(target / "run_selected.py"),
                    "--input",
                    str(settings.target_input),
                ],
                capture_output=True,
                text=True,
                encoding="utf-8",
            )
            self.assertEqual(proc.returncode, 0, proc.stderr)
            self.assertTrue(json.loads(proc.stdout)["evaluation"]["feasible"])
            (target / "runtime/stage1_llm/domain.py").write_text(
                "changed", encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "hash"):
                _runtime(target / "algorithm.py")

    def test_benchmark_has_no_api_and_no_silent_overwrite(self):
        from stage1_llm.benchmark import benchmark

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            settings = settings_at(root)
            candidate = root / "candidate.py"
            candidate.write_text(SOLO, encoding="utf-8")
            with contextlib.redirect_stdout(io.StringIO()), patch(
                "stage1_llm.client.DeepSeekClient",
                side_effect=AssertionError("paid API forbidden"),
            ):
                result = benchmark(settings, root / "outputs", "bench", [candidate])
            data = json.loads((result / "benchmark.json").read_text(encoding="utf-8"))
            self.assertTrue(data["complete"])
            self.assertTrue(data["results"][0]["feasible"])
            with self.assertRaises(FileExistsError):
                benchmark(settings, root / "outputs", "bench", [candidate])

    def test_preflight_soft_budget_clamps_to_hard_deadline(self):
        from stage1_llm.candidate_tools import CandidateTools

        code = """from candidate_tools import remaining
def solve(turbines, config, seed):
    if remaining()['seconds'] > 1.6:
        raise ValueError('soft budget exceeds preflight deadline')
    return {'nest_ids':[t.identifier for t in turbines], 'assignments':{t.identifier:t.identifier for t in turbines}}
"""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "candidate.py"
            path.write_text(code, encoding="utf-8")
            result = run_candidate(
                path,
                [Turbine("a", 0, 0)],
                DeploymentConfig(),
                1,
                2,
                limits={"soft_seconds": 120},
            )
            self.assertEqual(result.nest_ids, ["a"])

    def test_selected_nondefault_budget_source_hash_and_cli_environment(self):
        from stage1_llm.exports import export_candidate
        from stage1_llm import cli

        code = """from candidate_tools import remember, tick
def solve(turbines, config, seed):
    ids = [t.identifier for t in turbines]
    remember({'nest_ids':ids,'assignments':{i:i for i in ids}})
    while True:
        tick()
"""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            settings = settings_at(root, evaluation_budget=1, max_move_attempts=7)
            c = controller(settings, root, [code])
            with contextlib.redirect_stdout(io.StringIO()):
                out = c.run("limits")
            selected = export_candidate(out, "C0001", allow_mock=True)
            command = [
                sys.executable,
                "-I",
                str(selected / "run_selected.py"),
                "--input",
                str(settings.target_input),
            ]
            proc = subprocess.run(
                command, capture_output=True, text=True, encoding="utf-8"
            )
            self.assertEqual(proc.returncode, 0, proc.stderr)
            data = json.loads(proc.stdout)
            self.assertEqual(data["telemetry"]["move_attempts"], 7)
            self.assertEqual(data["origin"], "offline_mock")
            algorithm = selected / "algorithm.py"
            algorithm.write_text(code + "\n# changed", encoding="utf-8")
            proc = subprocess.run(
                command, capture_output=True, text=True, encoding="utf-8"
            )
            self.assertNotEqual(proc.returncode, 0)
            self.assertIn("source hash changed", proc.stderr)
            algorithm.write_text(code, encoding="utf-8")
            raw = dataclasses.asdict(settings)
            raw["deployment"]["wind_speed_mps"] = 5
            config = root / "config.json"
            config.write_text(json.dumps(raw, default=str), encoding="utf-8")
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(
                io.StringIO()
            ) as errors:
                status = cli.main(
                    [
                        "evaluate",
                        "--search-config",
                        str(config),
                        "--outputs",
                        str(root / "eval"),
                        "--run-id",
                        "mismatch",
                        "--candidate",
                        str(algorithm),
                        "--input",
                        str(settings.target_input),
                    ]
                )
            self.assertEqual(status, 1)
            self.assertIn("environment differs", errors.getvalue())
            raw["deployment"]["wind_speed_mps"] = 6
            config.write_text(json.dumps(raw, default=str), encoding="utf-8")
            with contextlib.redirect_stdout(io.StringIO()):
                status = cli.main(
                    [
                        "evaluate",
                        "--search-config",
                        str(config),
                        "--outputs",
                        str(root / "eval"),
                        "--run-id",
                        "correct",
                        "--candidate",
                        str(algorithm),
                        "--input",
                        str(settings.target_input),
                    ]
                )
            self.assertEqual(status, 0)
            self.assertTrue((root / "eval/correct/regions.json").is_file())

    def test_early_failure_has_explicit_partial_coverage_and_failed_time(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            validation = root / "validation.tsv"
            validation.write_text("v\t1\t1\n", encoding="utf-8")
            settings = settings_at(
                root,
                validation_inputs=(validation,),
                evaluation_budget=1,
                early_rejection=True,
            )
            settings.target_input.write_text("a\t0\t0\nb\t20\t0\n", encoding="utf-8")
            c = controller(settings, root, [GROUP])
            with contextlib.redirect_stdout(io.StringIO()):
                c.run("partial")
            result = c.history[0]["evaluation"]
            self.assertFalse(result["complete"])
            self.assertEqual((result["evaluated_count"], result["total_count"]), (1, 2))
            self.assertNotIn("behavior_sha256", result)
            self.assertFalse(result["duplicate"])

    def test_v4_config_and_invalid_settings(self):
        from stage1_llm.settings import load_search_settings

        settings = load_search_settings(
            Path(__file__).parents[1] / "config/search-v4.json"
        )
        self.assertEqual(settings.selector, "sliding_window")
        self.assertTrue(settings.constraint_preflight)
        for changes in (
            {"selector": "mcts"},
            {"ucb_window": 0},
            {"soft_seconds": float("nan")},
            {"early_rejection": 1},
            {"feedback_dataset_limit": 0},
        ):
            with self.assertRaises(ValueError):
                dataclasses.replace(settings, **changes)


if __name__ == "__main__":
    unittest.main()
