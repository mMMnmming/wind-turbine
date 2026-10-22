"""Quality regressions; reference heuristics are not LLM experiment results."""

import contextlib
import base64
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest

from test_v3 import controller, settings_at
from stage1_llm import candidate_tools
from stage1_llm.candidate_tools import CandidateTools, ToolContractError
from stage1_llm.domain import DeploymentConfig, Turbine
from stage1_llm.sandbox import run_candidate


REPAIR_ONLY = """from candidate_tools import repartition_repair, remember
def solve(turbines,config,seed):
    p = repartition_repair({t.identifier:turbines[0].identifier for t in turbines})
    remember(p)
    return p
"""


class QualityTests(unittest.TestCase):
    def tools(self, **kwargs):
        points = [Turbine(str(i), i * 0.1, 0) for i in range(12)]
        return CandidateTools(
            points, DeploymentConfig(max_endurance_seconds=70), **kwargs
        )

    def test_repartition_packs_tasks_instead_of_only_opening_singletons(self):
        assignments = {str(i): "0" for i in range(12)}
        old = self.tools().split_repair(assignments)
        tools = self.tools()
        new = tools.repartition_repair(assignments)
        checked = tools.deployment(new)
        self.assertTrue(checked["feasible"])
        self.assertLess(len(new["nest_ids"]), len(old["nest_ids"]))
        self.assertEqual(assignments, {str(i): "0" for i in range(12)})
        self.assertTrue(all(new["assignments"][n] == n for n in new["nest_ids"]))
        again = self.tools().repartition_repair(assignments)
        self.assertEqual(new, again)

    def test_repair_validation_budget_and_physically_impossible_singleton(self):
        from stage1_llm.candidate_tools import BudgetExceeded

        tools = self.tools()
        with self.assertRaises(ToolContractError):
            tools.repartition_repair({"0": "0"})
        with self.assertRaises(BudgetExceeded):
            self.tools(max_move_attempts=1).repartition_repair(
                {str(i): "0" for i in range(12)}
            )
        impossible = CandidateTools(
            [Turbine("a", 0, 0)], DeploymentConfig(max_endurance_seconds=1)
        )
        with self.assertRaisesRegex(ValueError, "singleton"):
            impossible.repartition_repair({"a": "a"})
        isolated = CandidateTools(
            [Turbine("a", 0, 0), Turbine("b", 30, 0)], DeploymentConfig()
        )
        p = isolated.repartition_repair({"a": "a", "b": "a"})
        self.assertEqual(len(p["nest_ids"]), 2)

    def test_explicit_preference_can_trade_flight_for_smaller_regions(self):
        tools = self.tools()
        old = [0.1, 10, 0.5]
        new = [0.1, 11, 0.2]
        self.assertFalse(tools.objective_dominates(old, new))
        self.assertFalse(tools.objective_dominates(new, old))
        self.assertTrue(
            tools.prefer_objectives(new, old, weights=[1, 1, 2], scales=[1, 10, 1])
        )
        self.assertFalse(
            tools.prefer_objectives(new, old, weights=[1, 100, 1], scales=[1, 10, 1])
        )
        self.assertFalse(
            tools.prefer_objectives(old, old, weights=[1, 1, 1], scales=[1, 1, 1])
        )
        self.assertTrue(tools.objective_dominates([0.1, 9, 0.2], old))
        for kwargs in (
            {"weights": [0, 0, 0], "scales": [1, 1, 1]},
            {"weights": [1, 1, 1], "scales": [1, 0, 1]},
        ):
            with self.assertRaises(ToolContractError):
                tools.prefer_objectives(new, old, **kwargs)

    def test_reference_matches_public_signatures_and_has_targeted_distance_help(self):
        text = candidate_tools.tool_reference()
        for name in candidate_tools.PUBLIC_METHODS:
            self.assertIn(name + "(", text)
        narrow = candidate_tools.tool_reference(["distance"])
        self.assertIn("identifier", narrow)
        self.assertIn("坐标", narrow)
        self.assertNotIn("split_repair(", narrow)

    def test_quality_is_diagnostic_not_a_fourth_objective(self):
        q = candidate_tools.deployment_quality([40] + [0] * 61, 102, {})
        self.assertEqual(q["zero_task_regions"], 61)
        self.assertAlmostEqual(q["zero_task_fraction"], 61 / 62)
        self.assertIn("zero_task_regions_present", q["flags"])
        self.assertEqual(q["repair_observability"], "unavailable")
        self.assertEqual(
            candidate_tools.deployment_quality([0], 1, {})["task_imbalance_cv"], 0
        )

    def test_partial_summary_preserves_expected_coverage_and_unavailable_values(self):
        rows = [
            {
                "name": "ok",
                "split": "training",
                "feasible": True,
                "metrics": {
                    "nest_count": 1,
                    "total_proxy_flight_seconds": 10,
                    "maximum_region_tasks": 1,
                },
                "quality": candidate_tools.deployment_quality([1], 2, {}),
            },
            {"name": "bad", "split": "training", "feasible": False, "metrics": None},
        ]
        summary = candidate_tools.quality_summary(
            rows, {"training": 2, "validation": 1}
        )
        self.assertEqual(summary["all"]["covered_instances"], 2)
        self.assertEqual(summary["all"]["expected_instances"], 3)
        self.assertEqual(summary["all"]["feasible_instances"], 1)
        self.assertEqual(summary["validation"]["covered_instances"], 0)
        self.assertIsNone(summary["validation"]["mean_nest_count"])

    def test_objective_contract_and_numeric_overflow_are_not_silently_accepted(self):
        for a in ([1, 2], [True, 1, 2], [1, float("nan"), 2], [10**1000, 0, 1]):
            with self.assertRaises(ToolContractError):
                self.tools().objective_dominates(a, [1, 2, 3])
        with self.assertRaises(ToolContractError):
            self.tools().prefer_objectives(
                [0, 1e308, 0], [1, 0, 1], weights=[1, 1, 1], scales=[1, 1e-300, 1]
            )

    def test_budget_exit_does_not_count_an_unfinished_external_evaluation(self):
        code = REPAIR_ONLY.replace(
            "repartition_repair, remember", "repartition_repair, remember, deployment"
        ).replace("    return p", "    deployment(p,proxy=True)\n    return p")
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "candidate.py"
            source.write_text(code, encoding="utf-8")
            telemetry = {}
            run_candidate(
                source,
                [Turbine("a", 0, 0), Turbine("b", 0.1, 0)],
                DeploymentConfig(),
                1,
                5,
                limits={"max_region_evaluations": 1},
                telemetry=telemetry,
            )
            self.assertEqual(telemetry["termination"], "budget_exhausted")
            self.assertEqual(telemetry["post_repair_evaluation_calls"], 0)
            q = candidate_tools.deployment_quality([1], 2, telemetry)
            self.assertIn("repair_exit_without_external_checks", q["flags"])

    def test_saved_changes_do_not_count_reordering_or_repeated_remember(self):
        tools = self.tools()
        p = tools.repartition_repair({str(i): "0" for i in range(12)})
        tools.remember(p)
        self.assertEqual(tools.stats()["changed_feasible_saves_after_repair"], 0)
        changed = tools.make_proposal({str(i): str(i) for i in range(12)})
        tools.remember(changed)
        tools.remember(changed)
        changed["nest_ids"].reverse()
        tools.remember(changed)
        self.assertEqual(tools.stats()["changed_feasible_saves_after_repair"], 1)

    def test_oversized_integer_contract_error_cannot_be_swallowed(self):
        from stage1_llm.sandbox import CandidateExecutionError

        code = """from candidate_tools import objective_dominates
def solve(turbines,config,seed):
    try: objective_dominates([10**1000,0,1],[0,1,0])
    except Exception: pass
    return {'nest_ids':['a'],'assignments':{'a':'a'}}
"""
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "candidate.py"
            source.write_text(code, encoding="utf-8")
            with self.assertRaises(CandidateExecutionError):
                run_candidate(source, [Turbine("a", 0, 0)], DeploymentConfig(), 1, 5)

    def test_real_trial_source_and_distance_failure_are_losslessly_preserved(self):
        from stage1_llm.sandbox import CandidateExecutionError

        fixture = json.loads(
            (Path(__file__).parent / "fixtures/trial_quality_failures.json").read_text(
                encoding="utf-8"
            )
        )
        for record in fixture["sources"]:
            raw = base64.b64decode(record["source_base64"])
            self.assertEqual(hashlib.sha256(raw).hexdigest(), record["original_sha256"])
            if record["filename"] != "0001.py":
                baseline = (
                    Path(__file__).parents[1]
                    / "baselines/trial-contract"
                    / ("C" + record["filename"])
                )
                self.assertEqual(
                    baseline.read_text(encoding="utf-8").strip(),
                    raw.decode("utf-8").replace("\r\n", "\n").strip(),
                )
            else:
                with tempfile.TemporaryDirectory() as directory:
                    path = Path(directory) / "candidate.py"
                    path.write_bytes(raw)
                    with self.assertRaises(CandidateExecutionError) as error:
                        run_candidate(
                            path,
                            [Turbine("a", 0, 0), Turbine("b", 0.1, 0)],
                            DeploymentConfig(),
                            1,
                            5,
                        )
                    self.assertEqual(
                        error.exception.telemetry["contract_errors"][0]["tool"],
                        "distance",
                    )
                    self.assertEqual(
                        error.exception.telemetry["contract_errors"][0][
                            "candidate_line"
                        ],
                        14,
                    )

    def test_distance_feedback_is_targeted_and_includes_the_actual_call_site(self):
        bad = """from candidate_tools import distance
def solve(turbines,config,seed):
    distance((0,0),(1,1))
"""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            c = controller(
                settings_at(root, evaluation_budget=2), root, [bad, REPAIR_ONLY]
            )
            with contextlib.redirect_stdout(io.StringIO()):
                c.run("targeted")
            feedback = json.loads(
                c.client.prompts[1]
                .split("固定评测诊断：")[1]
                .split("\n父代完整源码：")[0]
            )
            self.assertIn("distance(", feedback["targeted_tool_reference"])
            self.assertIn("identifier", feedback["targeted_tool_reference"])
            self.assertNotIn("split_repair(", feedback["targeted_tool_reference"])
            self.assertIn("distance((0,0)", str(feedback["error_context"]))

    def test_benchmark_has_quality_diagnostics_without_llm_calls(self):
        from stage1_llm.benchmark import benchmark

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            settings = settings_at(root)
            source = root / "repair.py"
            source.write_text(REPAIR_ONLY, encoding="utf-8")
            with contextlib.redirect_stdout(io.StringIO()):
                path = benchmark(settings, root / "outputs", "bench", [source])
            result = json.loads((path / "benchmark.json").read_text(encoding="utf-8"))[
                "results"
            ][0]
            self.assertTrue(result["feasible"])
            self.assertIn(
                "repair_return_without_external_checks",
                result["datasets"][0]["quality"]["flags"],
            )
            self.assertEqual(result["quality_summary"]["all"]["covered_instances"], 1)

    def test_subprocess_distinguishes_repair_return_from_observed_followup(self):
        for code, expected in (
            (REPAIR_ONLY, 0),
            (
                REPAIR_ONLY.replace(
                    "    return p", "    deployment(p,proxy=True)\n    return p"
                ).replace(
                    "repartition_repair, remember",
                    "repartition_repair, remember, deployment",
                ),
                1,
            ),
        ):
            with self.subTest(
                expected=expected
            ), tempfile.TemporaryDirectory() as directory:
                source = Path(directory) / "candidate.py"
                source.write_text(code, encoding="utf-8")
                telemetry = {}
                run_candidate(
                    source,
                    [Turbine("a", 0, 0), Turbine("b", 0.1, 0)],
                    DeploymentConfig(),
                    1,
                    5,
                    telemetry=telemetry,
                )

                self.assertEqual(telemetry["repair_calls"]["repartition_repair"], 1)
                self.assertEqual(telemetry["post_repair_evaluation_calls"], expected)
                q = candidate_tools.deployment_quality([1], 2, telemetry)
                self.assertEqual(
                    "repair_return_without_external_checks" in q["flags"], expected == 0
                )

    def test_current_worker_can_run_an_immutable_v2_tools_snapshot(self):
        from stage1_llm.exports import snapshot_runtime
        from test_v3 import SOLO

        fixture = json.loads(
            (Path(__file__).parent / "fixtures/legacy_tools_v2.json").read_text(
                encoding="utf-8"
            )
        )
        raw = base64.b64decode(fixture["source_base64"])
        self.assertEqual(hashlib.sha256(raw).hexdigest(), fixture["sha256"])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            hashes = snapshot_runtime(root)
            (root / "runtime/stage1_llm/candidate_tools.py").write_bytes(raw)
            hashes["candidate_tools.py"] = fixture["sha256"]
            source = root / "algorithm.py"
            source.write_text(SOLO, encoding="utf-8")
            (root / "selection.json").write_text(
                json.dumps(
                    {
                        "source_sha256": hashlib.sha256(SOLO.encode()).hexdigest(),
                        "runtime_hashes": hashes,
                    }
                ),
                encoding="utf-8",
            )
            telemetry = {}
            result = run_candidate(
                source,
                [Turbine("a", 0, 0)],
                DeploymentConfig(),
                1,
                5,
                telemetry=telemetry,
            )
            self.assertEqual(result.assignments, {"a": "a"})
            self.assertEqual(telemetry["tools_version"], 2)
            q = candidate_tools.deployment_quality([0], 1, telemetry)
            self.assertEqual(q["repair_observability"], "unavailable")
            self.assertEqual(
                (root / "runtime/stage1_llm/candidate_tools.py").read_bytes(), raw
            )

    def test_search_preserves_legal_front_and_reports_quality_by_split(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            c = controller(
                settings_at(root, evaluation_budget=2),
                root,
                [
                    REPAIR_ONLY,
                    "def solve(turbines,config,seed):\n return {'nest_ids':[t.identifier for t in turbines],'assignments':{t.identifier:t.identifier for t in turbines}}\n",
                ],
            )
            with contextlib.redirect_stdout(io.StringIO()) as output:
                path = c.run("quality")
            self.assertEqual(len(c.archive), 2)
            first = c.history[0]["evaluation"]
            self.assertIn("quality", first["datasets"][0])
            self.assertIn("training", first["quality_summary"])
            self.assertEqual(first["quality_summary"]["all"]["covered_instances"], 1)
            self.assertEqual(len(c.history[0]["objectives"]), 3)
            self.assertIn("quality=", output.getvalue())
            report = json.loads(
                (path / "quality_report.json").read_text(encoding="utf-8")
            )
            self.assertEqual(len(report["candidates"]), 2)


if __name__ == "__main__":
    unittest.main()
