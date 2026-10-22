"""Strict contract regressions, including lossless real provider failures."""

import base64
import contextlib
import copy
import csv
import dataclasses
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from test_v3 import SOLO, controller, settings_at
from stage1_llm.candidate_tools import CandidateTools
from stage1_llm.domain import DeploymentConfig, Turbine
from stage1_llm.sandbox import run_candidate


class ContractTests(unittest.TestCase):
    def tools(self, **kwargs):
        return CandidateTools(
            [Turbine("a", 0, 0), Turbine("b", 0.1, 0)], DeploymentConfig(), **kwargs
        )

    def test_helpers_copy_without_implying_feasibility(self):
        tools = self.tools()
        a = {"a": "a", "b": "a"}
        p = tools.make_proposal(a)
        a["b"] = "b"
        self.assertEqual(p["assignments"]["b"], "a")
        extracted = tools.assignments_from_proposal(p)
        extracted["b"] = "b"
        self.assertEqual(p["assignments"]["b"], "a")
        self.assertTrue(tools.deployment(p)["feasible"])
        invalid = tools.make_proposal({"a": "b", "b": "a"})
        self.assertFalse(tools.deployment(invalid)["feasible"])

    def test_schema_failures_are_not_coerced(self):
        from stage1_llm.candidate_tools import ToolContractError

        calls = [
            ("deployment", {"a": "a"}),
            ("remember", {"a": "a"}),
            ("make_proposal", {"a": ["a"]}),
            ("make_proposal", {"a": 1}),
            ("make_proposal", {1: "a"}),
            ("split_repair", {"nest_ids": ["a"], "assignments": {"a": "a"}}),
            (
                "assignments_from_proposal",
                {"nest_ids": ["a"], "assignments": {"a": {}}},
            ),
            ("deployment", {"nest_ids": ("a",), "assignments": {"a": "a"}}),
        ]
        for method, arg in calls:
            with self.subTest(method=method, arg=arg):
                tools = self.tools()
                with self.assertRaises(ToolContractError):
                    getattr(tools, method)(arg)
                self.assertEqual(tools.stats()["contract_error_count"], 1)

    def test_guarded_results_and_regular_list_index(self):
        from stage1_llm.candidate_tools import ToolContractError

        for mode in ("index", "get", "missing", "copy", "pop", "setdefault"):
            tools = self.tools()
            result = tools.region("a", ["a", "b"])
            with self.subTest(mode=mode), self.assertRaises(ToolContractError):
                if mode == "index":
                    result[0]
                elif mode == "get":
                    result.get(0)
                elif mode == "missing":
                    result["not_a_field"]
                elif mode == "copy":
                    copy.deepcopy(result)[0]
                elif mode == "pop":
                    result.pop(0, None)
                else:
                    result.setdefault(0, None)
            self.assertEqual(tools.stats()["contract_error_count"], 1)
        tools = self.tools()
        p = tools.make_proposal({"a": "a", "b": "a"})
        result = tools.deployment(p)
        self.assertEqual(result["objectives"][0], 0.5)
        self.assertEqual(result.get("optional", 17), 17)
        self.assertEqual(tools.stats()["contract_error_count"], 0)

    def test_region_input_shape_vs_domain_failures(self):
        from stage1_llm.candidate_tools import ToolContractError

        for members in ("ab", {"a": "a"}, [["a"]], [1]):
            with self.subTest(members=members), self.assertRaises(ToolContractError):
                self.tools().region("a", members)
        tools = self.tools()
        for members in (["a", "b"], ("a", "b"), {"a", "b"}, frozenset(("a", "b"))):
            self.assertTrue(tools.region("a", members)["feasible"])
        self.assertFalse(tools.region("a", ["a", "unknown"])["feasible"])
        self.assertFalse(
            tools.deployment(
                {
                    "nest_ids": ["missing"],
                    "assignments": {"a": "missing", "b": "missing"},
                }
            )["feasible"]
        )
        self.assertEqual(tools.stats()["contract_error_count"], 0)

    def test_error_journal_is_bounded(self):
        from stage1_llm.candidate_tools import ToolContractError

        tools = self.tools()
        for _ in range(20):
            try:
                tools.remaining()[0]
            except ToolContractError:
                pass
        stats = tools.stats()
        self.assertEqual(stats["contract_error_count"], 20)
        self.assertEqual(len(stats["contract_errors"]), 5)

    def test_stats_result_does_not_allow_integer_field_reads(self):
        from stage1_llm.candidate_tools import ToolContractError

        tools = self.tools()
        with self.assertRaises(ToolContractError):
            tools.stats()[0]
        self.assertEqual(tools.stats()["contract_error_count"], 1)

    def execute(self, code, **kwargs):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "candidate.py"
            path.write_text(code, encoding="utf-8")
            return run_candidate(
                path, [Turbine("a", 0, 0)], DeploymentConfig(), 1, 5, **kwargs
            )

    def test_caught_error_cannot_be_hidden_by_feasible_return_or_budget(self):
        from stage1_llm.sandbox import CandidateExecutionError

        for ending in ("return p", "while True: tick()"):
            code = (
                """from candidate_tools import make_proposal, remember, remaining, tick
def solve(turbines, config, seed):
    p = make_proposal({t.identifier:t.identifier for t in turbines})
    remember(p)
    try:
        remaining()[0]
    except Exception:
        pass
    """
                + ending
                + "\n"
            )
            telemetry = {}
            with self.subTest(ending=ending), self.assertRaises(
                CandidateExecutionError
            ) as error:
                self.execute(code, limits={"max_move_attempts": 2}, telemetry=telemetry)
            self.assertEqual(error.exception.status, "tool_contract_error")
            self.assertEqual(telemetry["contract_errors"][0]["tool"], "remaining")
            self.assertGreater(telemetry["contract_errors"][0]["candidate_line"], 0)

    def test_final_return_rejects_nonstring_ids(self):
        from stage1_llm.sandbox import CandidateExecutionError

        code = "def solve(turbines, config, seed):\n return {'nest_ids':['a'],'assignments':{'a':['a']}}\n"
        with self.assertRaises(CandidateExecutionError):
            self.execute(code)

    def test_caught_incorrect_call_signature_is_also_a_contract_failure(self):
        from stage1_llm.sandbox import CandidateExecutionError

        code = """from candidate_tools import deployment
def solve(turbines,config,seed):
    try: deployment()
    except Exception: pass
    return {'nest_ids':['a'],'assignments':{'a':'a'}}
"""
        with self.assertRaises(CandidateExecutionError) as error:
            self.execute(code)
        first = error.exception.telemetry["contract_errors"][0]
        self.assertEqual(first["tool"], "deployment")
        self.assertEqual(first["code"], "invalid_arguments")

    def test_contract_error_priority_and_unexpected_error_type(self):
        from stage1_llm.sandbox import CandidateExecutionError

        prefix = """from candidate_tools import remaining
def solve(turbines, config, seed):
    try: remaining()[0]
    except Exception: pass
    """
        with self.assertRaises(CandidateExecutionError) as error:
            self.execute(prefix + "raise ValueError('later unexpected error')\n")
        self.assertEqual(error.exception.telemetry["final_error_type"], "ValueError")
        with self.assertRaisesRegex(RuntimeError, "ZeroDivisionError") as error:
            self.execute("def solve(turbines,config,seed):\n return 1/0\n")
        self.assertNotIsInstance(error.exception, CandidateExecutionError)

    def test_hard_timeout_retains_caught_contract_error(self):
        from stage1_llm.sandbox import CandidateExecutionError

        code = """from candidate_tools import remaining
def solve(turbines,config,seed):
    try: remaining()[0]
    except Exception: pass
    while True: pass
"""
        telemetry = {}
        with self.assertRaises(CandidateExecutionError):
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "candidate.py"
                path.write_text(code, encoding="utf-8")
                run_candidate(
                    path,
                    [Turbine("a", 0, 0)],
                    DeploymentConfig(),
                    1,
                    2,
                    telemetry=telemetry,
                )
        self.assertEqual(telemetry["termination"], "hard_timeout")
        self.assertEqual(telemetry["final_error_type"], "TimeoutError")
        self.assertTrue(telemetry["contract_error_count_is_lower_bound"])

    def test_split_preconditions_and_unchanged_physical_failures(self):
        from stage1_llm.candidate_tools import ToolContractError

        tools = self.tools()
        for mapping in ({"a": "a"}, {"a": "unknown", "b": "a"}):
            with self.assertRaises(ToolContractError):
                tools.split_repair(mapping)
        tools = self.tools()
        # Removing a nest must not remove that turbine's inspection assignment.
        self.assertFalse(
            tools.deployment({"nest_ids": ["a"], "assignments": {"a": "a"}})["feasible"]
        )
        self.assertFalse(
            tools.deployment(
                {"nest_ids": ["a", "b"], "assignments": {"a": "a", "b": "a"}}
            )["feasible"]
        )
        self.assertTrue(
            tools.deployment({"nest_ids": ["a"], "assignments": {"a": "a", "b": "a"}})[
                "feasible"
            ]
        )
        self.assertEqual(tools.stats()["contract_error_count"], 0)

    def test_original_failed_sources_report_first_contract_error(self):
        from stage1_llm.sandbox import CandidateExecutionError

        fixture = json.loads(
            (Path(__file__).parent / "fixtures/tool_contract_failures.json").read_text(
                encoding="utf-8"
            )
        )
        for record in fixture["sources"]:
            raw = base64.b64decode(record["source_base64"])
            self.assertEqual(hashlib.sha256(raw).hexdigest(), record["sha256"])
            with self.subTest(
                candidate=record["filename"]
            ), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / record["filename"]
                path.write_bytes(raw)
                with self.assertRaises(CandidateExecutionError) as error:
                    run_candidate(
                        path,
                        [Turbine("preflight-0", 0, 0)],
                        DeploymentConfig(),
                        1234,
                        5,
                    )
                first = error.exception.telemetry["contract_errors"][0]
                self.assertIn(first["tool"], ("region", "deployment"))
                self.assertNotIn("unhashable", first["message"])
                self.assertLessEqual(
                    len(error.exception.telemetry["contract_errors"]), 5
                )

    def test_search_feedback_distinguishes_contract_error(self):
        bad = """from candidate_tools import region
def solve(turbines, config, seed):
    try: region(turbines[0].identifier,[turbines[0].identifier])[0]
    except Exception: pass
    return {'nest_ids':[t.identifier for t in turbines],'assignments':{t.identifier:t.identifier for t in turbines}}
"""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            c = controller(settings_at(root, evaluation_budget=2), root, [bad, SOLO])
            with contextlib.redirect_stdout(io.StringIO()) as output:
                c.run("contract")
            first = c.history[0]
            self.assertEqual(first["evaluation"]["status"], "tool_contract_error")
            self.assertEqual(first["evaluation"]["failure_stage"], "preflight")
            self.assertLess(first["reward"], 0)
            self.assertIn("contract_errors", c.client.prompts[1])
            self.assertIn("feasible", c.client.prompts[1])
            self.assertIn("tool_contract_error", output.getvalue())
            self.assertTrue(c.archive)
            feedback = json.loads(
                c.client.prompts[1]
                .split("固定评测诊断：", 1)[1]
                .split("\n父代完整源码：", 1)[0]
            )
            diagnostic = feedback["parent_diagnostic"]
            self.assertLessEqual(len(diagnostic["contract_errors"]), 3)
            self.assertNotIn("diagnostics", diagnostic)
            self.assertNotIn("violations", diagnostic)
            with (root / "outputs/contract/trajectory.csv").open(
                encoding="utf-8-sig", newline=""
            ) as handle:
                record = next(csv.DictReader(handle))
            self.assertEqual(record["status"], first["evaluation"]["status"])
            self.assertEqual(
                int(record["tool_contract_error_count"]),
                first["evaluation"]["contract_error_count"],
            )

    def test_later_phases_record_contract_diagnostics_and_elapsed_time(self):
        for phase, predicate in (
            (
                "constraint_preflight",
                "turbines[0].identifier.startswith('constraint-')",
            ),
            ("dataset", "turbines[0].identifier == 'a'"),
        ):
            code = """from candidate_tools import remaining
def solve(turbines,config,seed):
    if PREDICATE:
        try: remaining()[0]
        except Exception: pass
    return {'nest_ids':[t.identifier for t in turbines],'assignments':{t.identifier:t.identifier for t in turbines}}
""".replace(
                "PREDICATE", predicate
            )
            with self.subTest(phase=phase), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                c = controller(
                    settings_at(root, evaluation_budget=1, constraint_preflight=True),
                    root,
                    [code],
                )
                with contextlib.redirect_stdout(io.StringIO()):
                    c.run("phases")
                row = c.history[0]
                self.assertEqual(row["evaluation"]["status"], "tool_contract_error")
                self.assertEqual(row["evaluation"]["failure_stage"], phase)
                record = (
                    row["constraint_checks"][-1]
                    if phase == "constraint_preflight"
                    else row["failed_attempts"][-1]
                )
                self.assertGreater(
                    record.get("seconds", record.get("algorithm_seconds")), 0
                )
                self.assertEqual(
                    record["telemetry"]["contract_errors"][0]["tool"], "remaining"
                )
                self.assertFalse(c.archive)

    def test_failed_preflight_checkpoint_resumes_without_reexecution(self):
        for phase, predicate in (
            ("preflight", "True"),
            (
                "constraint_preflight",
                "turbines[0].identifier.startswith('constraint-')",
            ),
        ):
            code = """from candidate_tools import remaining
def solve(turbines,config,seed):
    if PREDICATE: remaining()[0]
    return {'nest_ids':[t.identifier for t in turbines],'assignments':{t.identifier:t.identifier for t in turbines}}
""".replace(
                "PREDICATE", predicate
            )
            with self.subTest(phase=phase), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                settings = settings_at(
                    root, evaluation_budget=1, constraint_preflight=True
                )
                first = controller(settings, root, [code])
                with contextlib.redirect_stdout(io.StringIO()), patch.object(
                    first, "_finish", side_effect=KeyboardInterrupt
                ):
                    with self.assertRaises(KeyboardInterrupt):
                        first.run("restore-contract")
                resumed = controller(settings, root, [])
                with contextlib.redirect_stdout(io.StringIO()), patch(
                    "stage1_llm.search.run_candidate",
                    side_effect=AssertionError("must not repeat failed case"),
                ):
                    resumed.run("restore-contract", resume=True)
                self.assertEqual(
                    resumed.history[0]["evaluation"]["status"], "tool_contract_error"
                )
                self.assertEqual(
                    resumed.history[0]["evaluation"]["failure_stage"], phase
                )
                self.assertFalse(resumed.client.prompts)

    def test_dataset_contract_checkpoint_cannot_retry_into_success(self):
        code = """from candidate_tools import remaining
def solve(turbines,config,seed):
    if turbines[0].identifier == 'a':
        try: remaining()[0]
        except Exception: pass
    return {'nest_ids':[t.identifier for t in turbines],'assignments':{t.identifier:t.identifier for t in turbines}}
"""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            settings = settings_at(root, evaluation_budget=1)
            first = controller(settings, root, [code])
            with contextlib.redirect_stdout(io.StringIO()), patch.object(
                first, "_finish", side_effect=KeyboardInterrupt
            ):
                with self.assertRaises(KeyboardInterrupt):
                    first.run("dataset-contract")
            resumed = controller(settings, root, [])
            with contextlib.redirect_stdout(io.StringIO()), patch(
                "stage1_llm.search.run_candidate",
                side_effect=AssertionError("failed execution must not rerun"),
            ):
                resumed.run("dataset-contract", resume=True)
            result = resumed.history[0]["evaluation"]
            self.assertEqual(result["status"], "tool_contract_error")
            self.assertEqual(result["failure_stage"], "dataset")
            self.assertFalse(resumed.archive)

    def test_interrupted_error_event_cannot_be_cleared_by_resume(self):
        code = """from candidate_tools import remaining
def solve(turbines,config,seed):
    try: remaining()[0]
    except Exception: pass
    while True: pass
"""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            settings = settings_at(root, evaluation_budget=1)
            first = controller(settings, root, [code])

            def interrupt_on_event(*args):
                if args[-1].get("kind") == "tool_contract_error":
                    raise KeyboardInterrupt

            with contextlib.redirect_stdout(io.StringIO()), patch.object(
                first, "_tool_event", side_effect=interrupt_on_event
            ):
                with self.assertRaises(KeyboardInterrupt):
                    first.run("interrupt-contract")
            resumed = controller(settings, root, [])
            with contextlib.redirect_stdout(io.StringIO()), patch(
                "stage1_llm.search.run_candidate",
                side_effect=AssertionError("known failed execution must not rerun"),
            ):
                resumed.run("interrupt-contract", resume=True)
            self.assertEqual(
                resumed.history[0]["evaluation"]["status"], "tool_contract_error"
            )
            self.assertFalse(resumed.archive)

    def test_export_helpers_and_legacy_runtime_version_branch(self):
        from stage1_llm.exports import export_candidate
        from stage1_llm.smoke import TOOL_GROUP

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            settings = settings_at(root, evaluation_budget=1)
            c = controller(settings, root, [TOOL_GROUP])
            with contextlib.redirect_stdout(io.StringIO()):
                out = c.run("export-contract")
            target = export_candidate(out, "C0001", allow_mock=True)
            selection = json.loads(
                (target / "selection.json").read_text(encoding="utf-8")
            )
            self.assertEqual(selection["run_identity"]["tool_contract_version"], 2)
            command = [
                sys.executable,
                "-I",
                str(target / "run_selected.py"),
                "--input",
                str(settings.target_input),
            ]
            proc = subprocess.run(
                command, capture_output=True, text=True, encoding="utf-8"
            )
            self.assertEqual(proc.returncode, 0, proc.stderr)
            self.assertEqual(json.loads(proc.stdout)["telemetry"]["tools_version"], 2)
            # Synthetic bundle replacement uses frozen v1 source, not a live old run.
            fixture = json.loads(
                (Path(__file__).parent / "fixtures/legacy_runtime_v1.json").read_text(
                    encoding="utf-8"
                )
            )
            for row in fixture["sources"]:
                raw = base64.b64decode(row["source_base64"])
                self.assertEqual(hashlib.sha256(raw).hexdigest(), row["sha256"])
                (target / "runtime/stage1_llm" / row["filename"]).write_bytes(raw)
                selection["runtime_hashes"][row["filename"]] = row["sha256"]
            (target / "algorithm.py").write_text(SOLO, encoding="utf-8")
            selection["source_sha256"] = hashlib.sha256(SOLO.encode()).hexdigest()
            selection["run_identity"]["tool_contract_version"] = 1
            (target / "selection.json").write_text(
                json.dumps(selection), encoding="utf-8"
            )
            before = {
                p.name: p.read_bytes()
                for p in (target / "runtime/stage1_llm").glob("*.py")
            }
            telemetry = {}
            result = run_candidate(
                target / "algorithm.py",
                [Turbine("a", 0, 0)],
                DeploymentConfig(),
                1,
                5,
                telemetry=telemetry,
            )
            self.assertEqual(result.assignments, {"a": "a"})
            self.assertEqual(telemetry["tools_version"], 1)
            proc = subprocess.run(
                command, capture_output=True, text=True, encoding="utf-8"
            )
            self.assertEqual(proc.returncode, 0, proc.stderr)
            self.assertTrue(json.loads(proc.stdout)["evaluation"]["feasible"])
            self.assertEqual(
                before,
                {
                    p.name: p.read_bytes()
                    for p in (target / "runtime/stage1_llm").glob("*.py")
                },
            )

    def test_cli_evaluation_keeps_structured_error_and_live_event(self):
        from stage1_llm.cli import main

        code = """from candidate_tools import remaining
def solve(turbines,config,seed):
    try: remaining()[0]
    except Exception: pass
    return {'nest_ids':[t.identifier for t in turbines],'assignments':{t.identifier:t.identifier for t in turbines}}
"""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            settings = settings_at(root)
            config = root / "search.json"
            config.write_text(
                json.dumps(dataclasses.asdict(settings), default=str), encoding="utf-8"
            )
            source = root / "algorithm.py"
            source.write_text(code, encoding="utf-8")
            with contextlib.redirect_stdout(
                io.StringIO()
            ) as output, contextlib.redirect_stderr(io.StringIO()):
                status = main(
                    [
                        "evaluate",
                        "--search-config",
                        str(config),
                        "--candidate",
                        str(source),
                        "--input",
                        str(settings.target_input),
                        "--outputs",
                        str(root / "outputs"),
                        "--run-id",
                        "cli-contract",
                    ]
                )
            self.assertEqual(status, 1)
            result = json.loads(
                (root / "outputs/cli-contract/evaluation.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(result["status"], "tool_contract_error")
            self.assertEqual(
                result["telemetry"]["contract_errors"][0]["tool"], "remaining"
            )
            self.assertIn("[tool_contract_error]", output.getvalue())


if __name__ == "__main__":
    unittest.main()
