"""Small offline checks for scheduling/persistence and saved-result plotting only."""
from __future__ import annotations

import contextlib
import copy
import io
import json
import sys
import unittest
from uuid import uuid4
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from windsearch.artifacts import Records, atomic_json, sha256_file
from windsearch.client import DeepSeekSettings
from windsearch.candidate import ModelCandidate
from windsearch.evolution import Evolution
from windsearch.plotting import main as plot_main, resolve_candidate, validate_result
from windsearch.settings import SearchSettings


@contextlib.contextmanager
def fixture_directory():
    # Keep artifacts under the workspace, also useful for manually viewing a plot.
    parent = Path(__file__).resolve().parents[1] / "outputs/targeted-checks"
    parent.mkdir(parents=True, exist_ok=True)
    path = (parent / ("check-" + uuid4().hex)).resolve()
    if not path.is_relative_to(parent.resolve()):
        raise RuntimeError("Fixture escaped its workspace")
    path.mkdir()
    yield str(path)


def instance_record(name="real102", objective=None):
    objective = objective or [2, 12.0, 12.0]
    metrics = {"nest_count": objective[0], "total_flight_seconds": objective[1],
               "maximum_flight_seconds": objective[2], "minimum_endurance_margin_seconds": 2400.0}
    return {"name": name, "split": "training", "code_success": True, "feasible": True,
            "stop_reason": "strategy_declared_neighborhood_exhausted", "only_initial_solution": False,
            "initial_objectives": [3, 0, 0], "objectives": objective, "metrics": metrics,
            "search": {"improvements": 1}, "failure_counts": {},
            "timing": {"search_seconds": 1.0, "strategy_seconds": .2, "evaluation_seconds": .8,
                       "total_seconds": 1.1}, "oracle_counters": {"inferences": 1},
            "final_audit_counters": {"inferences": 1}}


class OfflineEvolution(Evolution):
    """Replace paid generation and deployment evaluation, not scheduling/state logic."""
    def __init__(self, root, outcomes, target=2):
        self.outcomes, self.generated = outcomes, []
        settings = replace(SearchSettings(), evaluation_budget=target)
        llm = DeepSeekSettings("https://example.invalid", "offline", "", retries=0)
        datasets = [{"name": "real102"}] + [{"name": f"fixture-{i}"} for i in range(16)]
        super().__init__(settings, llm, {"fingerprint": "fixture", "datasets": datasets},
                         SimpleNamespace(metadata={"origin": "offline_fixture"}), Records(root))

    def generate(self, row, context):
        self.generated.append(copy.deepcopy(row))
        self.pending = {"phase": "request_started", "row": row}
        row["api_metadata"] = {"usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}}
        row["llm_seconds"] = .1
        aid = row["attempt_id"]
        row["source_path"] = "candidates/attempts/" + aid + ".py"
        self.records.text(row["source_path"], "# offline source " + aid + "\n")
        row["source_sha256"] = sha256_file(self.records.root / row["source_path"])
        self.pending = {"phase": "evaluate", "row": row}
        self.save()

    def evaluate(self, row):
        outcome = self.outcomes[row["attempt_id"]]
        if outcome == "fail":
            row.update(eligible=False, status="candidate_error",
                       error={"error": "offline fixture failure", "error_type": "FixtureError"})
        else:
            row.update(eligible=True, status="completed", objectives=outcome,
                       instances=[instance_record(d["name"], outcome) for d in self.bundle["datasets"]])


class ParentChecks(unittest.TestCase):
    def test_saved_source_resumes_evaluation_without_regeneration(self):
        with fixture_directory() as path, contextlib.redirect_stdout(io.StringIO()):
            engine = OfflineEvolution(Path(path), {"A0001": [3, 100, 90]}, target=1)
            model = ModelCandidate("# offline source", "idea", "hypothesis", "effect", "change", "initialize")
            calls = []
            def generate(*args, **kwargs):
                calls.append(1)
                return model
            engine.client = SimpleNamespace(generate=generate,
                last_metadata={"usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}},
                last_response={"finish_reason": "stop"})
            row = {"candidate_id": None, "attempt_id": "A0001", "round_index": 1,
                   "attempt_number": 1, "operator": "initialize", "parent_ids": [],
                   "repair_target": None, "selection_reason": "fixture", "instances": []}
            Evolution.generate(engine, row, {"direction_id": "initialize"})
            self.assertEqual(calls, [1])
            self.assertTrue((Path(path)/"candidates/attempts/A0001.py").is_file())
            resumed = OfflineEvolution(Path(path), {"A0001": [3, 100, 90]}, target=1)
            self.assertEqual(resumed.run(resume=True), "target_completed")
            self.assertEqual(resumed.generated, [])
            self.assertEqual(resumed.cases()[0]["candidate_id"], "C0001")

    def test_best_parent_survives_failure_and_worse_qualified_child(self):
        with fixture_directory() as path, contextlib.redirect_stdout(io.StringIO()):
            engine = OfflineEvolution(Path(path), {"A0001": [3, 100, 90], "A0002": "fail",
                                      "A0003": [4, 90, 70], "A0004": [3, 110, 95]}, target=3)
            self.assertEqual(engine.run(), "target_completed")
            self.assertEqual([r["candidate_id"] for r in engine.history], ["C0001", None, "C0002", "C0003"])
            self.assertEqual(engine.generated[2]["parent_ids"], ["C0001"])
            self.assertEqual(engine.generated[2]["repair_target"], "A0002")
            self.assertEqual(engine.generated[3]["parent_ids"], ["C0001"])
            self.assertEqual(engine.archive()[0]["candidate_id"], "C0001")
            index = json.loads((Path(path)/"diagnostics/runtime/candidate_index.json").read_text())
            self.assertEqual(index["cases"]["C0002"]["attempt_id"], "A0003")
            self.assertFalse(index["attempts"]["A0002"]["eligible"])

    def test_three_failures_stop_and_resume_does_not_reset_allowance(self):
        with fixture_directory() as path, contextlib.redirect_stdout(io.StringIO()):
            outcomes = {f"A{i:04d}": "fail" for i in range(1, 4)}
            engine = OfflineEvolution(Path(path), outcomes)
            self.assertEqual(engine.run(), "round_attempts_exhausted")
            self.assertEqual([r["operator"] for r in engine.generated], ["initialize", "repair", "initialize"])
            self.assertTrue(all(not r["parent_ids"] for r in engine.generated))
            resumed = OfflineEvolution(Path(path), outcomes)
            self.assertEqual(resumed.run(resume=True), "round_attempts_exhausted")
            self.assertEqual(resumed.generated, [])
            self.assertEqual(len(resumed.history), 3)

    def test_unknown_request_consumes_attempt_without_replay(self):
        with fixture_directory() as path, contextlib.redirect_stdout(io.StringIO()):
            engine = OfflineEvolution(Path(path), {"A0002": [3, 100, 90]}, target=1)
            engine.pending = {"phase": "request_started", "row": {
                "candidate_id": None, "attempt_id": "A0001", "round_index": 1,
                "attempt_number": 1, "operator": "initialize", "parent_ids": [],
                "repair_target": None, "selection_reason": "fixture", "instances": []}}
            engine.save()
            resumed = OfflineEvolution(Path(path), {"A0002": [3, 100, 90]}, target=1)
            self.assertEqual(resumed.run(resume=True), "target_completed")
            self.assertEqual([r["attempt_id"] for r in resumed.generated], ["A0002"])
            self.assertEqual(resumed.history[0]["status"], "interrupted_request_unknown")
            self.assertIsNone(resumed.token_usage()["tokens"]["total_tokens"]["total"])

    def test_fusion_distinct_sources_and_stable_objective_ties(self):
        with fixture_directory() as path:
            engine = OfflineEvolution(Path(path), {})
            engine.history = [
                {"candidate_id": "C0001", "eligible": True, "objectives": [3, 100, 90], "source_sha256": "same"},
                {"candidate_id": "C0002", "eligible": True, "objectives": [3, 100, 90], "source_sha256": "same"},
                {"candidate_id": "C0003", "eligible": True, "objectives": [4, 90, 70], "source_sha256": "other"},
                {"candidate_id": None, "eligible": False, "error": {"error": "latest failure"}},
            ]
            op, parents = engine.choose(5)
            self.assertEqual(op, "parent_fusion")
            self.assertEqual([r["candidate_id"] for r in parents], ["C0001", "C0003"])
            self.assertEqual(engine.choose(3)[1][0]["candidate_id"], "C0001")


def saved_plot_fixture(root, result_id="C0001", instance="real102"):
    problem = {"turbines": [{"identifier": "a", "x_km": 0, "y_km": 0},
                             {"identifier": "b", "x_km": 1, "y_km": 1},
                             {"identifier": "c", "x_km": 3, "y_km": 0}],
               "environment": {"wind_speed_mps": 6, "wind_direction_degrees": 90},
               "config": {"communication_radius_km": 7}}
    result = {**instance_record(instance), "solution": {"assignments": {"a": "a", "b": "a", "c": "c"},
              "nest_ids": ["a", "c"]}, "final_audit": {"feasible": True,
              "metrics": instance_record()["metrics"], "regions": [
                  {"nest_id": "a", "route_ids": ["a", "b", "a"], "feasible": True},
                  {"nest_id": "c", "route_ids": ["c", "c"], "feasible": True}]}}
    atomic_json(root/f"diagnostics/runtime/{result_id}/{instance}-problem.json", problem)
    atomic_json(root/f"diagnostics/evaluations/{result_id}/{instance}.json", result)
    return problem, result


class PlotChecks(unittest.TestCase):
    def test_old_new_external_and_missing_batch_inputs(self):
        with fixture_directory() as path, contextlib.redirect_stdout(io.StringIO()):
            root = Path(path)
            old = root/"old"
            problem, result = saved_plot_fixture(old)
            atomic_json(old/"summary.json", {"best_candidate": "C0001"})
            atomic_json(old/"diagnostics/evaluations/C0001.json", {"eligible": True, "status": "completed"})
            self.assertEqual(plot_main(["--run-dir", str(old)]), 0)
            plotted = json.loads((old/"plots/C0001/plot_summary.json").read_text())
            row = plotted["results"][0]
            self.assertEqual(row["nest_ids"], ["a", "c"])
            self.assertEqual(row["assignment_count"], 3)
            self.assertEqual(row["route_ids"]["a"], ["a", "b", "a"])
            self.assertTrue((old/"plots/C0001/real102.pdf").is_file())
            new = root/"new"
            saved_plot_fixture(new, "A0002")
            atomic_json(new/"summary.json", {"best_candidate": "C0001"})
            entry = {"attempt_id": "A0002", "eligible": True, "status": "completed",
                     "evaluation_directory": "diagnostics/evaluations/A0002",
                     "problem_directory": "diagnostics/runtime/A0002"}
            atomic_json(new/"diagnostics/runtime/candidate_index.json", {
                "cases": {"C0001": entry}, "attempts": {"A0002": entry},
                "expected_instances": ["real102", "missing-fixture"]})
            self.assertEqual(plot_main(["--run-dir", str(new), "--all-instances"]), 1)
            summary = json.loads((new/"plots/C0001/plot_summary.json").read_text())
            self.assertEqual((summary["plotted"], summary["missing"], summary["errors"]), (1, 1, 0))
            failed_entry = {**entry, "eligible": False, "status": "candidate_error"}
            atomic_json(new/"diagnostics/runtime/candidate_index.json", {
                "cases": {"C0001": entry}, "attempts": {"A0002": failed_entry},
                "expected_instances": ["real102"]})
            self.assertEqual(plot_main(["--run-dir", str(new), "--candidate", "A0002"]), 0)
            failed_plot = json.loads((new/"plots/A0002/plot_summary.json").read_text())
            self.assertIn("NOT QUALIFIED", failed_plot["results"][0]["candidate_status"])
            external = root/"external"
            saved_plot_fixture(external, "external", "turbine102")
            self.assertEqual(plot_main(["--run-dir", str(external), "--candidate", "external",
                                       "--instance", "turbine102", "--show-radius"]), 0)
            result["final_audit"]["regions"][0]["route_ids"][1] = "unknown"
            with self.assertRaisesRegex(ValueError, "invalid/unknown"):
                validate_result(problem, result, "fixture.json")

    def test_best_never_falls_back_to_failed_attempt(self):
        with fixture_directory() as path:
            root = Path(path)
            atomic_json(root/"summary.json", {"best_candidate": None})
            with self.assertRaisesRegex(ValueError, "No qualified best"):
                resolve_candidate(root, "best")


if __name__ == "__main__":
    unittest.main()
