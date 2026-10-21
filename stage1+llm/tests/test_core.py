import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "stage1+llm"))

from stage1_llm.candidate import CandidateProposal, validate_candidate_source
from stage1_llm.client import parse_model_response
from stage1_llm.evaluator import evaluate_proposal
from stage1_llm.reward import CandidateMetrics, candidate_reward, compare_metrics
from stage1_llm.selection import DiscountedUCB
from stage1_llm.settings import SearchSettings, load_search_settings
from stage1_llm.domain import DeploymentConfig, Turbine, load_turbines
from stage1_llm.sandbox import run_candidate
from stage1_llm.client import DeepSeekSettings, ModelCandidate
from stage1_llm.search import SearchController


GOOD_SOURCE = "def solve(turbines, config, seed):\n    return {'nest_ids': [t.identifier for t in turbines], 'assignments': {t.identifier: t.identifier for t in turbines}}\n"

def model_candidate(direction="structural_exploration", source=GOOD_SOURCE):
    return ModelCandidate(source, "独立候选", "验证部署假设", "产生可行部署", "完整候选实现", direction)


class SettingsTests(unittest.TestCase):
    def test_search_config_has_main_and_validation_inputs(self):
        settings = load_search_settings(Path(__file__).parents[1] / "config" / "search.json")
        self.assertEqual(settings.target_input.name, "turbine102.tsv")
        self.assertGreaterEqual(settings.archive_size, 1)
        self.assertEqual(settings.initial_candidates, 3)


class ModelResponseTests(unittest.TestCase):
    def test_parses_json_and_fenced_json(self):
        source = "def solve(turbines, config, seed):\n    return {'nest_ids': ['1'], 'assignments': {'1': '1'}}\n"
        payload = {"algorithm_code": source, "algorithm_idea": "idea", "change_hypothesis": "hypothesis", "expected_improvement": "improvement", "change_summary": "test", "direction_id": "reduce_nests"}
        parsed = parse_model_response(json.dumps(payload), "reduce_nests")
        self.assertEqual(parsed.algorithm_code, source)
        parsed = parse_model_response("```json\n" + json.dumps(payload) + "\n```", "reduce_nests")
        self.assertEqual(parsed.algorithm_code, source)
        with self.assertRaises(ValueError):
            parse_model_response(source)


class CandidateTests(unittest.TestCase):
    def test_rejects_forbidden_import_and_accepts_complete_interface(self):
        good = "def solve(turbines, config, seed):\n    return {'nest_ids': [t.identifier for t in turbines], 'assignments': {t.identifier: t.identifier for t in turbines}}\n"
        validate_candidate_source(good)
        with self.assertRaises(ValueError):
            validate_candidate_source("import os\ndef solve(turbines, config, seed): return None")
        with self.assertRaises(ValueError):
            validate_candidate_source("def other(turbines, config, seed): return None")


class EvaluatorTests(unittest.TestCase):
    def test_fixed_evaluator_rejects_uncovered_and_reports_metrics(self):
        turbines = [Turbine("a", 0, 0), Turbine("b", 0.1, 0)]
        config = DeploymentConfig(communication_radius_km=1.0)
        good = CandidateProposal(["a"], {"a": "a", "b": "a"})
        result = evaluate_proposal(turbines, config, good)
        self.assertTrue(result.feasible)
        self.assertEqual(result.metrics.nest_count, 1)
        bad = CandidateProposal(["a"], {"a": "a"})
        rejected = evaluate_proposal(turbines, config, bad)
        self.assertFalse(rejected.feasible)
        self.assertTrue(rejected.violations)


class RewardAndSelectionTests(unittest.TestCase):
    def test_feasible_nest_count_dominates_secondary_metrics(self):
        few = CandidateMetrics(True, 2, 10, 1, 20)
        many = CandidateMetrics(True, 3, 100, 100, 1)
        self.assertGreater(candidate_reward(few, 2), candidate_reward(many, 2))
        self.assertLess(compare_metrics(few, many), 0)
        self.assertGreater(candidate_reward(few, incumbent=many), candidate_reward(many, incumbent=few))

    def test_discounted_ucb_updates_and_selects_known_arm(self):
        ucb = DiscountedUCB(["repair", "rewrite"], decay=0.9, exploration=0.0, seed=1)
        ucb.update("repair", 1.0)
        ucb.update("rewrite", -1.0)
        self.assertEqual(ucb.select(), "repair")
        ucb.update("repair", 0.0)
        self.assertEqual(ucb.counts["repair"], 2)

    def test_discounted_ucb_state_round_trip(self):
        first = DiscountedUCB(["a", "b"], seed=7)
        first.update("a", 0.5)
        first.select()
        state = first.state_dict()
        second = DiscountedUCB(["a", "b"], seed=7)
        second.load_state_dict(state)
        self.assertEqual(first.select(), second.select())

    def test_discounted_ucb_reports_trace_and_honours_forced_repair(self):
        ucb = DiscountedUCB(["repair", "rewrite"], seed=3)
        choice, trace = ucb.select_with_trace("repair")
        self.assertEqual(choice, "repair")
        self.assertTrue(trace["repair"]["forced"])


class SecretHandlingTests(unittest.TestCase):
    def test_api_key_redaction(self):
        from stage1_llm.artifacts import redact_secrets
        self.assertNotIn("secret-value", json.dumps(redact_secrets({"api_key": "secret-value"})))
        self.assertNotIn("custom-key", json.dumps(redact_secrets("request failed custom-key", ("custom-key",))))


class SandboxTests(unittest.TestCase):
    def test_executes_proposal_and_enforces_timeout(self):
        turbines = [Turbine("a", 0.0, 0.0)]
        config = DeploymentConfig()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "candidate.py"
            path.write_text("def solve(turbines, config, seed):\n    return {'nest_ids':['a'], 'assignments':{'a':'a'}}\n", encoding="utf-8")
            self.assertEqual(run_candidate(path, turbines, config, 1, 3).nest_ids, ["a"])
            path.write_text("def solve(turbines, config, seed):\n    while True: pass\n", encoding="utf-8")
            with self.assertRaises(TimeoutError):
                run_candidate(path, turbines, config, 1, 0.1)

    def test_independent_domain_can_load_walney_input(self):
        self.assertEqual(len(load_turbines(ROOT / "turbine189.tsv")), 189)


class SearchIntegrationTests(unittest.TestCase):
    def test_constraint_diagnostic_forces_matching_repair_direction(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_path = root / "tiny.tsv"
            input_path.write_text("a\t0\t0\nb\t0.1\t0\n", encoding="utf-8")
            settings = SearchSettings(input_path, (), DeploymentConfig(), initial_candidates=1, evaluation_budget=2,
                                      training_virtual_fields=0, validation_virtual_fields=0, data_dir=root/'data')
            llm = DeepSeekSettings("https://example.invalid", "test", "not-used")
            broken = "def solve(turbines, config, seed):\n    return {'nest_ids':['a'], 'assignments':{'a':'a'}}\n"
            seen = []
            class RepairClient:
                def generate(self, _system, _prompt, direction):
                    seen.append(direction)
                    return model_candidate(direction, broken if len(seen) == 1 else GOOD_SOURCE)
            SearchController(settings, llm, root / "outputs", client=RepairClient()).run(run_id="repair")
            self.assertEqual(seen, ["structural_exploration", "constraint_repair"])

    def test_failed_llm_generation_never_exports_a_seed_algorithm(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_path = root / "tiny.tsv"
            input_path.write_text("a\t0\t0\n", encoding="utf-8")
            settings = SearchSettings(input_path, (), DeploymentConfig(), initial_candidates=1, evaluation_budget=1, no_improvement_limit=1,
                                      training_virtual_fields=0, validation_virtual_fields=0, data_dir=root/'data')
            llm = DeepSeekSettings("https://example.invalid", "test", "not-used")
            class BrokenClient:
                def generate(self, *_args): raise ValueError("empty response")
            output = SearchController(settings, llm, root / "outputs", client=BrokenClient()).run(run_id="broken")
            self.assertFalse((output / "best_algorithm.py").exists())
            self.assertEqual(json.loads((output / "summary.json").read_text(encoding="utf-8"))["status"], "no_deployable_llm_candidate")

    def test_llm_search_artifacts_and_resume_without_live_api(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_path = root / "tiny.tsv"
            input_path.write_text("a\t0\t0\nb\t0.1\t0\n", encoding="utf-8")
            from stage1_llm.settings import SearchSettings
            settings = SearchSettings(input_path, (), DeploymentConfig(), initial_candidates=1, evaluation_budget=1,
                                      training_virtual_fields=0, validation_virtual_fields=0, data_dir=root/'data')
            llm = DeepSeekSettings("https://example.invalid", "test", "not-used")
            class FakeClient:
                def generate(self, _system, _prompt, direction): return model_candidate(direction)
            first = SearchController(settings, llm, root / "outputs", client=FakeClient()).run(run_id="tiny-run")
            self.assertFalse((first / "best_algorithm.py").exists())
            self.assertTrue((first / "pareto.json").is_file())
            self.assertTrue((first / "events.jsonl").is_file())
            self.assertTrue((first / "trajectory.csv").is_file())
            resumed = SearchController(settings, llm, root / "outputs", client=FakeClient()).run(run_id="tiny-run", resume=True)
            self.assertEqual(first, resumed)

    def test_resume_can_reset_a_reached_plateau(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_path = root / "tiny.tsv"
            input_path.write_text("a\t0\t0\nb\t0.1\t0\n", encoding="utf-8")
            llm = DeepSeekSettings("https://example.invalid", "test", "not-used")
            class DuplicateClient:
                def generate(self, _system, _prompt, direction): return model_candidate(direction)
            first_settings = SearchSettings(input_path, (), DeploymentConfig(), initial_candidates=2,
                                            evaluation_budget=2, no_improvement_limit=1,
                                            training_virtual_fields=0, validation_virtual_fields=0, data_dir=root/'data')
            first = SearchController(first_settings, llm, root / "outputs", client=DuplicateClient()).run(run_id="plateau")
            second_settings = SearchSettings(input_path, (), DeploymentConfig(), initial_candidates=2,
                                             evaluation_budget=3, no_improvement_limit=1,
                                             training_virtual_fields=0, validation_virtual_fields=0, data_dir=root/'data')
            SearchController(second_settings, llm, root / "outputs", client=DuplicateClient()).run(
                run_id="plateau", resume=True, reset_stall=True)
            state = json.loads((first / "search_state.json").read_text(encoding="utf-8"))
            self.assertEqual(len(state["history"]), 3)


if __name__ == "__main__":
    unittest.main()
