"""Offline regressions for provider-generated input mistakes and repair feedback."""

import contextlib
import dataclasses
import hashlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from test_v3 import GROUP, SOLO, controller, settings_at
from stage1_llm.domain import DeploymentConfig, Turbine
from stage1_llm.sandbox import run_candidate


MAPPING_SOURCE = """def solve(turbines, config, seed):
    ids = [t['identifier'] for t in turbines]
    pos = {t['identifier']: (float(t['x_km']), float(t['y_km'])) for t in turbines}
    if not config.max_endurance_seconds > 0:
        raise ValueError('bad endurance')
    return {'nest_ids':ids, 'assignments':{t:t for t in ids}}
"""
RUNTIME_ERROR = "def solve(turbines, config, seed):\n raise ValueError('repair this exact failure')\n"


class CandidateContractTests(unittest.TestCase):
    def test_mapping_access_from_actual_failure_and_attribute_access_are_equivalent(
        self,
    ):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "candidate.py"
            points = [Turbine("a", 0, 0), Turbine("b", 0.1, 0)]
            proposals = []
            for code in (MAPPING_SOURCE, SOLO):
                source.write_text(code, encoding="utf-8")
                proposals.append(
                    run_candidate(source, points, DeploymentConfig(), 1234, 3)
                )
                self.assertEqual(source.read_text(encoding="utf-8"), code)
            self.assertEqual(proposals[0], proposals[1])

    def test_mixed_access_returns_identical_values_and_unknown_fields_fail(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "candidate.py"
            code = """def solve(turbines, config, seed):
    for t in turbines:
        if (t.identifier,t.x_km,t.y_km) != (t['identifier'],t['x_km'],t['y_km']):
            raise ValueError('input fields differ')
    return {'nest_ids':[t.identifier for t in turbines], 'assignments':{t['identifier']:t.identifier for t in turbines}}
"""
            source.write_text(code, encoding="utf-8")
            result = run_candidate(
                source, [Turbine("a", 1.2, -0.7)], DeploymentConfig(), 1, 3
            )
            self.assertEqual(result.nest_ids, ["a"])
            source.write_text(
                "def solve(turbines, config, seed):\n return turbines[0]['missing']\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(RuntimeError, "KeyError"):
                run_candidate(source, [Turbine("a", 0, 0)], DeploymentConfig(), 1, 3)

    def test_preflight_runs_before_real_data_and_does_not_rewrite_source(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            c = controller(
                settings_at(root, evaluation_budget=1), root, [MAPPING_SOURCE]
            )
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                path = c.run("mapping")
            row = c.history[0]
            self.assertEqual(row["preflight"]["status"], "passed")
            self.assertEqual(len(row["preflight"]["cases"]), 2)
            self.assertEqual(len(row["evaluation"]["datasets"]), 1)
            self.assertEqual(
                row["source_sha256"],
                hashlib.sha256(MAPPING_SOURCE.encode()).hexdigest(),
            )
            self.assertEqual(
                (path / "candidates/0001.py").read_text(encoding="utf-8"),
                MAPPING_SOURCE,
            )
            self.assertLess(
                out.getvalue().index("[preflight_start]"),
                out.getvalue().index("[dataset_start]"),
            )
            self.assertTrue(c.archive)

    def test_runtime_failure_repaired_on_next_round_before_more_initial_attempts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            c = controller(
                settings_at(root, evaluation_budget=3, initial_candidates=3),
                root,
                [RUNTIME_ERROR, SOLO, GROUP],
            )
            with contextlib.redirect_stdout(io.StringIO()):
                c.run("repair-first")
            first, repaired, independent = c.history
            self.assertEqual(first["evaluation"]["failure_stage"], "preflight")
            self.assertEqual(first["preflight"]["status"], "failed")
            self.assertEqual(first["evaluation"]["datasets"], [])
            self.assertEqual(repaired["parent_id"], "C0001")
            self.assertEqual(repaired["selection_reason"], "repair_failed_child")
            self.assertIn(RUNTIME_ERROR, c.client.prompts[1])
            self.assertIn("repair this exact failure", c.client.prompts[1])
            self.assertEqual(independent["selection_reason"], "independent_initial")
            self.assertTrue(c.archive)

    def test_preflight_timeout_never_becomes_deployable(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            c = controller(
                settings_at(root, evaluation_budget=1, candidate_timeout_seconds=1),
                root,
                ["def solve(turbines, config, seed):\n while True: pass\n"],
            )
            with contextlib.redirect_stdout(io.StringIO()):
                c.run("timeout")
            self.assertFalse(c.archive)
            self.assertEqual(c.history[0]["evaluation"]["failure_stage"], "preflight")
            self.assertIn("time limit", c.history[0]["evaluation"]["violations"][0])

    def test_interrupted_preflight_resumes_completed_cases_without_new_generation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            settings = settings_at(root, evaluation_budget=1)
            first = controller(settings, root, [SOLO])
            calls = []

            def interrupt(path, points, *args, **kwargs):
                calls.append(len(points))
                if len(calls) == 2:
                    raise KeyboardInterrupt
                return run_candidate(path, points, *args, **kwargs)

            with contextlib.redirect_stdout(io.StringIO()), patch(
                "stage1_llm.search.run_candidate", side_effect=interrupt
            ):
                with self.assertRaises(KeyboardInterrupt):
                    first.run("interrupted")
            state = json.loads(
                (root / "outputs/interrupted/search_state.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(len(state["pending"]["preflight"]["cases"]), 1)
            resumed = controller(settings, root, [])
            with contextlib.redirect_stdout(io.StringIO()), patch(
                "stage1_llm.search.run_candidate", wraps=run_candidate
            ) as check:
                resumed.run("interrupted", resume=True)
            self.assertEqual(
                check.call_count, 2
            )  # Remaining preflight + the real dataset.
            self.assertEqual(resumed.client.prompts, [])
            self.assertTrue(resumed.archive)

    def test_exception_after_successful_dataset_remains_in_repair_prompt(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            validation = root / "valid.tsv"
            validation.write_text("v1\t1\t1\nv2\t1.1\t1\n", encoding="utf-8")
            code = """def solve(turbines, config, seed):
    if turbines[0].identifier == 'v1':
        raise ValueError('validation runtime crash 123')
    return {'nest_ids':[t.identifier for t in turbines], 'assignments':{t.identifier:t.identifier for t in turbines}}
"""
            settings = settings_at(
                root, evaluation_budget=2, validation_inputs=(validation,)
            )
            c = controller(settings, root, [code, SOLO])
            with contextlib.redirect_stdout(io.StringIO()):
                c.run("later-error")
            self.assertEqual(len(c.history[0]["evaluation"]["datasets"]), 1)
            self.assertEqual(c.history[0]["evaluation"]["failure_stage"], "dataset")
            self.assertIn("validation runtime crash 123", c.client.prompts[1])


if __name__ == "__main__":
    unittest.main()
