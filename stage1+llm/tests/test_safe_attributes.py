"""Regressions for generated getattr/hasattr and unsupported builtin diagnostics."""

import contextlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from test_v3 import controller, settings_at
from stage1_llm.candidate import validate_candidate_source
from stage1_llm.domain import DeploymentConfig, Turbine
from stage1_llm.sandbox import run_candidate


class SafeAttributeTests(unittest.TestCase):
    def run_code(self, source):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "candidate.py"
            path.write_text(source, encoding="utf-8")
            return run_candidate(
                path,
                [Turbine("a", 0, 0), Turbine("b", 0.1, 0)],
                DeploymentConfig(),
                1,
                5,
            )

    def test_actual_server_getattr_statement_and_public_turbine_attributes(self):
        source = """def solve(turbines, config, seed):
    R = float(getattr(config, 'communication_radius_km', 7.0))
    if R != config.communication_radius_km:
        raise ValueError('configuration mismatch')
    ids = [getattr(t, 'identifier') for t in turbines]
    for t in turbines:
        if getattr(t, 'x_km') != t.x_km or getattr(t, 'y_km') != t.y_km:
            raise ValueError('coordinate mismatch')
    return {'nest_ids':ids,'assignments':{t:t for t in ids}}
"""
        result = self.run_code(source)
        self.assertEqual(result.assignments, {"a": "a", "b": "b"})

    def test_missing_public_attribute_default_hasattr_and_alias(self):
        source = """reader = getattr
def solve(turbines, config, seed):
    if not hasattr(config, 'communication_radius_km') or hasattr(config, 'not_a_field'):
        raise ValueError('hasattr mismatch')
    if reader(config, 'not_a_field', 99) != 99:
        raise ValueError('default mismatch')
    try:
        reader(config, 'not_a_field')
        raise ValueError('missing field was accepted')
    except AttributeError:
        pass
    ids = [t.identifier for t in turbines]
    return {'nest_ids':ids,'assignments':{t:t for t in ids}}
"""
        self.assertEqual(self.run_code(source).nest_ids, ["a", "b"])

    def test_dynamic_private_names_and_non_input_objects_rejected(self):
        for expression in (
            "getattr(config, '_'+'_'+'dict'+'_'+'_', {})",
            "hasattr(turbines[0], '_'+'identifier')",
            "getattr(math, 'sqrt', None)",
            "getattr(solve, 'not_a_field', None)",
            "getattr({}, 'get', None)",
        ):
            with self.subTest(expression=expression), self.assertRaisesRegex(
                RuntimeError, "public input|private"
            ):
                self.run_code(
                    f"def solve(turbines, config, seed):\n    {expression}\n    return {{}}\n"
                )

    def test_non_string_name_and_turbine_missing_attribute(self):
        for statement, error in (
            ("getattr(config, 12)", "TypeError"),
            ("getattr(turbines[0], 'missing')", "AttributeError"),
        ):
            with self.subTest(statement=statement), self.assertRaisesRegex(
                RuntimeError, error
            ):
                self.run_code(
                    f"def solve(turbines, config, seed):\n    {statement}\n    return {{}}\n"
                )

    def test_unsupported_builtin_is_rejected_before_process_creation(self):
        source = (
            'def solve(turbines, config, seed):\n    print("debug")\n    return {}\n'
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "candidate.py"
            path.write_text(source, encoding="utf-8")
            with patch(
                "stage1_llm.sandbox.subprocess.Popen",
                side_effect=AssertionError("must fail before spawn"),
            ):
                with self.assertRaisesRegex(
                    ValueError, "print.*not available|not available.*print"
                ):
                    run_candidate(path, [Turbine("a", 0, 0)], DeploymentConfig(), 1, 5)

    def test_scope_analysis_accepts_helpers_closures_imports_and_comprehensions(self):
        source = """SCALE = 2
def helper(value):
    return value * SCALE
def solve(turbines, config, seed):
    import math as geometry
    offset = geometry.sqrt(4)
    def nested(value):
        return helper(value) + offset
    ids = [t.identifier for t in turbines if nested(t.x_km) >= 0]
    return {'nest_ids':ids,'assignments':{t:t for t in ids}}
"""
        validate_candidate_source(source)
        self.assertEqual(self.run_code(source).nest_ids, ["a", "b"])

    def test_undefined_global_diagnostic_reaches_repair_prompt(self):
        source = "def solve(turbines, config, seed):\n    return unavailable_helper(turbines)\n"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repaired = "def solve(turbines, config, seed):\n return {'nest_ids':[t.identifier for t in turbines], 'assignments':{t.identifier:t.identifier for t in turbines}}\n"
            c = controller(
                settings_at(root, evaluation_budget=2), root, [source, repaired]
            )
            with contextlib.redirect_stdout(io.StringIO()):
                c.run("unknown-name")
            self.assertEqual(
                c.history[0]["evaluation"]["failure_stage"], "source_validation"
            )
            self.assertIn("unavailable_helper", c.client.prompts[1])
            self.assertTrue(c.archive)


if __name__ == "__main__":
    unittest.main()
