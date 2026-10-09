"""CLI defaults and split-output isolation tests."""

from __future__ import annotations

import unittest
from pathlib import Path

from stage2.run import ROOT, _resolve_input_path


class RunCliTests(unittest.TestCase):
    def test_training_and_evaluation_default_to_separate_datasets(self) -> None:
        self.assertEqual(_resolve_input_path("train", None), ROOT.parent / "turbine102.tsv")
        self.assertEqual(_resolve_input_path("evaluate", None), ROOT.parent / "turbine87.tsv")

    def test_explicit_input_path_is_preserved(self) -> None:
        explicit = Path("custom-test.tsv")
        self.assertEqual(_resolve_input_path("evaluate", explicit), explicit)


if __name__ == "__main__":
    unittest.main()
