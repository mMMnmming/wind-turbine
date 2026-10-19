"""Small end-to-end RL runs: refresh, fixed validation, and exact CPU resume."""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from stage2.pool import TrainingPool
from stage2.train import TrainingConfig, train
from stage2.tests.test_pool import ROOT, fake_fields, small_instance


class RefreshTrainingTests(unittest.TestCase):
    def make_pool(self) -> TrainingPool:
        instances = [small_instance("real-depot", 0.0), *fake_fields(None, None, 2, 1234)]
        validation = [small_instance("validation-depot", 20.0, nodes=4)]
        return TrainingPool(instances, validation, ROOT / "turbine102.tsv", ROOT / "stage1/config.json",
                            seed=1234, refresh_every_epochs=1, refresh_fields=1)

    @patch("stage2.pool.build_kde_instances", side_effect=fake_fields)
    def test_resume_restores_pool_and_rng_instead_of_repeating_training(self, builder) -> None:
        config = TrainingConfig(device="cpu", hidden_dim=8, encoder_layers=1, heads=2,
                                batch_size=2, epochs=3, batches_per_epoch=1, stratified_sampling=True)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            full = self.make_pool()
            train(full.instances, config, root / "full", full.validation_instances, pool=full)
            partial = self.make_pool()
            train(partial.instances, replace(config, epochs=1), root / "partial",
                  partial.validation_instances, pool=partial)
            resumed = self.make_pool()
            train(resumed.instances, config, root / "resumed", resumed.validation_instances,
                  resume_checkpoint=root / "partial/latest.pt", pool=resumed)
            full_state = torch.load(root / "full/latest.pt", weights_only=False)
            resumed_state = torch.load(root / "resumed/latest.pt", weights_only=False)
            self.assertEqual(full_state["training_pool"], resumed_state["training_pool"])
            for key in full_state["model"]:
                torch.testing.assert_close(full_state["model"][key], resumed_state["model"][key], rtol=0, atol=0)
            self.assertEqual(len(resumed_state["history"]), 3)
            self.assertEqual([row["pool_version"] for row in full_state["history"]], [0, 1, 2])
            self.assertIn("mean_relative_nn_improvement", full_state["history"][-1]["validation"])
            self.assertIn("validation_fingerprint", full_state)
            self.assertTrue((root / "full/pool_history/pool_0002.json").exists())
            self.assertEqual(json.loads((root / "resumed/training_history.json").read_text())[-1]["epoch"], 3)

    def test_changed_validation_is_rejected_on_resume(self) -> None:
        config = TrainingConfig(device="cpu", hidden_dim=8, encoder_layers=1, heads=2,
                                batch_size=2, epochs=1, batches_per_epoch=1)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pool = self.make_pool()
            train(pool.instances, config, root / "first", pool.validation_instances)
            changed = [small_instance("another-validation", 90.0, nodes=5)]
            with self.assertRaisesRegex(ValueError, "validation"):
                train(pool.instances, replace(config, epochs=2), root / "second", changed,
                      resume_checkpoint=root / "first/latest.pt")

    def test_resume_in_place_when_latest_is_the_selected_best(self) -> None:
        config = TrainingConfig(device="cpu", hidden_dim=8, encoder_layers=1, heads=2,
                                batch_size=2, epochs=1, batches_per_epoch=1)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pool = self.make_pool()
            train(pool.instances, config, root, pool.validation_instances)
            train(pool.instances, replace(config, epochs=2), root, pool.validation_instances,
                  resume_checkpoint=root / "latest.pt")
            latest = torch.load(root / "latest.pt", weights_only=False)
            self.assertEqual(latest["epoch"], 2)


if __name__ == "__main__":
    unittest.main()
