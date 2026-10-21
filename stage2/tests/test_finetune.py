"""Independent warm-start experiments must never lose their baseline."""

from dataclasses import asdict, replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from stage2.data import split_fingerprint
from stage2.policy import build_policy
from stage2.tests.test_pool import small_instance
from stage2.train import TrainingConfig


class FineTuneTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.cfg = TrainingConfig(
            device="cpu",
            architecture="edge_attention_v2",
            hidden_dim=16,
            encoder_layers=1,
            heads=2,
            seed=1236,
            batch_size=2,
            rollouts_per_instance=2,
            epochs=2,
            batches_per_epoch=1,
            learning_rate=3e-5,
            lr_min=3e-6,
            lr_schedule_epochs=2,
            decode_strategy="best_of_n",
            decode_samples=4,
            decode_chunk_size=2,
        )
        self.train = [small_instance("T", 0, nodes=4), small_instance("U", 3, nodes=5)]
        self.val = [small_instance("V", 9, nodes=4)]

    def source(self, root):
        path = root / "parent.pt"
        torch.save(
            {
                "architecture": "edge_attention_v2",
                "model": build_policy(asdict(self.cfg)).state_dict(),
                "training_config": asdict(self.cfg),
                "epoch": 192,
                "wind_environment": asdict(self.train[0].config),
                "initial_training_fingerprint": split_fingerprint(self.train),
                "validation_fingerprint": split_fingerprint(self.val),
                "training_pool": None,
            },
            path,
        )
        return path

    def test_entropy_loss_zero_equivalence_gradient_and_schedule(self):
        from stage2.finetune import entropy_coefficient, reinforce_entropy_loss

        logits = torch.tensor([[1.0, 0.0]], requires_grad=True)
        dist = torch.distributions.Categorical(logits=logits)
        logp = dist.log_prob(torch.tensor([0])).reshape(1, 1)
        advantage = torch.ones(1, 1)
        entropy = dist.entropy()
        old = -(advantage * logp).mean()
        zero, policy, bonus = reinforce_entropy_loss(advantage, logp, entropy, 0.0)
        torch.testing.assert_close(old, zero, rtol=0, atol=0)
        torch.testing.assert_close(policy, old)
        self.assertEqual(bonus.item(), 0.0)
        loss, _, _ = reinforce_entropy_loss(advantage * 0, logp, entropy, 0.01)
        loss.backward()
        self.assertGreater(logits.grad.abs().sum().item(), 0)
        self.assertEqual(entropy_coefficient(0, 0.01, 0.0, 50), 0.01)
        self.assertEqual(entropy_coefficient(49, 0.01, 0.0, 50), 0.0)
        with self.assertRaises(ValueError):
            entropy_coefficient(0, -0.01, 0.0, 50)

    def test_two_arms_keep_parent_and_epoch_zero_and_reject_overwrite(self):
        from stage2.finetune import run_comparison

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self.source(root)
            before = source.read_bytes()
            report = run_comparison(
                source, self.cfg, root / "run", self.train, self.val
            )
            self.assertEqual(source.read_bytes(), before)
            a = torch.load(root / "run/control/epoch0.pt", weights_only=False)
            b = torch.load(root / "run/entropy/epoch0.pt", weights_only=False)
            self.assertEqual(a["validation_metrics"], b["validation_metrics"])
            for key in a["model"]:
                torch.testing.assert_close(
                    a["model"][key], b["model"][key], rtol=0, atol=0
                )
            self.assertEqual(
                report["selected_sampling"],
                json.loads((root / "run/comparison.json").read_text())[
                    "selected_sampling"
                ],
            )
            for arm in ("control", "entropy"):
                state = torch.load(root / f"run/{arm}/latest.pt", weights_only=False)
                self.assertEqual(state["epoch"], 2)
                for name in ("best.pt", "best_sampling.pt", "best_greedy.pt"):
                    self.assertTrue((root / f"run/{arm}/{name}").exists())
            with self.assertRaisesRegex(ValueError, "resume|exists"):
                run_comparison(source, self.cfg, root / "run", self.train, self.val)

    def test_no_improvement_preserves_epoch_zero(self):
        from stage2.finetune import run_comparison

        with tempfile.TemporaryDirectory() as directory, patch(
            "stage2.finetune.validation_is_better", return_value=False
        ):
            root = Path(directory)
            run_comparison(
                self.source(root), self.cfg, root / "run", self.train, self.val
            )
            state = torch.load(root / "run/entropy/best.pt", weights_only=False)
            self.assertEqual(state["epoch"], 0)
            self.assertEqual(
                torch.load(root / "run/entropy/best_greedy.pt", weights_only=False)[
                    "epoch"
                ],
                0,
            )

    def test_interrupted_arm_recovers_exactly_and_skips_completed_control(self):
        from stage2.finetune import run_comparison, write_json

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self.source(root)
            run_comparison(source, self.cfg, root / "full", self.train, self.val)

            def interrupt(path, value):
                write_json(path, value)
                if (
                    path.name == "training_history.json"
                    and path.parent.name == "entropy"
                    and len(value) == 1
                ):
                    raise RuntimeError("interrupt")

            with patch("stage2.finetune.write_json", side_effect=interrupt):
                with self.assertRaisesRegex(RuntimeError, "interrupt"):
                    run_comparison(
                        source, self.cfg, root / "part", self.train, self.val
                    )
            control_before = (root / "part/control/latest.pt").read_bytes()
            run_comparison(
                source, self.cfg, root / "part", self.train, self.val, resume=True
            )
            self.assertEqual(
                control_before, (root / "part/control/latest.pt").read_bytes()
            )
            a = torch.load(root / "full/entropy/latest.pt", weights_only=False)
            b = torch.load(root / "part/entropy/latest.pt", weights_only=False)
            for key in a["model"]:
                torch.testing.assert_close(
                    a["model"][key], b["model"][key], rtol=0, atol=0
                )
            self.assertEqual(a["scheduler"], b["scheduler"])
            with self.assertRaisesRegex(ValueError, "identity|setting"):
                run_comparison(
                    source,
                    replace(self.cfg, learning_rate=2e-5),
                    root / "part",
                    self.train,
                    self.val,
                    resume=True,
                )

    def test_greedy_and_sampling_best_epochs_are_independent(self):
        from stage2.finetune import run_comparison

        def report(main, greedy):
            return {
                "feasible_rate": 1.0,
                "mean_normalized_flight_cost": main,
                "greedy": {"feasible_rate": 1.0, "mean_normalized_flight_cost": greedy},
            }

        reports = [report(0.5, 0.6), report(0.4, 0.7), report(0.45, 0.3)] * 2
        with tempfile.TemporaryDirectory() as directory, patch(
            "stage2.finetune.evaluate_edge_policy", side_effect=reports
        ):
            root = Path(directory)
            result = run_comparison(
                self.source(root), self.cfg, root / "run", self.train, self.val
            )
            self.assertEqual(result["selected_sampling"], "control")
            self.assertEqual(result["selected_greedy"], "control")
            self.assertEqual(
                torch.load(root / "run/control/best_sampling.pt", weights_only=False)[
                    "epoch"
                ],
                1,
            )
            self.assertEqual(
                torch.load(root / "run/control/best_greedy.pt", weights_only=False)[
                    "epoch"
                ],
                2,
            )

    def test_cli_pool_refresh_and_resume_keep_identical_data_stream(self):
        from stage2.finetune import run_comparison
        from stage2.pool import TrainingPool
        from stage2.tests.test_pool import ROOT, fake_fields
        from stage2.run import main

        self.train = [small_instance("real", 0), *fake_fields(None, None, 2, 1234)]

        def factory():
            return TrainingPool(
                self.train,
                self.val,
                ROOT / "turbine102.tsv",
                ROOT / "stage1/config.json",
                refresh_every_epochs=1,
                refresh_fields=1,
            )

        with tempfile.TemporaryDirectory() as directory, patch(
            "stage2.pool.build_kde_instances", side_effect=fake_fields
        ):
            root = Path(directory)
            source = self.source(root)
            parent = torch.load(source, weights_only=False)
            original = factory()
            original.maybe_refresh(1)
            parent["training_pool"] = original.state_dict()
            torch.save(parent, source)
            data = root / "data"
            data.mkdir()
            (data / "data_manifest.json").write_text(
                json.dumps({"validation_kde_fields": 1})
            )
            argv = [
                "run",
                "finetune_compare",
                "--checkpoint",
                str(source),
                "--data-dir",
                str(data),
                "--output",
                str(root / "cli"),
                "--epochs",
                "2",
                "--device",
                "cpu",
            ]
            with patch("sys.argv", argv), patch(
                "stage2.run._prepare_or_load_dataset",
                return_value=(self.train, self.val, {}),
            ):
                self.assertEqual(main(), 0)
            a = torch.load(root / "cli/control/latest.pt", weights_only=False)
            b = torch.load(root / "cli/entropy/latest.pt", weights_only=False)
            self.assertEqual(a["training_pool"], b["training_pool"])
            self.assertEqual(a["training_pool"]["version"], 2)
            self.assertEqual(a["training_pool"]["next_field_id"], 4)
            with patch("sys.argv", [*argv, "--resume"]), patch(
                "stage2.run._prepare_or_load_dataset",
                return_value=(self.train, self.val, {}),
            ):
                self.assertEqual(main(), 0)
            # Wrong validation and source changes fail before creating an experiment.
            with self.assertRaisesRegex(ValueError, "fingerprint"):
                run_comparison(
                    source,
                    self.cfg,
                    root / "bad",
                    self.train,
                    [small_instance("bad", 100)],
                    pool_factory=factory,
                )

    def test_zero_and_single_task_entropy_are_finite_with_padding(self):
        from stage2.finetune import reinforce_entropy_loss
        from stage2.train import _batch_features, _pad_instance_batch

        model = build_policy(asdict(self.cfg))
        batch = [
            small_instance("Z", 0, nodes=1),
            small_instance("S", 3, nodes=2),
            self.train[0],
        ]
        coords, valid, times, hover, endurance = _pad_instance_batch(
            batch, torch.device("cpu")
        )
        cache = model.encode(
            _batch_features(coords, valid, batch),
            valid,
            time_matrix=times,
            hover=hover,
            endurance=endurance,
        )
        _, logp, details = model.decode(cache, rollouts=2, return_details=True)
        self.assertTrue(torch.isfinite(details["entropy"]).all())
        self.assertTrue(torch.equal(details["entropy"][:4], torch.zeros(4)))
        loss, _, _ = reinforce_entropy_loss(
            torch.zeros(3, 2), logp.reshape(3, 2), details["entropy"], 0.01
        )
        loss.backward()
        self.assertTrue(
            all(
                torch.isfinite(p.grad).all()
                for p in model.parameters()
                if p.grad is not None
            )
        )

    def test_resume_repairs_best_files_after_latest_commit(self):
        from stage2.finetune import run_comparison, _materialize_best

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self.source(root)

            def interrupt(state, output):
                if output.name == "control" and state["epoch"] == 1:
                    raise RuntimeError("latest committed before best files")
                _materialize_best(state, output)

            with patch("stage2.finetune._materialize_best", side_effect=interrupt):
                with self.assertRaisesRegex(RuntimeError, "latest committed"):
                    run_comparison(source, self.cfg, root / "run", self.train, self.val)
            latest = torch.load(root / "run/control/latest.pt", weights_only=False)
            self.assertEqual(latest["epoch"], 1)
            # Corrupt a derived file: recovery must not trust it over latest.
            (root / "run/control/best_greedy.pt").write_bytes(b"stale")
            run_comparison(
                source, self.cfg, root / "run", self.train, self.val, resume=True
            )
            best = torch.load(root / "run/control/best_greedy.pt", weights_only=False)
            self.assertGreaterEqual(best["epoch"], 0)
            old = source.read_bytes()
            parent = torch.load(source, weights_only=False)
            parent["epoch"] = 193
            torch.save(parent, source)
            changed = source.read_bytes()
            self.assertNotEqual(old, changed)
            with self.assertRaisesRegex(ValueError, "identity"):
                run_comparison(
                    source, self.cfg, root / "run", self.train, self.val, resume=True
                )
            self.assertEqual(changed, source.read_bytes())

    def test_invalid_source_mode_and_protected_settings_are_rejected(self):
        from stage2.finetune import run_comparison

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self.source(root)
            for cfg, pattern in (
                (replace(self.cfg, architecture="legacy"), "architecture"),
                (replace(self.cfg, energy_weight=0.5), "protected"),
            ):
                with self.assertRaisesRegex(ValueError, pattern):
                    run_comparison(source, cfg, root / "bad", self.train, self.val)
            with self.assertRaisesRegex(ValueError, "nonempty"):
                run_comparison(source, self.cfg, root / "bad", [], self.val)
            with self.assertRaisesRegex(ValueError, "contain"):
                run_comparison(source, self.cfg, root, self.train, self.val)
            with self.assertRaisesRegex(ValueError, "Missing"):
                run_comparison(
                    source, self.cfg, root / "bad", self.train, self.val, resume=True
                )
            (root / "bad/control").mkdir(parents=True)
            with self.assertRaisesRegex(ValueError, "orphan"):
                run_comparison(source, self.cfg, root / "bad", self.train, self.val)


if __name__ == "__main__":
    unittest.main()
