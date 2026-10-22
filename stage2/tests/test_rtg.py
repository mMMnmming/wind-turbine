"""Reward-to-go algebra, including asymmetric return and padding."""

import unittest
import tempfile
import json
from pathlib import Path
from dataclasses import asdict, replace
from unittest.mock import patch
import torch

from stage2.rtg import event_rewards, future_returns, step_advantage, step_policy_loss
from stage2.train import TrainingConfig, _reward


class RewardToGoTests(unittest.TestCase):
    def test_cli_preserves_active_pool_and_paired_refresh(self):
        from stage2.tests.test_finetune import FineTuneTests
        from stage2.tests.test_pool import ROOT, fake_fields, small_instance
        from stage2.pool import TrainingPool
        from stage2.run import main

        fixture = FineTuneTests()
        fixture.setUp()
        fixture.train = [small_instance("real", 0), *fake_fields(None, None, 2, 1234)]
        pool = TrainingPool(
            fixture.train,
            fixture.val,
            ROOT / "turbine102.tsv",
            ROOT / "stage1/config.json",
            refresh_every_epochs=1,
            refresh_fields=1,
        )
        with tempfile.TemporaryDirectory() as directory, patch(
            "stage2.pool.build_kde_instances", side_effect=fake_fields
        ):
            root = Path(directory)
            source = fixture.source(root)
            parent = torch.load(source, weights_only=False)
            pool.maybe_refresh(1)
            parent["training_pool"] = pool.state_dict()
            torch.save(parent, source)
            data = root / "data"
            data.mkdir()
            (data / "data_manifest.json").write_text(
                json.dumps({"validation_kde_fields": 1})
            )
            argv = [
                "run",
                "greedy_rtg_compare",
                "--checkpoint",
                str(source),
                "--data-dir",
                str(data),
                "--epochs",
                "2",
                "--batch-size",
                "2",
                "--batches-per-epoch",
                "1",
                "--device",
                "cpu",
                "--output",
                str(root / "run"),
            ]
            with patch("sys.argv", argv), patch(
                "stage2.run._prepare_or_load_dataset",
                return_value=(fixture.train, fixture.val, {}),
            ):
                self.assertEqual(main(), 0)
            a = torch.load(root / "run/trajectory/latest.pt", weights_only=False)
            b = torch.load(root / "run/reward_to_go/latest.pt", weights_only=False)
            self.assertEqual(a["training_pool"], b["training_pool"])
            self.assertEqual(a["training_pool"]["next_field_id"], 4)
            self.assertEqual(a["training_config"]["decode_strategy"], "greedy")
            self.assertEqual(a["training_config"]["decode_samples"], 1)
            self.assertEqual(
                a["experiment_identity"]["experiment_type"], "greedy_rtg_compare"
            )

    def test_events_and_return(self):
        times = torch.tensor(
            [[[0.0, 2.0, 9.0], [7.0, 0.0, 3.0], [5.0, 8.0, 0.0]]]
        ).repeat(3, 1, 1)
        routes = torch.tensor([[1, 2], [1, 0], [0, 0]])
        counts = torch.tensor([2, 1, 0])
        hover = torch.tensor([4.0, 4.0, 4.0])
        limit = torch.tensor([12.0, 100.0, 100.0])
        flight = torch.tensor([10.0, 9.0, 0.0])
        energy = flight + counts * hover
        cfg = TrainingConfig()
        events = event_rewards(times, routes, counts, hover, limit, cfg)
        expected = _reward(flight, energy, energy <= limit, limit, cfg)
        torch.testing.assert_close(events.sum(1), expected)
        torch.testing.assert_close(future_returns(events)[:, 0], expected)
        self.assertEqual(events[2, 0].item(), cfg.completion_bonus)
        self.assertEqual(events[2, 1:].abs().sum().item(), 0)

    def test_loo_detach_and_gradient_equivalence(self):
        rewards = torch.tensor([[1.0, 3.0, 8.0]], requires_grad=True)
        returns = rewards[:, :, None].expand(-1, -1, 4)
        advantages = step_advantage(returns)
        self.assertFalse(advantages.requires_grad)
        torch.testing.assert_close(advantages[0, :, 0], torch.tensor([-4.5, -1.5, 6.0]))
        logp = torch.randn(1, 3, 4, requires_grad=True)
        mask = torch.tensor([[[True, True, False, False]]]).expand_as(logp)
        actual = step_policy_loss(advantages, logp, mask)
        expected = -(advantages[:, :, 0] * (logp * mask).sum(-1)).mean()
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(
            torch.autograd.grad(actual, logp)[0], torch.autograd.grad(expected, logp)[0]
        )

    def test_reverse_sum_and_zero_actions(self):
        torch.testing.assert_close(
            future_returns(torch.tensor([[1.0, 2.0, 4.0]])),
            torch.tensor([[7.0, 6.0, 4.0]]),
        )
        empty = torch.empty(2, 8, 0, requires_grad=True)
        loss = step_policy_loss(empty.detach(), empty, empty.bool())
        loss.backward()
        self.assertEqual(loss.item(), 0.0)

    def test_decoder_and_paired_two_epoch_resume(self):
        from stage2.tests.test_finetune import FineTuneTests
        from stage2.finetune import run_comparison
        from stage2.train import _pad_instance_batch, _batch_features
        from stage2.policy import build_policy

        fixture = FineTuneTests()
        fixture.setUp()
        cfg = replace(fixture.cfg, decode_strategy="greedy", decode_samples=1)
        model = build_policy(asdict(cfg))
        batch = fixture.train
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
        state = torch.get_rng_state()
        routes, logp, details = model.decode(cache, rollouts=2, return_details=True)
        torch.set_rng_state(state)
        old_routes, old_logp = model.decode(cache, rollouts=2)
        torch.testing.assert_close(routes, old_routes)
        torch.testing.assert_close(logp, old_logp)
        torch.testing.assert_close(details["step_log_probs"].sum(1), logp)
        self.assertEqual(details["step_mask"].sum().item(), 14)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = fixture.source(root)
            report = run_comparison(
                source, cfg, root / "run", fixture.train, fixture.val, greedy_only=True
            )
            self.assertEqual(set(report["arms"]), {"trajectory", "reward_to_go"})
            self.assertNotIn("sampling", report["arms"]["reward_to_go"])
            a = torch.load(root / "run/trajectory/epoch0.pt", weights_only=False)
            b = torch.load(root / "run/reward_to_go/epoch0.pt", weights_only=False)
            for key in a["model"]:
                torch.testing.assert_close(
                    a["model"][key], b["model"][key], rtol=0, atol=0
                )
            self.assertEqual(a["validation_metrics"], b["validation_metrics"])
            for arm in report["arms"]:
                best = torch.load(root / f"run/{arm}/best.pt", weights_only=False)
                greedy = torch.load(
                    root / f"run/{arm}/best_greedy.pt", weights_only=False
                )
                self.assertEqual(best["epoch"], greedy["epoch"])
            with patch(
                "stage2.finetune.evaluate_edge_policy",
                side_effect=AssertionError("completed arm reran"),
            ):
                resumed = run_comparison(
                    source,
                    cfg,
                    root / "run",
                    fixture.train,
                    fixture.val,
                    greedy_only=True,
                    resume=True,
                )
            self.assertEqual(report, resumed)

    def test_interruption_restores_exact_rtg_weights(self):
        from stage2.tests.test_finetune import FineTuneTests
        from stage2.finetune import run_comparison, write_json

        fixture = FineTuneTests()
        fixture.setUp()
        cfg = replace(fixture.cfg, decode_strategy="greedy", decode_samples=1)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = fixture.source(root)
            run_comparison(
                source, cfg, root / "full", fixture.train, fixture.val, greedy_only=True
            )

            def interrupt(path, value):
                write_json(path, value)
                if (
                    path.name == "training_history.json"
                    and path.parent.name == "reward_to_go"
                    and len(value) == 1
                ):
                    raise RuntimeError("interrupted")

            with patch("stage2.finetune.write_json", side_effect=interrupt):
                with self.assertRaisesRegex(RuntimeError, "interrupted"):
                    run_comparison(
                        source,
                        cfg,
                        root / "part",
                        fixture.train,
                        fixture.val,
                        greedy_only=True,
                    )
            completed = (root / "part/trajectory/latest.pt").read_bytes()
            (root / "part/reward_to_go/best.pt").write_bytes(b"interrupted best write")
            run_comparison(
                source,
                cfg,
                root / "part",
                fixture.train,
                fixture.val,
                greedy_only=True,
                resume=True,
            )
            self.assertEqual(
                completed, (root / "part/trajectory/latest.pt").read_bytes()
            )
            a = torch.load(root / "full/reward_to_go/latest.pt", weights_only=False)
            b = torch.load(root / "part/reward_to_go/latest.pt", weights_only=False)
            for key in a["model"]:
                torch.testing.assert_close(
                    a["model"][key], b["model"][key], rtol=0, atol=0
                )
            self.assertEqual(a["scheduler"], b["scheduler"])
            self.assertEqual(a["training_pool"], b["training_pool"])

    def test_decoder_zero_single_padding(self):
        from stage2.tests.test_pool import small_instance
        from stage2.policy import build_policy
        from stage2.train import _pad_instance_batch, _batch_features

        cfg = TrainingConfig(
            architecture="edge_attention_v2", hidden_dim=16, heads=2, encoder_layers=1
        )
        model = build_policy(asdict(cfg))
        for batch in (
            [small_instance("empty", 0, nodes=1)],
            [small_instance("zero", 0, nodes=1), small_instance("single", 2, nodes=2)],
        ):
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
            routes, logp, details = model.decode(cache, rollouts=2, return_details=True)
            torch.testing.assert_close(details["step_log_probs"].sum(1), logp)
            self.assertEqual(
                details["step_mask"].sum().item(), 2 * (valid.sum(1) - 1).sum().item()
            )
            events = event_rewards(
                times.repeat_interleave(2, 0),
                routes,
                (valid.sum(1) - 1).repeat_interleave(2),
                hover.repeat_interleave(2),
                endurance.repeat_interleave(2),
                cfg,
            )
            steps = routes.size(1)
            adv = step_advantage(
                future_returns(events)[:, :steps].reshape(len(batch), 2, steps)
            )
            loss = step_policy_loss(
                adv,
                details["step_log_probs"].reshape_as(adv),
                details["step_mask"].reshape_as(adv),
            )
            self.assertTrue(torch.isfinite(loss))
            loss.backward()


if __name__ == "__main__":
    unittest.main()
