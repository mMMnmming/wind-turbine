"""Directed policy, policy sampling and leave-one-out RL contracts."""

from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from stage2.model import GraphPointerNetwork, make_node_features
from stage2.tests.test_pool import small_instance


class EdgePolicyTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(12)

    def inputs(self, nodes=5):
        instance = small_instance("D", 0, nodes=nodes)
        features = make_node_features(torch.tensor(instance.coordinates_km, dtype=torch.float32)[None], 6, 90)
        return instance, features, torch.tensor(instance.travel_time_seconds, dtype=torch.float32)[None]

    def test_directed_edges_receive_gradient_and_zero_tasks_work(self):
        from stage2.edge_model import EdgeAttentionPolicy
        model = EdgeAttentionPolicy(16, 1, 2)
        _, features, times = self.inputs()
        cache = model.encode(features, time_matrix=times, hover=torch.tensor([11.4]), endurance=torch.tensor([2460.]))
        route, logp, details = model.decode(cache, rollouts=4, return_details=True)
        self.assertEqual(route.shape, (4, 4))
        for row in route:
            self.assertEqual(sorted(row.tolist()), [1, 2, 3, 4])
        (-logp.mean()).backward()
        self.assertGreater(model.layers[0].edge_bias[0].weight.grad.abs().sum().item(), 0)
        self.assertTrue(torch.isfinite(details["entropy"]).all())
        zero = model.encode(features[:, :1], time_matrix=times[:, :1, :1],
                            hover=torch.tensor([11.4]), endurance=torch.tensor([2460.]))
        order, lp = model.decode(zero)
        self.assertEqual(order.shape, (1, 0))
        self.assertEqual(lp.item(), 0)

    def test_padding_and_permutation_leave_decisions_equivariant(self):
        from stage2.edge_model import EdgeAttentionPolicy
        model = EdgeAttentionPolicy(16, 1, 2).eval()
        _, features, times = self.inputs()
        kwargs = {"hover": torch.tensor([11.4]), "endurance": torch.tensor([2460.])}
        original = model.encode(features, time_matrix=times, **kwargs)
        perm = torch.tensor([0, 3, 1, 4, 2])
        changed = model.encode(features[:, perm], time_matrix=times[:, perm][:, :, perm], **kwargs)
        torch.testing.assert_close(original["encoded"][:, perm], changed["encoded"], atol=2e-6, rtol=2e-6)
        order, _ = model.decode(original, "greedy")
        shuffled, _ = model.decode(changed, "greedy")
        self.assertEqual(order.tolist(), perm[shuffled].tolist())
        padded = model.encode(torch.cat((features, torch.zeros(1, 2, 7)), 1),
                              time_matrix=torch.nn.functional.pad(times, (0, 2, 0, 2)),
                              valid_node_mask=torch.tensor([[1, 1, 1, 1, 1, 0, 0]], dtype=torch.bool), **kwargs)
        torch.testing.assert_close(original["encoded"], padded["encoded"][:, :5], atol=2e-6, rtol=2e-6)
        self.assertEqual(model.decode(padded, "greedy")[0].tolist(), order.tolist())
        reverse = model.encode(features, time_matrix=times.transpose(1, 2), **kwargs)
        self.assertFalse(torch.allclose(original["encoded"], reverse["encoded"]))

    def test_leave_one_out_excludes_self_and_detaches_advantage(self):
        from stage2.train_v2 import leave_one_out_advantage
        rewards = torch.tensor([[1., 2., 3.], [5., 5., 5.]], requires_grad=True)
        advantage = leave_one_out_advantage(rewards)
        torch.testing.assert_close(advantage, torch.tensor([[-1.5, 0., 1.5], [0., 0., 0.]]))
        self.assertFalse(advantage.requires_grad)
        with self.assertRaises(ValueError):
            leave_one_out_advantage(rewards[:, :1])

    def test_best_of_n_preserves_rng_includes_greedy_and_encodes_once(self):
        from unittest.mock import patch
        from stage2.edge_model import EdgeAttentionPolicy
        from stage2.policy import infer_instance
        instance, _, _ = self.inputs()
        for model in (GraphPointerNetwork(16, 1, 2).eval(), EdgeAttentionPolicy(16, 1, 2).eval()):
            before = torch.get_rng_state().clone()
            with patch.object(model, "encode", wraps=model.encode) as encoder:
                result = infer_instance(model, instance, torch.device("cpu"), "best_of_n", 16, 4, 1234)
                self.assertEqual(encoder.call_count, 1)
            self.assertTrue(torch.equal(before, torch.get_rng_state()))
            self.assertEqual(result["candidate_count"], 16)
            self.assertLessEqual(result["metrics"].flight_seconds, result["greedy_flight_seconds"] + 1e-8)
            again = infer_instance(model, instance, torch.device("cpu"), "best_of_n", 16, 4, 1234)
            self.assertEqual(result["route"], again["route"])
            self.assertGreater(result["runtime_seconds"], 0)

    def test_legacy_cached_greedy_is_unchanged_and_infeasible_selection_is_honest(self):
        from stage2.policy import infer_instance
        model = GraphPointerNetwork(16, 1, 2).eval()
        instance, features, _ = self.inputs()
        before, lp = model(features, "greedy")
        after, cached_lp = model.decode(model.encode(features), "greedy")
        torch.testing.assert_close(before, after, rtol=0, atol=0)
        torch.testing.assert_close(lp, cached_lp, rtol=0, atol=0)
        unsafe = replace(instance, config=replace(instance.config, max_endurance_seconds=1.))
        result = infer_instance(model, unsafe, torch.device("cpu"), "best_of_n", 8, 4, 7)
        self.assertFalse(result["metrics"].feasible)
        self.assertEqual(result["feasible_candidate_count"], 0)

    def test_dynamic_remaining_context_and_mixed_zero_task_mask(self):
        from unittest.mock import patch
        from stage2.edge_model import EdgeAttentionPolicy
        model = EdgeAttentionPolicy(16, 1, 2).eval()
        _, features, times = self.inputs()
        features = features.expand(3, -1, -1).clone()
        times = times.expand(3, -1, -1).clone()
        valid = torch.tensor([[1, 1, 1, 1, 1], [1, 1, 0, 0, 0], [1, 0, 0, 0, 0]], dtype=torch.bool)
        cache = model.encode(features, valid, time_matrix=times, hover=torch.tensor([11.4] * 3),
                             endurance=torch.tensor([2460.] * 3))
        contexts = []
        handle = model.context.register_forward_pre_hook(lambda module, args: contexts.append(args[0].detach().clone()))
        try:
            routes, logp = model.decode(cache, "greedy")
        finally:
            handle.remove()
        self.assertEqual(routes[1].tolist(), [1, 0, 0, 0])
        self.assertEqual(routes[2].tolist(), [0, 0, 0, 0])
        self.assertEqual(logp[2].item(), 0)
        self.assertTrue(torch.isfinite(logp).all())
        self.assertLess(contexts[1][0, -3], contexts[0][0, -3])
        self.assertLess(contexts[1][0, -2], contexts[0][0, -2])
        self.assertFalse(torch.equal(contexts[1][0, 16:32], contexts[0][0, 16:32]))

    def test_invalid_inputs_and_factory_architecture_are_rejected(self):
        from stage2.edge_model import EdgeAttentionPolicy
        from stage2.policy import build_policy, infer_instance
        model = EdgeAttentionPolicy(16, 1, 2)
        instance, features, times = self.inputs()
        kwargs = {"time_matrix": times, "hover": torch.tensor([11.4]), "endurance": torch.tensor([2460.])}
        for changed in ({"features": features[:, :, :6]}, {"valid_node_mask": torch.zeros(1, 5, dtype=torch.bool)},
                        {"time_matrix": times[:, :3, :3]}, {"time_matrix": -times},
                        {"hover": torch.tensor([-1.])}, {"endurance": torch.tensor([0.])}):
            with self.assertRaises(ValueError):
                model.encode(**{"features": features, **kwargs, **changed})
        with self.assertRaises(ValueError):
            build_policy({"architecture": "unknown"})
        for strategy, count, chunk, seed in (("bad", 8, 4, 1), ("best_of_n", 0, 4, 1), ("greedy", 1, 0, 1), ("greedy", 1, 1, -1)):
            with self.assertRaises(ValueError):
                infer_instance(model, instance, torch.device("cpu"), strategy, count, chunk, seed)
        cache = model.encode(features, **kwargs)
        with self.assertRaises(ValueError):
            model.decode(cache, "bad")

    def test_v2_zero_cost_arbitrary_depot_and_unsafe_candidates(self):
        from stage2.edge_model import EdgeAttentionPolicy
        from stage2.policy import infer_instance
        model = EdgeAttentionPolicy(16, 1, 2).eval()
        instance, _, _ = self.inputs()
        unsafe = replace(instance, depot_index=2, config=replace(instance.config, max_endurance_seconds=1.))
        result = infer_instance(model, unsafe, torch.device("cpu"), "best_of_n", 8, 4, 3)
        self.assertFalse(result["metrics"].feasible)
        self.assertEqual(result["route"][0], 2)
        zero = replace(instance, coordinates_km=np.zeros_like(instance.coordinates_km))
        self.assertEqual(infer_instance(model, zero, torch.device("cpu"), "best_of_n", 8, 4, 3)["metrics"].flight_seconds, 0)
        single = replace(instance, identifiers=("D",), coordinates_km=np.zeros((1, 2)))
        self.assertEqual(infer_instance(model, single, torch.device("cpu"))["runtime_seconds"], 0)

    def test_exact_resume_with_fixed_cosine_horizon_and_no_frozen_network(self):
        from stage2.train import TrainingConfig, train
        config = TrainingConfig(device="cpu", architecture="edge_attention_v2", hidden_dim=16,
                                encoder_layers=1, heads=2, batch_size=2, epochs=2, batches_per_epoch=1,
                                rollouts_per_instance=3, decode_strategy="best_of_n", decode_samples=8,
                                decode_chunk_size=4, lr_schedule_epochs=2)
        training = [small_instance("D", 0, nodes=4), small_instance("E", 3, nodes=5)]
        validation = [small_instance("V", 10, nodes=4)]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train(training, config, root / "full", validation)
            train(training, replace(config, epochs=1), root / "partial", validation)
            train(training, config, root / "resumed", validation, resume_checkpoint=root / "partial/latest.pt")
            full = torch.load(root / "full/latest.pt", weights_only=False)
            resumed = torch.load(root / "resumed/latest.pt", weights_only=False)
            self.assertNotIn("baseline", full)
            for name in full["model"]:
                torch.testing.assert_close(full["model"][name], resumed["model"][name], rtol=0, atol=0)
            self.assertEqual(full["scheduler"], resumed["scheduler"])
            self.assertIn("greedy", full["history"][-1]["validation"])
            self.assertIn("policy_entropy", full["history"][-1])
            with self.assertRaisesRegex(ValueError, "architecture"):
                train(training, replace(config, architecture="legacy"), root / "wrong", validation,
                      resume_checkpoint=root / "partial/latest.pt")

    def test_resume_recovers_latest_if_best_write_was_interrupted(self):
        from unittest.mock import patch
        from stage2.train import TrainingConfig, train, _save_checkpoint
        cfg = TrainingConfig(device="cpu", architecture="edge_attention_v2", hidden_dim=16,
                             encoder_layers=1, heads=2, epochs=2, batches_per_epoch=1, batch_size=1,
                             rollouts_per_instance=2, decode_strategy="best_of_n", decode_samples=4)
        training = [small_instance("D", 0, nodes=4)]
        validation = [small_instance("V", 9, nodes=4)]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train(training, replace(cfg, epochs=1), root, validation)
            def fail_on_best(payload, path):
                if path.name == "best.pt" and payload["epoch"] == 2:
                    raise RuntimeError("simulated interruption after latest commit")
                _save_checkpoint(payload, path)
            with patch("stage2.train_v2.validation_is_better", return_value=True), \
                    patch("stage2.train_v2._save_checkpoint", side_effect=fail_on_best):
                with self.assertRaisesRegex(RuntimeError, "simulated interruption"):
                    train(training, cfg, root, validation, resume_checkpoint=root / "latest.pt")
            self.assertEqual(torch.load(root / "best.pt", weights_only=False)["epoch"], 1)
            train(training, cfg, root, validation, resume_checkpoint=root / "latest.pt")
            self.assertEqual(torch.load(root / "best.pt", weights_only=False)["epoch"], 2)

    def test_v2_refresh_resume_preserves_pool_and_rejects_changed_settings(self):
        from unittest.mock import patch
        from stage2.pool import TrainingPool
        from stage2.tests.test_pool import ROOT, fake_fields
        from stage2.train import TrainingConfig, train
        cfg = TrainingConfig(device="cpu", architecture="edge_attention_v2", hidden_dim=16,
                             encoder_layers=1, heads=2, epochs=2, batches_per_epoch=1, batch_size=2,
                             rollouts_per_instance=2, decode_strategy="best_of_n", decode_samples=4)
        def make_pool():
            return TrainingPool([small_instance("real", 0), *fake_fields(None, None, 2, 1234)],
                                [small_instance("validation", 15, nodes=4)], ROOT / "turbine102.tsv",
                                ROOT / "stage1/config.json", refresh_every_epochs=1, refresh_fields=1)
        with tempfile.TemporaryDirectory() as directory, patch("stage2.pool.build_kde_instances", side_effect=fake_fields):
            root = Path(directory)
            full = make_pool()
            train(full.instances, cfg, root / "full", full.validation_instances, pool=full)
            part = make_pool()
            train(part.instances, replace(cfg, epochs=1), root / "part", part.validation_instances, pool=part)
            resumed = make_pool()
            train(resumed.instances, cfg, root / "resumed", resumed.validation_instances,
                  resume_checkpoint=root / "part/latest.pt", pool=resumed)
            a = torch.load(root / "full/latest.pt", weights_only=False)
            b = torch.load(root / "resumed/latest.pt", weights_only=False)
            self.assertEqual(a["training_pool"], b["training_pool"])
            for key in a["model"]:
                torch.testing.assert_close(a["model"][key], b["model"][key], rtol=0, atol=0)
            wrong = make_pool()
            with self.assertRaisesRegex(ValueError, "setting changed"):
                train(wrong.instances, replace(cfg, decode_samples=8), root / "wrong", wrong.validation_instances,
                      resume_checkpoint=root / "part/latest.pt", pool=wrong)

    def test_cli_new_model_evaluation_and_comparison_keep_decoding_metadata(self):
        import importlib
        import json
        from dataclasses import asdict
        from unittest.mock import patch
        from stage2.data import instance_to_record
        from stage2.policy import build_policy
        from stage2.run import main
        instance, _, _ = self.inputs()
        cfg = {"hidden_dim": 16, "encoder_layers": 1, "heads": 2, "architecture": "edge_attention_v2",
               "decode_strategy": "best_of_n", "decode_samples": 8, "decode_chunk_size": 4}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "regions.json"
            source.write_text(json.dumps({"regions": [instance_to_record(instance)]}))
            checkpoint = root / "best.pt"
            torch.save({"model": build_policy(cfg).state_dict(), "training_config": cfg,
                        "architecture": "edge_attention_v2", "wind_environment": asdict(instance.config)}, checkpoint)
            arguments = ["stage2.run", "evaluate", "--regions-json", str(source), "--checkpoint", str(checkpoint),
                         "--device", "cpu", "--random-samples", "1", "--plot-dpi", "30",
                         "--evaluation-output", str(root / "evaluation")]
            with patch("sys.argv", arguments):
                self.assertEqual(main(), 0)
            payload = json.loads((root / "evaluation/model_metrics.json").read_text())
            greedy = json.loads((root / "evaluation/greedy_metrics.json").read_text())
            self.assertEqual(payload["method"], "model_best_of_8")
            self.assertEqual(payload["candidate_count"], 8)
            self.assertEqual(greedy["method"], "model_greedy")
            solver = importlib.import_module("stage2.Branch and Bound.solver")
            compare = importlib.import_module("stage2.Branch and Bound.compare")
            report = compare.compare_results(payload, {"schema_version": 1, "results": [solver.solve_instance(instance, progress=None)]})
            self.assertEqual(report["model_method"], "model_best_of_8")
            self.assertEqual(report["model_candidate_count"], 8)
            self.assertGreaterEqual(report["summary"]["flight_gap_percent"], 0)
            with patch("sys.argv", arguments):
                with self.assertRaises(SystemExit):
                    main()


if __name__ == "__main__":
    unittest.main()
