"""Offline v5 acceptance: no paid provider and no production checkpoint."""

import copy
import contextlib
import dataclasses
import io
import json
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from stage1_llm.domain import DeploymentConfig, Turbine
from stage1_llm.joint_tools import JointTools, aggregate_metrics, ranking_key
from stage1_llm.joint_planner import CheckedPlanner, InfrastructureError


class FakeDecoder:
    """Input-order tour, explicitly NOT a trained model or NN route."""

    metadata = {"origin": "offline_mock", "checkpoint_sha256": "test"}

    def __init__(self):
        self.calls = 0

    def decode(self, instance):
        self.calls += 1
        return tuple(range(instance.node_count)) + (0,), 0.001


class JointContracts(unittest.TestCase):
    def setUp(self):
        self.points = [Turbine("a", 0, 0), Turbine("b", 1, 0), Turbine("c", 1, 1)]
        self.config = DeploymentConfig()
        self.decoder = FakeDecoder()
        self.planner = CheckedPlanner(self.decoder, self.config, 12.0)

    def tools(self, **kwargs):
        return JointTools(self.points, self.config, planner=self.planner, **kwargs)

    def test_network_wind_return_hover_and_cache(self):
        tools = self.tools()
        with patch(
            "stage1_llm.routing.proxy_route", side_effect=AssertionError("NN forbidden")
        ), patch(
            "stage1_llm.domain.mission_metrics",
            side_effect=AssertionError("NN forbidden"),
        ):
            first = tools.region("a", ["c", "a", "b"])
            self.assertTrue(first["feasible"])
            self.assertEqual(first["route_ids"], ["a", "b", "c", "a"])
            self.assertAlmostEqual(
                first["energy_seconds"] - first["flight_seconds"], 22.8
            )
            self.assertEqual(first, tools.region("a", ["a", "b", "c"]))
            self.assertEqual(self.decoder.calls, 1)
            self.assertNotEqual(
                tools.directed_time("a", "b"), tools.directed_time("b", "a")
            )

    def test_singleton_radius_and_zero_work(self):
        tools = self.tools()
        row = tools.region("a", ["a"])
        self.assertEqual(row["energy_seconds"], 0)
        self.assertEqual(row["route_ids"], ["a", "a"])
        far = JointTools(
            [Turbine("a", 0, 0), Turbine("b", 8, 0)], self.config, planner=self.planner
        )
        self.assertFalse(far.region("a", ["a", "b"])["feasible"])
        self.assertEqual(self.decoder.calls, 0)

    def test_ranking_and_average(self):
        self.assertLess(ranking_key([4, 9000, 8000]), ranking_key([5, 100, 100]))
        self.assertLess(ranking_key([4, 100, 90]), ranking_key([4, 101, 80]))
        self.assertLess(ranking_key([4, 100, 80]), ranking_key([4, 100, 90]))
        checked = self.tools().deployment(
            {"nest_ids": ["a"], "assignments": {t.identifier: "a" for t in self.points}}
        )
        self.assertEqual(
            checked["metrics"]["mean_flight_seconds"],
            checked["metrics"]["total_flight_seconds"],
        )
        self.assertEqual(checked["objectives"][0], 1)
        with self.assertRaises(TypeError):
            self.tools().region("a", ["a"], proxy=False)

    def test_failures_and_actual_route_authority(self):
        bad = copy.copy(self.decoder)
        bad.decode = lambda instance: ((0, 0), 0.0)
        with self.assertRaises(InfrastructureError):
            CheckedPlanner(bad, self.config, 12).region(self.points, "a")
        tight = dataclasses.replace(self.config, max_endurance_seconds=1.0)
        planner = CheckedPlanner(self.decoder, tight, 12)
        row = JointTools(self.points, tight, planner=planner).region("a", ["a", "b"])
        self.assertFalse(row["feasible"])
        self.assertEqual(row["diagnostics"][0]["kind"], "network_route_endurance")

    def test_repairs_and_saved_copy(self):
        tools = self.tools()
        proposal = tools.repartition_repair({"a": "a", "b": "a", "c": "a"})
        self.assertTrue(tools.remember(proposal))
        proposal["assignments"]["b"] = "b"
        self.assertEqual(tools.best()["assignments"]["b"], "a")
        self.assertTrue(
            tools.deployment(tools.split_repair({"a": "a", "b": "a", "c": "a"}))[
                "feasible"
            ]
        )

    def test_candidate_rpc_and_swallowed_contract_failure(self):
        from stage1_llm.sandbox import run_candidate, CandidateExecutionError

        with tempfile.TemporaryDirectory() as temporary:
            candidate = Path(temporary) / "candidate.py"
            candidate.write_text(
                "from candidate_tools import region, make_proposal\ndef solve(turbines, config, seed):\n ids=[t.identifier for t in turbines]\n r=region(ids[0],ids)\n return make_proposal({t:ids[0] for t in ids})\n",
                encoding="utf-8",
            )
            result = run_candidate(
                candidate, self.points, self.config, 1234, 10, planner=self.planner
            )
            self.assertEqual(result.nest_ids, ["a"])
            candidate.write_text(
                "from candidate_tools import region, make_proposal\ndef solve(turbines, config, seed):\n try:\n  region(('bad',0),[])\n except Exception:\n  pass\n return make_proposal({t.identifier:t.identifier for t in turbines})\n",
                encoding="utf-8",
            )
            with self.assertRaises(CandidateExecutionError):
                run_candidate(
                    candidate, self.points, self.config, 1234, 10, planner=self.planner
                )

    def test_wind_and_no_radius_conservative_truncation(self):
        from stage2.atsp import WindATSPConfig, WindATSPInstance
        import numpy as np

        row = self.planner.region(self.points, "a")
        changed = dataclasses.replace(self.config, wind_direction_degrees=0)
        other = CheckedPlanner(self.decoder, changed, 12).region(self.points, "a")
        self.assertNotEqual(row["cache_key"], other["cache_key"])
        instance = WindATSPInstance(
            ("a", "b", "c"), np.array([(0, 0), (1, 0), (1, 1)]), 0, WindATSPConfig()
        )
        self.assertEqual(
            row["flight_seconds"], instance.evaluate_route((0, 1, 2, 0)).flight_seconds
        )

    def test_v5_offline_flow(self):
        from stage1_llm.joint_smoke import smoke_test

        with tempfile.TemporaryDirectory() as temporary:
            with contextlib.redirect_stdout(io.StringIO()):
                root = smoke_test(Path(temporary), "offline")
            state = json.loads((root / "search_state.json").read_text(encoding="utf-8"))
            self.assertEqual(len(state["history"]), 6)
            self.assertEqual(state["search_version"], 5)
            self.assertTrue(state["archive"])

    def test_checkpoint_direct_and_worker_match_without_training(self):
        import torch
        from stage2.policy import build_policy, infer_instance
        from stage2.atsp import WindATSPConfig, WindATSPInstance
        from stage1_llm.joint_planner import GreedyDecoder, PlannerService, environment
        import numpy as np

        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary) / "test-policy.pt"
            config = {
                "architecture": "edge_attention_v2",
                "hidden_dim": 16,
                "encoder_layers": 1,
                "heads": 2,
            }
            torch.manual_seed(1234)
            model = build_policy(config)
            env = environment(self.config, 12)
            torch.save(
                {
                    "training_config": config,
                    "model": model.state_dict(),
                    "wind_environment": env,
                },
                checkpoint,
            )
            decoder = GreedyDecoder(checkpoint, "cpu", self.config, 12)
            original = {k: v.clone() for k, v in decoder.model.state_dict().items()}
            instance = WindATSPInstance(
                ("a", "b", "c"),
                np.array([(0, 0), (1, 0), (1, 1)]),
                0,
                WindATSPConfig(**env),
            )
            expected = infer_instance(
                decoder.model,
                instance,
                torch.device("cpu"),
                strategy="greedy",
                candidates=1,
            )
            route, _ = decoder.decode(instance)
            self.assertEqual(route, expected["route"])
            self.assertTrue(
                all(
                    torch.equal(original[k], v)
                    for k, v in decoder.model.state_dict().items()
                )
            )
            service = PlannerService(checkpoint, "cpu", self.config, 12, timeout=30)
            try:
                checked = service.region(self.points, "a")
                self.assertEqual(
                    checked["route_ids"], [instance.identifiers[i] for i in route]
                )
                self.assertEqual(
                    checked["flight_seconds"], expected["metrics"].flight_seconds
                )
                self.assertTrue(service.region(self.points, "a")["cache_hit"])
                self.assertEqual(service.stats()["inferences"], 1)
                with self.assertRaises(ValueError):
                    GreedyDecoder(
                        checkpoint,
                        "cpu",
                        dataclasses.replace(self.config, wind_direction_degrees=0),
                        12,
                    )
            finally:
                service.close()

    def test_missing_checkpoint_worker_is_infrastructure_not_route_failure(self):
        from stage1_llm.joint_planner import PlannerService

        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaises(InfrastructureError):
                PlannerService(
                    Path(temporary) / "missing.pt", "cpu", self.config, timeout=30
                )

    def test_contract_error_details_and_json_normalization(self):
        from stage1_llm.client import parse_model_response, REQUIRED_RESPONSE_FIELDS
        from stage1_llm.candidate import validate_candidate_source

        content = {k: "value" for k in REQUIRED_RESPONSE_FIELDS}
        content["type"] = "json_object"
        self.assertEqual(
            parse_model_response(json.dumps(content)).algorithm_idea, "value"
        )
        content["unknown"] = "bad"
        with self.assertRaises(ValueError):
            parse_model_response(json.dumps(content))
        with self.assertRaisesRegex(ValueError, "__class__.*line 2"):
            validate_candidate_source(
                "def solve(turbines, config, seed):\n return seed.__class__\n"
            )

    def test_proxy_removed_and_schema_failure_not_swallowable(self):
        from stage1_llm.candidate_tools import ToolContractError, BudgetExceeded

        tools = self.tools(max_region_evaluations=1)
        tools.region("a", ["a", "b"])
        with self.assertRaises(BudgetExceeded):
            tools.region("b", ["a", "b"])
        with self.assertRaises(ToolContractError):
            self.tools().better_metrics([True, 1, 1], [1, 1, 1])
        bad = self.tools().deployment({"nest_ids": ["a"], "assignments": {"a": "a"}})
        self.assertFalse(bad["feasible"])

    def test_real_network_can_pass_when_legacy_audit_fails(self):
        from stage1_llm.domain import mission_metrics

        # Winds reduce actual eastbound cost; no legacy 17 m/s audit may veto it.
        points = [Turbine("a", 0, 0), Turbine("b", 1, 0)]
        _, _, _, old_energy = mission_metrics(points[0], tuple(points), self.config)
        new_energy = CheckedPlanner(self.decoder, self.config, 12).region(points, "a")[
            "energy_seconds"
        ]
        self.assertLess(new_energy, old_energy)
        config = dataclasses.replace(
            self.config, max_endurance_seconds=(old_energy + new_energy) / 2
        )
        tools = JointTools(
            points, config, planner=CheckedPlanner(self.decoder, config, 12)
        )
        with patch(
            "stage1_llm.routing.proxy_route", side_effect=AssertionError("forbidden")
        ):
            self.assertTrue(
                tools.deployment(
                    {"nest_ids": ["a"], "assignments": {"a": "a", "b": "a"}}
                )["feasible"]
            )

    def test_candidate_deadline_drains_reply_without_infrastructure_failure(self):
        from stage1_llm.joint_planner import PlannerService, CandidateDeadlineExceeded
        from stage1_llm.sandbox import run_candidate

        class DelayedPipe:
            def send(pipe, request):
                pipe.ready = time.perf_counter() + 1.0
                pipe.result = self.planner.region(self.points, "a")

            def poll(pipe, timeout):
                time.sleep(min(max(0.0, pipe.ready - time.perf_counter()), timeout))
                return time.perf_counter() >= pipe.ready

            def recv(pipe):
                return {"ok": True, "result": pipe.result, "stats": {"inferences": 1}}

        service = PlannerService.__new__(PlannerService)
        service.connection = DelayedPipe()
        service.timeout = 3.0
        service.metadata = self.planner.metadata
        service.counters = {}
        service._pending_started = None
        with self.assertRaises(CandidateDeadlineExceeded):
            service.region(self.points, "a", timeout=0.01)
        service.settle()
        self.assertIsNone(service._pending_started)
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "slow.py"
            source.write_text(
                "from candidate_tools import region\ndef solve(turbines,config,seed):\n region(turbines[0].identifier,[t.identifier for t in turbines])\n return {}\n",
                encoding="utf-8",
            )
            with self.assertRaises(TimeoutError) as caught:
                run_candidate(
                    source, self.points, self.config, 1234, 0.8, planner=service
                )
            self.assertNotIsInstance(caught.exception, InfrastructureError)
            self.assertIsNone(service._pending_started)

    def test_soft_budget_counts_inference_wait(self):
        from stage1_llm.candidate_tools import BudgetExceeded

        slow = copy.copy(self.planner)
        original = slow.region

        def region(points, nest):
            time.sleep(0.04)
            return original(points, nest)

        slow.region = region
        tools = JointTools(self.points, self.config, planner=slow, soft_seconds=0.02)
        with self.assertRaises(BudgetExceeded):
            tools.region("a", ["a", "b"])

    def test_export_uses_existing_gap_interface(self):
        from stage1_llm.joint_artifacts import export_deployment
        from stage1_llm.candidate import CandidateProposal
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "joint_compare",
            Path(__file__).resolve().parents[2] / "stage2/Branch and Bound/compare.py",
        )
        compare = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(compare)
        proposal = CandidateProposal(["a"], {p.identifier: "a" for p in self.points})
        checked = self.tools().deployment(dataclasses.asdict(proposal))
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            export_deployment(
                root,
                self.points,
                proposal,
                checked,
                self.planner.metadata,
                self.config,
                plots=True,
            )
            model = json.loads(
                (root / "model_metrics.json").read_text(encoding="utf-8")
            )
            record = model["results"][0]
            self.assertEqual(record["route"], [0, 1, 2, 0])
            reference = {
                "schema_version": 1,
                "results": [
                    dict(
                        record, status=9, termination="TIME_LIMIT", proven_optimal=False
                    )
                ],
            }
            result = compare.compare_results(model, reference)
            self.assertIsNone(result["rows"][0]["flight_gap_percent"])
            self.assertTrue((root / "region_01_network_route.png").is_file())
            with self.assertRaises(FileExistsError):
                export_deployment(
                    root,
                    self.points,
                    proposal,
                    checked,
                    self.planner.metadata,
                    self.config,
                )

    def test_repair_invalid_shapes_and_forced_split(self):
        from stage1_llm.candidate_tools import ToolContractError

        tools = self.tools()
        for invalid in ("a", {"a": "a"}, [["a"]]):
            with self.assertRaises(ToolContractError):
                tools.region("a", invalid)
        self.assertFalse(tools.region("a", ["b"])["feasible"])
        self.assertFalse(tools.region("a", ["a", "a"])["feasible"])
        with self.assertRaises(ToolContractError):
            tools.repartition_repair({"a": "a"})
        short = dataclasses.replace(self.config, max_endurance_seconds=1.0)
        tight = JointTools(
            self.points, short, planner=CheckedPlanner(self.decoder, short, 12)
        )
        answer = tight.repartition_repair({p.identifier: "a" for p in self.points})
        self.assertEqual(len(answer["nest_ids"]), 3)
        tight = JointTools(
            self.points, short, planner=CheckedPlanner(self.decoder, short, 12)
        )
        answer = tight.split_repair({p.identifier: "a" for p in self.points})
        self.assertEqual(len(answer["nest_ids"]), 3)
        self.assertFalse(tight.remember({"nest_ids": ["a"], "assignments": {"a": "a"}}))

    def test_worker_message_protocol_offline(self):
        from stage1_llm.joint_planner import _serve

        class Connection:
            def __init__(connection, requests):
                connection.requests = iter(requests)
                connection.responses = []
                connection.closed = False

            def recv(connection):
                return next(connection.requests)

            def send(connection, message):
                connection.responses.append(message)

            def close(connection):
                connection.closed = True

        request = {"points": [dataclasses.asdict(p) for p in self.points], "nest": "a"}
        connection = Connection([request, None])
        with patch("stage1_llm.joint_planner.GreedyDecoder", return_value=self.decoder):
            _serve(connection, "offline", "cpu", dataclasses.asdict(self.config), 12)
        self.assertTrue(connection.responses[0]["ok"])
        self.assertTrue(connection.responses[1]["result"]["feasible"])
        self.assertTrue(connection.closed)
        connection = Connection([])
        with patch(
            "stage1_llm.joint_planner.GreedyDecoder",
            side_effect=ValueError("test load fault"),
        ):
            _serve(connection, "offline", "cpu", dataclasses.asdict(self.config), 12)
        self.assertFalse(connection.responses[0]["ok"])

    def test_rank_validation_aggregation_and_lock(self):
        from stage1_llm.joint_search import run_lock

        with self.assertRaises(ValueError):
            ranking_key([1, float("nan"), 0])
        with self.assertRaises(ValueError):
            aggregate_metrics([])
        records = [
            {
                "feasible": True,
                "metrics": {
                    "nest_count": 2,
                    "total_flight_seconds": 10,
                    "mean_flight_seconds": 5,
                    "maximum_flight_seconds": 8,
                },
            },
            {
                "feasible": True,
                "metrics": {
                    "nest_count": 4,
                    "total_flight_seconds": 12,
                    "mean_flight_seconds": 3,
                    "maximum_flight_seconds": 4,
                },
            },
        ]
        self.assertEqual(
            aggregate_metrics(records),
            {
                "mean_nest_count": 3,
                "mean_total_flight_seconds": 11,
                "mean_region_flight_seconds": 4,
                "maximum_flight_seconds": 8,
            },
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with run_lock(root, "same"):
                with self.assertRaises(FileExistsError):
                    with run_lock(root, "same"):
                        pass
            self.assertFalse((root / ".same.lock").exists())


if __name__ == "__main__":
    unittest.main()
