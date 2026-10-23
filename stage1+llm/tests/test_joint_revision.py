"""Revision acceptance: full coverage, same-K improvement and auditable wind."""

import copy
import dataclasses
import unittest

from stage1_llm.domain import DeploymentConfig, Turbine
from stage1_llm.joint_planner import CheckedPlanner
from stage1_llm.joint_smoke import MockDecoder
from stage1_llm.joint_tools import JointTools


class RevisionToolsTests(unittest.TestCase):
    def setUp(self):
        self.config = DeploymentConfig()
        self.points = [
            Turbine("a", 0, 0),
            Turbine("b", 1, 0),
            Turbine("c", 2, 0),
            Turbine("d", 2, 1),
        ]
        self.planner = CheckedPlanner(MockDecoder(), self.config)
        self.tools = JointTools(self.points, self.config, planner=self.planner)
        self.assign = {"a": "a", "b": "a", "c": "c", "d": "c"}

    def test_remove_merge_and_relocate_preserve_tasks(self):
        original = copy.deepcopy(self.assign)
        merged = self.tools.merge_nests(self.assign, "a", "c")
        self.assertEqual(merged["assignments"], dict.fromkeys(self.assign, "c"))
        self.assertEqual(merged["affected_nests"], ["a", "c"])
        moved = self.tools.relocate_nest(self.assign, "a", "d")
        self.assertEqual(moved["assignments"], {"a": "d", "b": "d", "c": "c", "d": "d"})
        self.assertEqual(moved["affected_nests"], ["a", "c", "d"])
        removed = self.tools.remove_nest(self.assign, "a", {"a": "c", "b": "c"})
        self.assertEqual(removed["assignments"], merged["assignments"])
        self.assertEqual(self.assign, original)
        self.assertEqual(self.planner.stats()["inferences"], 0)

    def test_incomplete_removal_is_contract_error(self):
        from stage1_llm.candidate_tools import ToolContractError

        with self.assertRaises(ToolContractError):
            self.tools.remove_nest(self.assign, "a", {"b": "c"})

    def test_reassignment_and_fixed_k_comparison(self):
        change = self.tools.reassign_tasks(self.assign, {"b": "c"})
        self.assertEqual(set(change["assignments"]), set(self.assign))
        self.assertEqual(len(set(change["assignments"].values())), 2)
        self.assertTrue(self.tools.better_metrics([2, 100, 80], [2, 110, 70]))
        self.assertFalse(self.tools.better_metrics([2, 110, 60], [2, 100, 80]))
        self.assertFalse(self.tools.better_metrics([3, 10, 5], [2, 100, 80]))

    def test_observations_survive_repair_and_ignore_repeats(self):
        p = self.tools.make_proposal(self.assign)
        self.tools.deployment(p)
        self.tools.deployment(p)
        self.tools.remember(p)
        first = self.tools.stats()["search_evidence"]
        self.assertEqual(first["distinct_feasible_deployments"], 1)
        self.assertEqual(first["strict_improvements"], 0)
        self.tools.repartition_repair(self.assign)
        self.assertEqual(
            self.tools.stats()["search_evidence"]["distinct_feasible_deployments"], 1
        )
        self.tools.finalize_return(p)
        self.assertEqual(
            self.tools.stats()["search_evidence"]["returned"]["objectives"],
            first["first_feasible"]["objectives"],
        )

    def test_wind_audit_and_cache_reset(self):
        row = self.planner.region(self.points, "a")
        audit = row["wind_audit"]
        self.assertAlmostEqual(audit["closed_flight_seconds"], row["flight_seconds"])
        self.assertAlmostEqual(sum(audit["leg_seconds"]), row["flight_seconds"])
        self.assertEqual(audit["return_leg_seconds"], audit["leg_seconds"][-1])
        self.assertAlmostEqual(audit["hover_equivalent_seconds"], 3 * 10 * 1.14)
        changed = CheckedPlanner(
            MockDecoder(), dataclasses.replace(self.config, wind_direction_degrees=0)
        )
        self.assertNotEqual(
            audit["environment_fingerprint"],
            changed.region(self.points, "a")["wind_audit"]["environment_fingerprint"],
        )
        self.planner.reset_cache()
        self.assertEqual(self.planner.stats()["inferences"], 0)
        self.assertFalse(self.planner.region(self.points, "a")["cache_hit"])

    def test_compass_direction_and_return_time(self):
        self.assertAlmostEqual(self.tools.directed_time("a", "b"), 1000 / 29)
        self.assertAlmostEqual(self.tools.directed_time("b", "a"), 1000 / 17)
        north = dataclasses.replace(self.config, wind_direction_degrees=0)
        points = [Turbine("a", 0, 0), Turbine("b", 0, 1)]
        tools = JointTools(points, north, planner=CheckedPlanner(MockDecoder(), north))
        self.assertAlmostEqual(tools.directed_time("a", "b"), 1000 / 29)
        self.assertAlmostEqual(tools.directed_time("b", "a"), 1000 / 17)

    def test_manual_candidate_uses_new_facade_inside_worker(self):
        from pathlib import Path

        from stage1_llm.sandbox import run_candidate

        source = (
            Path(__file__).resolve().parents[1] / "baselines/joint-v5/C0003_manual.py"
        )
        telemetry = {}
        answer = run_candidate(
            source,
            self.points,
            self.config,
            1234,
            10,
            planner=self.planner,
            telemetry=telemetry,
        )
        self.assertEqual(set(answer.assignments), set(self.assign))
        self.assertTrue(self.tools.deployment(dataclasses.asdict(answer))["feasible"])
        self.assertGreater(
            telemetry["structural_move_attempts"].get("relocate_nest", 0), 0
        )
        self.assertIsNotNone(telemetry["search_evidence"]["returned"])

    def test_cache_reset_rpc_keeps_loaded_decoder(self):
        from unittest.mock import Mock, patch

        from stage1_llm.joint_planner import PlannerService, _serve

        connection = Mock()
        connection.recv.side_effect = [{"command": "reset_cache"}, None]
        with patch(
            "stage1_llm.joint_planner.GreedyDecoder", return_value=MockDecoder()
        ) as decoder:
            _serve(connection, "unused.pt", "cpu", dataclasses.asdict(self.config), 12)
        self.assertEqual(decoder.call_count, 1)
        self.assertEqual(
            connection.send.call_args_list[-1].args[0]["stats"]["inferences"], 0
        )
        service = PlannerService.__new__(PlannerService)
        service.connection = Mock()
        service.connection.recv.return_value = {"ok": True, "stats": {"inferences": 0}}
        service._pending_started = None
        service.timeout = 1
        service.reset_cache()
        service.connection.send.assert_called_once_with({"command": "reset_cache"})
        self.assertEqual(service.stats()["inferences"], 0)

    def test_real_fixed_k_improvement_is_observed_and_saved(self):
        points = [
            Turbine("a", 0, 0),
            Turbine("b", 6, 0),
            Turbine("c", 6, 0),
            Turbine("d", 6.1, 0),
        ]
        tools = JointTools(
            points, self.config, planner=CheckedPlanner(MockDecoder(), self.config)
        )
        old = tools.make_proposal(self.assign)
        before = tools.deployment(old)
        tools.remember(old)
        change = tools.reassign_tasks(self.assign, {"b": "c"})
        new = tools.make_proposal(change["assignments"])
        after = tools.deployment(new)
        self.assertEqual(before["objectives"][0], after["objectives"][0])
        self.assertTrue(tools.better_metrics(after["objectives"], before["objectives"]))
        tools.remember(new)
        tools.deployment(new)
        tools.phase("fake_extra_improvement")
        stats = tools.stats()["search_evidence"]
        self.assertEqual(stats["strict_improvements"], 1)
        self.assertEqual(stats["fixed_nest_strict_improvements"], 1)
        self.assertEqual(stats["distinct_feasible_deployments"], 2)
        tools.repartition_repair(change["assignments"])
        self.assertEqual(tools.stats()["search_evidence"]["strict_improvements"], 1)

    def test_invalid_and_radius_failures_are_not_feasible_searches(self):
        malformed = {"nest_ids": ["c"], "assignments": {"c": "c"}}
        checked = self.tools.deployment(malformed)
        self.assertEqual(checked["diagnostics"][0]["missing_ids"], ["a", "b", "d"])
        self.tools.deployment(malformed)
        evidence = self.tools.stats()["search_evidence"]
        self.assertEqual(evidence["distinct_structural_deployments"], 0)
        self.assertEqual(evidence["invalid_structure_requests"], 2)
        far = [Turbine("a", 0, 0), Turbine("b", 8, 0)]
        tools = JointTools(
            far, self.config, planner=CheckedPlanner(MockDecoder(), self.config)
        )
        self.assertFalse(tools.region("a", ["a", "b"])["feasible"])
        self.assertEqual(tools.stats()["region_failure_counts"]["service_radius"], 1)

    def test_structural_edge_cases_do_not_invoke_planner(self):
        from stage1_llm.candidate_tools import ToolContractError

        for method, args in [
            ("reassign_tasks", (self.assign, {"a": "c"})),
            ("relocate_nest", (self.assign, "a", "c")),
            ("relocate_nest", (self.assign, "missing", "b")),
            ("merge_nests", (self.assign, "a", "missing")),
            ("remove_nest", (self.assign, "a", {"a": "a", "b": "a"})),
            ("reassign_tasks", ({"a": "a"}, {})),
        ]:
            with self.subTest(method=method, args=args), self.assertRaises(
                ToolContractError
            ):
                getattr(self.tools, method)(*args)
        singleton = {"a": "a"}
        from stage1_llm.joint_moves import transform

        with self.assertRaises(ValueError):
            transform(singleton, ["a"], "remove_nest", source="a", updates={"a": "a"})
        self.assertEqual(self.planner.stats()["inferences"], 0)

    def test_real_failure_sources_and_manual_runtime(self):
        from pathlib import Path

        from stage1_llm.candidate import validate_candidate_source

        root = Path(__file__).resolve().parents[1] / "baselines/joint-v5"
        with self.assertRaisesRegex(ValueError, "config.*line"):
            validate_candidate_source(
                (root / "original/C0001.py").read_text(encoding="utf-8")
            )
        validate_candidate_source(
            (root / "C0003_manual.py").read_text(encoding="utf-8")
        )
        # Reproduce the actual C0005 singleton-task deletion using the archived helper.
        scope = {}
        source = (root / "original/C0005.py").read_text(encoding="utf-8")
        source = source[source.index("def _members(") : source.index("def solve(")]
        from collections import defaultdict

        scope.update(defaultdict=defaultdict, _dist_xy=lambda a, b: abs(a[0] - b[0]))
        exec(source, scope)
        import random

        broken = scope["_move_one"](
            {"a": "a", "b": "b"}, {"a": (0, 0), "b": (1, 0)}, 7, random.Random(1)
        )
        self.assertNotEqual(set(broken), {"a", "b"})


class FocusedSearchTests(unittest.TestCase):
    def test_seed_reevaluation_six_rounds_and_completed_resume(self):
        import contextlib
        import io
        import json
        import tempfile
        from pathlib import Path

        from stage1_llm.client import DeepSeekSettings
        from stage1_llm.joint_search import JointSearch
        from stage1_llm.joint_settings import JointSettings
        from stage1_llm.joint_smoke import GROUP, MockClient
        from stage1_llm.settings import SearchSettings

        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(
            io.StringIO()
        ):
            root = Path(directory)
            input_path = root / "tiny.tsv"
            input_path.write_text("a\t0\t0\nb\t1\t0\nc\t1\t1\n", encoding="utf-8")
            source = root / "seed.py"
            source.write_text(GROUP, encoding="utf-8")
            settings = SearchSettings(
                input_path,
                (),
                DeploymentConfig(),
                data_dir=root / "data",
                training_virtual_fields=0,
                validation_virtual_fields=0,
                evaluation_budget=6,
                no_improvement_limit=6,
                show_code=False,
            )
            joint = JointSettings(
                settings, device="cpu", search_mode="focused_six", seed_candidate=source
            )
            planner = CheckedPlanner(MockDecoder(), settings.deployment)
            client = MockClient([GROUP] * 6)
            llm = DeepSeekSettings("https://example.invalid", "offline_mock", "")
            path = JointSearch(joint, llm, root / "out", planner, client).run("focused")
            state = json.loads((path / "search_state.json").read_text(encoding="utf-8"))
            self.assertEqual(client.calls, 6)
            self.assertEqual(len(state["history"]), 6)
            self.assertTrue(state["seed_candidate"]["evaluation"]["complete"])
            self.assertEqual(
                [r["direction_id"] for r in state["history"]],
                ["reduce_route_distance", "reduce_route_distance", "reduce_nests"] * 2,
            )
            self.assertTrue(all(r["parent_id"] == "S0000" for r in state["history"]))
            resumed = MockClient([])
            source.unlink()
            JointSearch(joint, llm, root / "out", planner, resumed).run(
                "focused", resume=True
            )
            self.assertEqual(resumed.calls, 0)
            from unittest.mock import Mock, patch

            from stage1_llm.joint_cli import main

            standard = dataclasses.replace(
                joint, search_mode="standard", seed_candidate=None
            )
            planner.close = Mock()
            with patch(
                "stage1_llm.joint_cli.load_joint_settings", return_value=standard
            ), patch(
                "stage1_llm.joint_cli.load_deepseek_settings", return_value=llm
            ), patch(
                "stage1_llm.joint_cli.PlannerService", return_value=planner
            ), patch(
                "stage1_llm.joint_cli.JointSearch",
                side_effect=lambda j, l, o, p: JointSearch(j, l, o, p, MockClient([])),
            ):
                self.assertEqual(
                    main(
                        [
                            "resume",
                            "--resume-run-id",
                            "focused",
                            "--outputs",
                            str(root / "out"),
                        ]
                    ),
                    0,
                )
            source.write_text(GROUP, encoding="utf-8")
            failed = MockClient([ValueError("mock provider failure")] * 6)
            failed_path = JointSearch(joint, llm, root / "out", planner, failed).run(
                "failed"
            )
            failed_state = json.loads(
                (failed_path / "search_state.json").read_text(encoding="utf-8")
            )
            self.assertEqual(failed.calls, 6)
            self.assertEqual(len(failed_state["history"]), 6)
            self.assertTrue(
                all(
                    r["evaluation"]["status"] == "generation_error"
                    for r in failed_state["history"]
                )
            )
            # A bad seed must fail before the first provider call.
            source.write_text(
                "def solve(turbines, config, seed):\n return unknown_global\n",
                encoding="utf-8",
            )
            unused = MockClient([])
            with self.assertRaises(ValueError):
                JointSearch(joint, llm, root / "out", planner, unused).run("bad-seed")
            self.assertEqual(unused.calls, 0)

    def test_provenance_and_null_metrics_feedback(self):
        from pathlib import Path
        from unittest.mock import Mock

        from stage1_llm.client import DeepSeekSettings
        from stage1_llm.joint_provenance import source_provenance
        from stage1_llm.joint_search import JointSearch
        from stage1_llm.joint_settings import JointSettings
        from stage1_llm.joint_smoke import MockClient
        from stage1_llm.settings import SearchSettings

        root = Path(__file__).resolve().parents[1]
        self.assertEqual(
            source_provenance(root / "baselines/joint-v5/original/C0003.py")["origin"],
            "original_llm",
        )
        self.assertEqual(
            source_provenance(root / "baselines/joint-v5/C0003_manual.py")["origin"],
            "manual_revision",
        )
        settings = SearchSettings(
            root / "baselines/joint-v5/original/C0003.py", (), DeploymentConfig()
        )
        search = JointSearch(
            JointSettings(settings, device="cpu"),
            DeepSeekSettings("https://example.invalid", "offline_mock", ""),
            root / "outputs",
            CheckedPlanner(MockDecoder(), settings.deployment),
            MockClient([]),
        )
        search.bundle = {"datasets": []}
        parent = {
            "candidate_id": "bad",
            "source_path": "original/C0003.py",
            "evaluation": {
                "status": "network_deployment_infeasible",
                "datasets": [
                    {
                        "name": "tiny",
                        "split": "training",
                        "feasible": False,
                        "metrics": None,
                        "diagnostics": [{"kind": "coverage_or_self_assignment"}],
                    }
                ],
            },
        }
        prompt = search._prompt(
            parent, "constraint_repair", Mock(root=root / "baselines/joint-v5")
        )
        self.assertIn("coverage_or_self_assignment", prompt)


if __name__ == "__main__":
    unittest.main()
