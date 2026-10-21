"""Independent enumeration checks for the directed branch-and-bound solver."""

from __future__ import annotations

import importlib
from itertools import permutations
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from stage2.benchmark import load_references
from stage2.tests.test_pool import small_instance


solver = importlib.import_module("stage2.Branch and Bound.solver")


def enumerate_cost(costs, forced=(), forbidden=()):
    """Enumerate independently; the oracle never calls solver helpers."""
    n = len(costs)
    best = float("inf")
    for order in permutations(range(1, n)):
        route = (0, *order, 0)
        edges = set(zip(route, route[1:]))
        if any((i, j) not in edges for i, j in enumerate(forced) if j >= 0):
            continue
        if edges.intersection(forbidden):
            continue
        best = min(best, sum(costs[i, j] for i, j in zip(route, route[1:])))
    return best


class SolverTests(unittest.TestCase):
    def test_small_directed_instances_match_enumeration(self):
        for n in range(2, 9):
            for seed in range(3):
                with self.subTest(n=n, seed=seed):
                    costs = np.random.default_rng(seed + n * 10).uniform(0.1, 30, (n, n))
                    np.fill_diagonal(costs, 0)
                    expected = enumerate_cost(costs)
                    result = solver.BranchAndBound(costs, 10000, progress=None).solve()
                    self.assertTrue(result["proven_optimal"])
                    self.assertAlmostEqual(result["flight_seconds"], expected, places=7)
                    self.assertEqual(set(result["route"][:-1]), set(range(n)))
                    self.assertLessEqual(result["objective_bound_seconds"], expected + 1e-8)

    def test_relaxation_is_a_lower_bound_for_forced_and_excluded_edges(self):
        costs = np.random.default_rng(61).uniform(1, 50, (6, 6))
        cases = [((-1,) * 6, ()), ((1, -1, -1, -1, -1, -1), ((2, 3),)),
                 ((1, 2, -1, -1, -1, -1), ((3, 4),))]
        engine = solver.BranchAndBound(costs, 10000, progress=None)
        for forced, forbidden in cases:
            node = engine.relax(solver.SearchNode(forced, forbidden))
            self.assertIsNotNone(node)
            self.assertLessEqual(node.bound, enumerate_cost(costs, forced, forbidden) + 1e-8)

    def test_subtours_and_incoming_conflicts_are_rejected(self):
        engine = solver.BranchAndBound(np.ones((4, 4)), 100, progress=None)
        self.assertIsNone(engine.relax(solver.SearchNode((1, 0, -1, -1))))
        self.assertIsNone(engine.relax(solver.SearchNode((2, 2, -1, -1))))

    def test_equal_costs_zero_costs_and_smallest_cases(self):
        for n in (1, 2, 5):
            for cost in (0.0, 1.0):
                matrix = np.full((n, n), cost)
                np.fill_diagonal(matrix, 0)
                result = solver.BranchAndBound(matrix, 100, progress=None).solve()
                self.assertTrue(result["proven_optimal"])
                self.assertEqual(result["flight_seconds"], 0 if n == 1 else n * cost)
        self.assertEqual(solver.BranchAndBound(np.array([[99.0]]), 0, progress=None)
                         .solve()["flight_seconds"], 0)

    def test_include_exclude_children_partition_all_parent_tours(self):
        costs = np.full((6, 6), 100.0)
        for i, j in ((0, 1), (1, 0), (2, 3), (3, 2), (4, 5), (5, 4)):
            costs[i, j] = 1
        engine = solver.BranchAndBound(costs, 10000, progress=None)
        children = engine._expand(solver.SearchNode((-1,) * 6))
        self.assertEqual(len(children), 2)
        for order in permutations(range(1, 6)):
            route = (0, *order, 0)
            edges = set(zip(route, route[1:]))
            memberships = [not edges.intersection(child.forbidden)
                           and all(j < 0 or (i, j) in edges for i, j in enumerate(child.forced))
                           for child in children]
            self.assertEqual(sum(memberships), 1)

    def test_endurance_boundary_is_not_relaxed(self):
        costs = np.ones((4, 4))
        self.assertTrue(solver.BranchAndBound(costs, 4, progress=None).solve()["feasible"])
        result = solver.BranchAndBound(costs, 4 - 1e-10, progress=None).solve()
        self.assertEqual(result["termination"], "INFEASIBLE")
        self.assertFalse(result["proven_optimal"])
        self.assertIsNone(result["route"])

    def test_failed_initial_heuristic_does_not_imply_infeasible(self):
        costs = np.random.default_rng(5).uniform(1, 20, (7, 7))
        expected = enumerate_cost(costs)
        with patch.object(solver.BranchAndBound, "_initial_routes", return_value=None):
            result = solver.BranchAndBound(costs, expected + 0.01, progress=None).solve()
        self.assertTrue(result["proven_optimal"])
        self.assertAlmostEqual(result["flight_seconds"], expected)

    def test_impossible_assignment_and_bad_input(self):
        costs = np.full((3, 3), np.inf)
        result = solver.BranchAndBound(costs, 100, progress=None).solve()
        self.assertEqual(result["termination"], "INFEASIBLE")
        for matrix in (np.zeros((0, 0)), np.zeros((2, 3)), np.array([[0, -1], [1, 0]]),
                       np.array([[0, np.nan], [1, 0]])):
            with self.assertRaises(ValueError):
                solver.BranchAndBound(matrix, 100)

    def test_defaults_and_option_validation(self):
        self.assertEqual(solver.Options().time_limit, 0)
        self.assertEqual(solver.Options().log_interval, 10)
        self.assertEqual(solver.Options().checkpoint_interval, 60)
        for kwargs in ({"time_limit": -1}, {"log_interval": 0}, {"tolerance": float("nan")}):
            with self.assertRaises(ValueError):
                solver.Options(**kwargs)

    def test_interrupt_in_expansion_preserves_parent_and_resumes(self):
        costs = np.random.default_rng(91).uniform(1, 50, (7, 7))
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "state.json"
            engine = solver.BranchAndBound(costs, 10000, checkpoint_path=checkpoint, progress=None)
            with patch.object(engine, "_expand", side_effect=KeyboardInterrupt):
                interrupted = engine.solve()
            self.assertEqual(interrupted["termination"], "INTERRUPTED")
            self.assertFalse(interrupted["proven_optimal"])
            self.assertGreater(len(json.loads(checkpoint.read_text())["pending"]), 0)
            resumed = solver.BranchAndBound(costs, 10000, checkpoint_path=checkpoint,
                                            resume=True, progress=None).solve()
            self.assertAlmostEqual(resumed["flight_seconds"], enumerate_cost(costs))
            self.assertGreaterEqual(resumed["runtime_seconds"], interrupted["runtime_seconds"])
            with self.assertRaises(ValueError):
                solver.BranchAndBound(costs + 1, 10000, checkpoint_path=checkpoint, resume=True)
            with self.assertRaises(FileExistsError):
                solver.BranchAndBound(costs, 10000, checkpoint_path=checkpoint)

    def test_time_limit_and_periodic_log(self):
        ticks = iter(index * 0.01 for index in range(10000))
        messages = []
        options = solver.Options(time_limit=0.09, log_interval=0.01, checkpoint_interval=0.01)
        with tempfile.TemporaryDirectory() as directory:
            result = solver.BranchAndBound(np.ones((8, 8)), 100, options=options,
                                           clock=lambda: next(ticks), progress=messages.append,
                                           checkpoint_path=Path(directory) / "state.json").solve()
        self.assertEqual(result["termination"], "TIME_LIMIT")
        self.assertFalse(result["proven_optimal"])
        self.assertTrue(any("lower_bound=" in line for line in messages))

    def test_memory_limit_and_completed_checkpoint_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "state.json"
            engine = solver.BranchAndBound(np.ones((4, 4)), 100, checkpoint_path=checkpoint, progress=None)
            with patch.object(engine, "_expand", side_effect=MemoryError):
                result = engine.solve()
            self.assertEqual(result["termination"], "MEMORY_LIMIT")
            self.assertFalse(result["proven_optimal"])
            final = solver.BranchAndBound(np.ones((4, 4)), 100, checkpoint_path=checkpoint,
                                          resume=True, progress=None).solve()
            cached = solver.BranchAndBound(np.ones((4, 4)), 100, checkpoint_path=checkpoint,
                                           resume=True, progress=None).solve()
            self.assertEqual(final, cached)
            with self.assertRaises(FileNotFoundError):
                solver.BranchAndBound(np.ones((4, 4)), 100, checkpoint_path=Path(directory) / "absent", resume=True)

    def test_invalid_forced_edge_and_forbidden_assignment(self):
        engine = solver.BranchAndBound(np.ones((3, 3)), 100, progress=None)
        self.assertIsNone(engine.relax(solver.SearchNode((5, -1, -1))))
        self.assertIsNone(engine.relax(solver.SearchNode((1, -1, -1), ((0, 1),))))

    def test_nonzero_depot_and_reference_loader_compatibility(self):
        from dataclasses import replace

        instance = replace(small_instance("D", 0, nodes=5), depot_index=2)
        result = solver.solve_instance(instance, progress=None)
        self.assertEqual(result["route"][0], 2)
        self.assertEqual(result["route"][-1], 2)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "reference_results.json"
            path.write_text(json.dumps({"schema_version": 1, "results": [result]}))
            records = load_references(path, [instance])
        self.assertTrue(next(iter(records.values()))["proven_optimal"])


if __name__ == "__main__":
    unittest.main()
