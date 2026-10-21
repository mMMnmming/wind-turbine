"""Comparison formulas, identity matching and honest missing-data reporting."""

from __future__ import annotations

import copy
import importlib
from pathlib import Path
import tempfile
import unittest

from stage2.data import instance_fingerprint, instance_to_record
from stage2.tests.test_pool import small_instance


comparison = importlib.import_module("stage2.Branch and Bound.compare")
solver = importlib.import_module("stage2.Branch and Bound.solver")


def payloads():
    instances = [small_instance("D", 0, nodes=4), small_instance("E", 10, nodes=5)]
    references, models = [], []
    for region, instance in enumerate(instances, 1):
        reference = solver.solve_instance(instance, progress=None)
        reference["region"] = region
        references.append(reference)
        models.append({"region": region, "nest_id": instance.identifiers[0],
                       "node_count": instance.node_count, "instance": instance_to_record(instance),
                       "instance_fingerprint": instance_fingerprint(instance), "route": reference["route"],
                       "flight_seconds": reference["flight_seconds"], "runtime_seconds": 0.01,
                       "feasible": True})
    return {"schema_version": 1, "device": "cpu", "results": models}, {
        "schema_version": 1, "results": references}


class ComparisonTests(unittest.TestCase):
    def test_five_percent_and_zero_reference(self):
        self.assertAlmostEqual(comparison.flight_gap_percent(105, 100), 5)
        self.assertIsNone(comparison.flight_gap_percent(0, 0))
        with self.assertRaises(ValueError):
            comparison.flight_gap_percent(90, 100)

    def test_matching_uses_fingerprint_not_list_order(self):
        model, references = payloads()
        references["results"].reverse()
        result = comparison.compare_results(model, references)
        self.assertEqual(result["summary"]["valid_gap_count"], 2)
        self.assertAlmostEqual(result["summary"]["flight_gap_percent"], 0)
        self.assertEqual(result["rows"][0]["region"], 1)

    def test_duplicate_missing_and_mismatched_environment(self):
        model, references = payloads()
        duplicate = copy.deepcopy(model)
        duplicate["results"].append(duplicate["results"][0])
        with self.assertRaises(ValueError):
            comparison.compare_results(duplicate, references)
        references["results"].pop()
        result = comparison.compare_results(model, references)
        self.assertTrue(result["warnings"])
        self.assertEqual(result["summary"]["coverage"], "partial")
        bad = copy.deepcopy(model)
        bad["results"][0]["instance"]["config"]["wind_speed_mps"] = 5
        with self.assertRaises(ValueError):
            comparison.compare_results(bad, references)

    def test_unproven_and_model_infeasible_have_no_optimality_gap(self):
        model, references = payloads()
        references["results"][0].update(proven_optimal=False, status=9, termination="TIME_LIMIT")
        # A deliberately low endurance produces a genuinely infeasible model route.
        from dataclasses import replace
        instance = small_instance("F", 20, nodes=4)
        instance = replace(instance, config=replace(instance.config, max_endurance_seconds=1))
        measured = instance.evaluate_route((0, 1, 2, 3, 0))
        record = model["results"][1]
        record.update(instance=instance_to_record(instance), instance_fingerprint=instance_fingerprint(instance),
                      node_count=4, route=list(measured.route), flight_seconds=measured.flight_seconds,
                      feasible=False)
        references["results"][1] = {**solver.solve_instance(instance, progress=None), "region": 2}
        result = comparison.compare_results(model, references)
        self.assertEqual(result["summary"]["valid_gap_count"], 0)
        self.assertTrue(all(row["flight_gap_percent"] is None for row in result["rows"]))

    def test_missing_runtime_and_false_optimality_are_rejected(self):
        model, references = payloads()
        del model["results"][0]["runtime_seconds"]
        with self.assertRaises(ValueError):
            comparison.compare_results(model, references)
        model, references = payloads()
        references["results"][0]["objective_bound_seconds"] = 0
        with self.assertRaises(ValueError):
            comparison.compare_results(model, references)

    def test_raw_totals_and_effective_gap_have_distinct_scopes(self):
        model, references = payloads()
        references["results"][0].update(proven_optimal=False, status=9, termination="TIME_LIMIT")
        report = comparison.compare_results(model, references)
        summary = report["summary"]
        self.assertEqual(summary["valid_gap_count"], 1)
        self.assertAlmostEqual(summary["model_all_regions"]["total_flight_seconds"],
                               sum(row["flight_seconds"] for row in model["results"]))
        self.assertAlmostEqual(summary["bnb_all_regions"]["total_runtime_seconds"],
                               sum(row["runtime_seconds"] for row in references["results"]))
        self.assertEqual(summary["bnb_all_regions"]["unproven_count"], 1)
        self.assertNotEqual(summary["model_flight_seconds"], summary["model_all_regions"]["total_flight_seconds"])
        with tempfile.TemporaryDirectory() as directory:
            table = comparison.write_comparison(report, Path(directory))
            self.assertIn("全部区域原始合计", table)
            self.assertIn("有效对比区域合计", table)
            import csv
            with (Path(directory) / "comparison.csv").open(encoding="utf-8-sig", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual([row["row_type"] for row in rows[-2:]], ["all_regions_total", "valid_regions_total"])

    def test_weighted_aggregate_and_formats(self):
        rows = [{"model_flight_seconds": 105.0, "bnb_flight_seconds": 100.0,
                 "model_runtime_seconds": 0.01, "bnb_runtime_seconds": 2.0, "flight_gap_percent": 5.0},
                {"model_flight_seconds": 1200.0, "bnb_flight_seconds": 1000.0,
                 "model_runtime_seconds": 0.02, "bnb_runtime_seconds": 3.0, "flight_gap_percent": 20.0}]
        summary = comparison.summarize_rows(rows)
        self.assertAlmostEqual(summary["flight_gap_percent"], 205 / 1100 * 100)
        self.assertAlmostEqual(summary["model_runtime_seconds"], 0.03)
        model, references = payloads()
        result = comparison.compare_results(model, references)
        with tempfile.TemporaryDirectory() as directory:
            table = comparison.write_comparison(result, Path(directory))
            self.assertIn("gap", table)
            for suffix in ("csv", "md", "json"):
                self.assertTrue((Path(directory) / f"comparison.{suffix}").is_file())
            with self.assertRaises(FileExistsError):
                comparison.write_comparison(result, Path(directory))


if __name__ == "__main__":
    unittest.main()
