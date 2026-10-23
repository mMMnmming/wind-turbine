"""Plot-only fixtures: no algorithm execution, Stage2 loading or API requests."""
from __future__ import annotations

import contextlib
import io
import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch
from uuid import uuid4

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from windsearch.artifacts import atomic_json
from windsearch.plotting import main, paper_figure


def workspace():
    root = Path(__file__).resolve().parents[1]/"outputs/plot-checks"/uuid4().hex
    root.mkdir(parents=True)
    return root


def fixture(root, count, instance, candidate="external", source="def example():\n    return 1\n"):
    ids = [f"t{i:03d}" for i in range(count)]
    nests = [ids[0], ids[count//2]]
    assignments = {t: nests[int(i >= count//2)] for i, t in enumerate(ids)}
    points = [{"identifier": t, "x_km": (i%12)*.4, "y_km": (i//12)*.4} for i, t in enumerate(ids)]
    regions = []
    for nest in nests:
        members = [t for t in ids if assignments[t] == nest]
        regions.append({"nest_id": nest, "route_ids": [nest]+[t for t in members if t != nest]+[nest],
                        "feasible": True})
    metrics = {"nest_count": 2, "total_flight_seconds": 120., "maximum_flight_seconds": 70.,
               "minimum_endurance_margin_seconds": 2000.}
    problem = {"turbines": points, "config": {"communication_radius_km": 7},
               "environment": {"wind_speed_mps": 6, "wind_direction_degrees": 90}}
    result = {"name": instance, "split": "final_test" if count in (87, 189) else "training",
              "code_success": True, "stop_reason": "stagnation_limit",
              "solution": {"assignments": assignments, "nest_ids": nests},
              "final_audit": {"feasible": True, "regions": regions, "metrics": metrics}}
    atomic_json(root/f"diagnostics/runtime/{candidate}/{instance}-problem.json", problem)
    atomic_json(root/f"diagnostics/evaluations/{candidate}/{instance}.json", result)
    path = root/f"candidates/{candidate}.py"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(source.encode("utf-8"))
    return problem, result


class PlotStyleSources(unittest.TestCase):
    def test_paper_parameters_match_stage1_and_no_extra_labels(self):
        import matplotlib
        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
        from matplotlib.colors import to_rgba
        fig, ax = paper_figure({"a": (0, 0), "b": (2, 3)}, ["a"], [], 7)
        try:
            self.assertEqual(tuple(fig.get_size_inches()), (6., 7.5))
            self.assertEqual(ax.collections[0].get_sizes().tolist(), [30])
            self.assertEqual(ax.collections[1].get_sizes().tolist(), [30])
            self.assertEqual(tuple(ax.collections[0].get_facecolors()[0]), to_rgba("darkblue"))
            self.assertEqual(tuple(ax.collections[1].get_facecolors()[0]), to_rgba("red"))
            self.assertEqual(ax.collections[1].get_linewidths().tolist(), [1.5])
            self.assertEqual(ax.patches[0].radius, 7)
            self.assertEqual(ax.patches[0].get_alpha(), .5)
            self.assertFalse(ax.patches[0].get_fill())
            self.assertEqual(ax.xaxis.label.get_fontsize(), 18)
            self.assertEqual(ax.get_legend().get_texts()[0].get_fontsize(), 16)
            self.assertEqual(ax.get_title(), "")
            self.assertEqual(len(ax.texts), 0)
            self.assertEqual(ax.get_xgridlines()[0].get_linestyle(), "--")
        finally:
            plt.close(fig)

    def test_standalone_87_and_189_paper_and_diagnostic(self):
        root = workspace()
        with contextlib.redirect_stdout(io.StringIO()):
            for count in (87, 189):
                run = root/str(count)
                name = f"turbine{count}"
                fixture(run, count, name)
                self.assertEqual(main(["--run-dir", str(run), "--candidate", "external", "--instance", name]), 0)
                output = run/"plots/external/paper"
                summary = json.loads((output/"plot_summary.json").read_text())
                self.assertEqual(summary["results"][0]["assignment_count"], count)
                self.assertTrue(summary["results"][0]["show_radius"])
                self.assertFalse(summary["results"][0]["show_routes"])
                self.assertTrue((output/(name+".png")).is_file())
                self.assertTrue((output/(name+".pdf")).is_file())
            self.assertEqual(main(["--run-dir", str(root/"87"), "--candidate", "external",
                                   "--instance", "turbine87", "--style", "diagnostic",
                                   "--no-show-radius", "--no-show-routes"]), 0)

    def test_batch_19_sources_and_source_mismatch_or_missing(self):
        root = workspace()
        search, run87, run189 = root/"search", root/"87", root/"189"
        names = ["real102"]+[f"training-kde-{i:03d}" for i in range(8)]+[f"validation-kde-{i:03d}" for i in range(8)]
        for name in names:
            fixture(search, 102, name, "C0001")
        atomic_json(search/"summary.json", {"best_candidate": "C0001"})
        atomic_json(search/"diagnostics/runtime/identity.json", {"settings": {
            "training_virtual_fields": 8, "validation_virtual_fields": 8}})
        fixture(run87, 87, "turbine87", source="def example():\r\n    return 1\r\n")
        fixture(run189, 189, "turbine189")
        args = ["--run-dir", str(search), "--all-instances", "--extra-run-dir", str(run87),
                "--extra-run-dir", str(run189)]
        # Check source scheduling without redundantly rendering all 19 PNG/PDF pairs.
        with contextlib.redirect_stdout(io.StringIO()), patch("windsearch.plotting.render", return_value={"status": "plotted"}) as render:
            self.assertEqual(main(args), 0)
            self.assertEqual(render.call_count, 19)
            summary_path = search/"plots/C0001/paper/plot_summary.json"
            summary = json.loads(summary_path.read_text())
            self.assertEqual((summary["requested"], summary["plotted"]), (19, 19))
            self.assertEqual(sum(r["source_group"] == "extra_independent_evaluation" for r in summary["results"]), 2)
            (run189/"candidates/external.py").write_text("different source", encoding="utf-8")
            self.assertEqual(main(args), 1)
            summary = json.loads(summary_path.read_text())
            self.assertEqual((summary["plotted"], summary["errors"]), (18, 1))
            self.assertIn("differs", summary["results"][-1]["error"])
            self.assertEqual(main(args+["--extra-run-dir", str(root/"missing")]), 1)
            summary = json.loads(summary_path.read_text())
            self.assertEqual(summary["missing"], 1)

    def test_extra_count_mismatch_and_routes_switch(self):
        root = workspace()
        search, extra = root/"search", root/"extra"
        fixture(search, 102, "real102", "C0001")
        atomic_json(search/"summary.json", {"best_candidate": "C0001"})
        fixture(extra, 86, "turbine87")
        with contextlib.redirect_stdout(io.StringIO()), patch("windsearch.plotting.render", return_value={"status": "plotted"}) as render:
            self.assertEqual(main(["--run-dir", str(search), "--extra-run-dir", str(extra),
                                   "--show-routes", "--no-show-radius"]), 1)
            self.assertEqual(render.call_count, 1)
            self.assertEqual(render.call_args.args[-3:], (False, "paper", True))
            summary = json.loads((search/"plots/C0001/paper/plot_summary.json").read_text())
            self.assertIn("expected 87 turbines, found 86", summary["results"][-1]["error"])


if __name__ == "__main__":
    unittest.main()
