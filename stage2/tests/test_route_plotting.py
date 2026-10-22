"""Remaining-energy colour uses the original flight-plus-hover endurance proxy."""
import importlib
import unittest

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.patches import Circle
import numpy as np

from stage2.atsp import WindATSPConfig, WindATSPInstance
from stage2.evaluation import _draw_route, evaluate_route


class RoutePlottingTests(unittest.TestCase):
    def setUp(self):
        self.instance = WindATSPInstance(("D", "A", "B"), np.array([[0., 0.], [1., 0.], [1., 1.]]),
                                         0, WindATSPConfig())
        self.route = (0, 1, 2, 0)

    def tearDown(self):
        plt.close("all")

    def test_segments_use_remaining_energy_with_asymmetric_flight_and_hover(self):
        from stage2.route_plotting import route_energy_segments
        segments, remaining = route_energy_segments(self.instance, self.route, subdivisions=2)
        times = self.instance.travel_time_seconds
        expected = []
        elapsed = 0.
        for a, b in zip(self.route, self.route[1:]):
            expected.extend([100 * (1 - (elapsed + times[a, b] * fraction) / self.instance.config.max_endurance_seconds)
                             for fraction in (.25, .75)])
            elapsed += times[a, b]
            if b != self.instance.depot_index:
                elapsed += self.instance.config.hover_equivalent_seconds
        np.testing.assert_allclose(remaining, expected)
        self.assertAlmostEqual(elapsed, self.instance.evaluate_route(self.route).equivalent_energy_seconds)
        self.assertEqual(segments.shape, (6, 2, 2))
        np.testing.assert_allclose(segments[-1, -1], self.instance.coordinates_km[0])
        self.assertNotEqual(times[0, 1], times[1, 0])

    def test_model_plot_has_no_radius_circle_and_has_energy_colourbar(self):
        fig, ax = plt.subplots()
        report = evaluate_route(self.instance, "model_best_of_128", self.route)
        _draw_route(ax, self.instance, report, 7.)
        self.assertFalse(any(isinstance(p, Circle) for p in ax.patches))
        lines = [c for c in ax.collections if isinstance(c, LineCollection)]
        self.assertEqual(len(lines), 1)
        self.assertEqual(lines[0].norm.vmax, 100)
        self.assertEqual(fig.axes[-1].get_title(), "SoC")
        self.assertEqual(lines[0].cmap.name, "viridis_r")
        early, late = lines[0].cmap(1.), lines[0].cmap(0.)
        self.assertGreater(early[2], early[0])
        self.assertGreater(late[0], late[2])

    def test_bnb_uses_same_gradient_and_keeps_unproven_status(self):
        plots = importlib.import_module("stage2.Branch and Bound.plots")
        fig, ax = plt.subplots()
        plots._draw(ax, 1, self.instance, {"route": self.route, "proven_optimal": False,
                    "termination": "TIME_LIMIT", "flight_seconds": self.instance.evaluate_route(self.route).flight_seconds})
        self.assertTrue(any(isinstance(c, LineCollection) for c in ax.collections))
        self.assertIn("TIME_LIMIT", ax.get_title())
        self.assertIn("not proven", " ".join(ax.get_legend_handles_labels()[1]))
        self.assertFalse(any(isinstance(p, Circle) for p in ax.patches))

    def test_zero_flight_and_no_route_do_not_invent_elapsed_time(self):
        from stage2.route_plotting import route_energy_segments
        zero = WindATSPInstance(("D",), np.zeros((1, 2)), 0, WindATSPConfig())
        segments, values = route_energy_segments(zero, (0, 0))
        self.assertEqual(len(segments), 0)
        self.assertEqual(len(values), 0)
        fig, ax = plt.subplots()
        _draw_route(ax, zero, evaluate_route(zero, "model_greedy", (0, 0)), 7.)
        self.assertTrue(np.isfinite(ax.get_xlim()).all())
        plots = importlib.import_module("stage2.Branch and Bound.plots")
        fig, ax = plt.subplots()
        plots._draw(ax, 1, self.instance, {"route": None, "termination": "TIME_LIMIT"})
        self.assertEqual(len(fig.axes), 1)
        self.assertFalse(any(isinstance(c, LineCollection) for c in ax.collections))

    def test_unsafe_energy_is_not_silently_reported_as_safe(self):
        from stage2.route_plotting import route_energy_segments
        unsafe = WindATSPInstance(self.instance.identifiers, self.instance.coordinates_km, 0,
                                 WindATSPConfig(max_endurance_seconds=1.))
        _, values = route_energy_segments(unsafe, self.route)
        self.assertLess(values.min(), 0.)
        fig, ax = plt.subplots()
        _draw_route(ax, unsafe, evaluate_route(unsafe, "model_greedy", self.route), 7.)
        self.assertIn("INFEASIBLE", ax.get_xlabel())
        with self.assertRaises(ValueError):
            route_energy_segments(unsafe, self.route, subdivisions=0)

    def test_zero_length_leg_still_charges_inspection_hover(self):
        from stage2.route_plotting import route_energy_segments
        instance = WindATSPInstance(("D", "A", "B"), np.array([[0., 0.], [0., 0.], [1., 0.]]),
                                   0, WindATSPConfig())
        _, values = route_energy_segments(instance, (0, 1, 2, 0), subdivisions=1)
        expected_first = 100 * (1 - (instance.config.hover_equivalent_seconds +
                                    .5 * instance.travel_time_seconds[1, 2]) / instance.config.max_endurance_seconds)
        self.assertAlmostEqual(values[0], expected_first)

    def test_nonzero_depot_and_fixed_scale_do_not_reset_battery_on_return(self):
        from stage2.route_plotting import route_energy_segments
        instance = WindATSPInstance(self.instance.identifiers, self.instance.coordinates_km, 1, WindATSPConfig())
        route = (1, 2, 0, 1)
        segments, values = route_energy_segments(instance, route, subdivisions=1)
        self.assertEqual(len(segments), 3)
        self.assertGreater(values[-1], 50.)
        np.testing.assert_allclose(segments[-1, -1], instance.coordinates_km[1])
        fig, ax = plt.subplots()
        _draw_route(ax, instance, evaluate_route(instance, "model_greedy", route), 99.)
        collection = next(c for c in ax.collections if isinstance(c, LineCollection))
        self.assertEqual(collection.norm.vmax, 100)


if __name__ == "__main__":
    unittest.main()
