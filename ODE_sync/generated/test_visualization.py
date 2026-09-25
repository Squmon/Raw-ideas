"""Regression tests for detail-plot selection by system dimension."""

from __future__ import annotations

import json
from pathlib import Path
import unittest

import numpy as np
import plotly.graph_objects as go

from show_results import ResultRecord, create_detail_figures


class DetailFigureTests(unittest.TestCase):
    def make_record(self, dimension: int, zero: bool = False) -> ResultRecord:
        times = np.linspace(0.0, 1.0, 4)
        values = np.zeros((4, dimension)) if zero else np.tile(
            np.linspace(-1.0, 1.0, dimension), (4, 1)
        )
        state_b = values * 0.5
        arrays = {
            "ts": times,
            "state_a": values,
            "state_b": state_b,
            "sync_indices": np.asarray([0, 0]),
            "coupling": np.asarray(1.0),
            "config_json": np.asarray(json.dumps({"coupling": 1.0})),
        }
        final_mse = float(np.mean(np.square(values[-1] - state_b[-1])))
        return ResultRecord(
            Path(f"solution_seed_1_dim_{dimension}.npz"),
            1,
            dimension,
            final_mse,
            arrays,
        )

    def test_dimension_fifteen_keeps_line_plot_layout(self) -> None:
        figures = create_detail_figures(self.make_record(15))
        self.assertEqual(len(figures), 5)
        self.assertFalse(
            any(isinstance(trace, go.Heatmap) for figure in figures for trace in figure.data)
        )
        self.assertEqual(len(figures[0].data), 15)
        self.assertEqual(len(figures[1].data), 15)
        self.assertEqual(len(figures[4].data), 15)
        self.assertTrue(all(figure.layout.legend is not None for figure in figures))

    def test_dimension_sixteen_uses_state_and_difference_heatmaps(self) -> None:
        figures = create_detail_figures(self.make_record(16))
        heatmaps = [trace for figure in figures for trace in figure.data if isinstance(trace, go.Heatmap)]
        scatters = [trace for figure in figures for trace in figure.data if isinstance(trace, go.Scatter)]
        self.assertEqual(len(figures), 5)
        self.assertEqual(len(heatmaps), 3)
        self.assertEqual(len(scatters), 3)
        self.assertTrue(all(np.asarray(trace.z).shape == (16, 4) for trace in heatmaps))
        self.assertTrue(all(trace.colorscale and len(trace.colorscale) > 1 for trace in heatmaps))

    def test_high_dimension_zero_state_has_finite_color_range(self) -> None:
        figures = create_detail_figures(self.make_record(16, zero=True))
        for figure in figures:
            for trace in figure.data:
                if isinstance(trace, go.Heatmap):
                    self.assertLess(trace.zmin, trace.zmax)


if __name__ == "__main__":
    unittest.main()