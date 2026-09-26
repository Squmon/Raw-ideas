"""Regression tests for detail-plot selection by system dimension."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import plotly.graph_objects as go

from show_results import (
    MAX_ANIMATION_FRAMES,
    ResultRecord,
    create_animated_graph_figure,
    create_detail_figures,
    generate_batch_table,
)


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
            "matrix_a": np.zeros((dimension, dimension)),
            "matrix_b": np.zeros((dimension, dimension)),
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
        self.assertEqual(len(figures), 11)
        self.assertFalse(
            any(isinstance(trace, go.Heatmap) for figure in figures for trace in figure.data)
        )
        self.assertEqual(len(figures[6].data), 15)
        self.assertEqual(len(figures[7].data), 15)
        self.assertEqual(len(figures[10].data), 15)
        self.assertTrue(all(figure.layout.title.text for figure in figures))
        self.assertTrue(all(figure.layout.template.layout.plot_bgcolor for figure in figures))
        self.assertTrue(all(figure.layout.template.layout.paper_bgcolor for figure in figures))

    def test_dimension_sixteen_uses_state_and_difference_heatmaps(self) -> None:
        figures = create_detail_figures(self.make_record(16))
        heatmaps = [trace for figure in figures for trace in figure.data if isinstance(trace, go.Heatmap)]
        self.assertEqual(len(figures), 11)
        self.assertEqual(len(heatmaps), 3)
        self.assertTrue(all(np.asarray(trace.z).shape == (16, 4) for trace in heatmaps))
        self.assertTrue(all(trace.colorscale and len(trace.colorscale) > 1 for trace in heatmaps))
        self.assertEqual(len(figures[0].data[1].x), 16)
        self.assertEqual(len(figures[1].data[1].x), 16)

    def test_animated_graphs_cover_all_metrics_and_time_frames(self) -> None:
        record = self.make_record(4)
        figures = create_detail_figures(record)
        animated = figures[2:6]
        self.assertEqual(len(animated), 4)
        self.assertTrue(all(len(figure.frames) == len(record.arrays["ts"]) for figure in animated))
        self.assertTrue(all(frame.traces == (1,) for figure in animated for frame in figure.frames))
        self.assertTrue(all(len(figure.layout.sliders[0].steps) == len(record.arrays["ts"]) for figure in animated))
        self.assertTrue(all(figure.layout.template.layout.paper_bgcolor != "white" for figure in figures))
        np.testing.assert_array_equal(animated[0].frames[-1].data[0].marker.color, record.arrays["state_a"][-1])
        np.testing.assert_array_equal(animated[1].frames[-1].data[0].marker.color, record.arrays["state_b"][-1])
        np.testing.assert_array_equal(
            animated[2].frames[-1].data[0].marker.color,
            record.arrays["state_a"][-1] - record.arrays["state_b"][-1],
        )
        np.testing.assert_array_equal(
            animated[3].frames[-1].data[0].marker.color,
            np.abs(record.arrays["state_a"][-1] - record.arrays["state_b"][-1]),
        )

    def test_animation_caps_frames_and_keeps_time_endpoints(self) -> None:
        times = np.linspace(0.0, 5.0, 1000)
        values = np.stack((times, -times), axis=1)
        matrix = np.asarray([[0.0, 1.0], [1.0, 0.0]])
        figure = create_animated_graph_figure(
            matrix, values, times, "Long simulation", 0
        )
        self.assertEqual(len(figure.frames), MAX_ANIMATION_FRAMES)
        self.assertEqual(len(figure.layout.sliders[0].steps), MAX_ANIMATION_FRAMES)
        self.assertEqual(figure.frames[0].layout.title.text, "Long simulation<br><sup>t = 0</sup>")
        self.assertEqual(figure.frames[-1].layout.title.text, "Long simulation<br><sup>t = 5</sup>")
        self.assertEqual(figure.layout.template.layout.paper_bgcolor, "rgb(17,17,17)")

    def test_generated_detail_page_contains_animations_and_dark_theme(self) -> None:
        record = self.make_record(3)
        with tempfile.TemporaryDirectory() as temp_directory:
            results_directory = Path(temp_directory)
            np.savez_compressed(
                results_directory / "solution_seed_1.npz",
                seed=np.asarray(record.seed),
                **record.arrays,
            )
            index_path = generate_batch_table(results_directory)
            detail_path = (
                results_directory
                / "batch_table_details"
                / "seed_1_dimension_3.html"
            )
            page = detail_path.read_text(encoding="utf-8")
            self.assertTrue(index_path.exists())
            self.assertEqual(page.count('class="plotly-graph-div"'), 11)
            self.assertEqual(page.count("cdn.plot.ly"), 1)
            self.assertEqual(page.count('"label":"Play"'), 4)
            self.assertEqual(page.count('"label":"Pause"'), 4)
            self.assertIn("color-scheme: dark", page)

    def test_graph_panels_use_matrix_edges_and_mark_hub(self) -> None:
        record = self.make_record(4)
        adjacency = np.asarray(
            [
                [0.0, 1.0, 0.0, 1.0],
                [1.0, 0.0, 1.0, 0.0],
                [0.0, 1.0, 0.0, 1.0],
                [1.0, 0.0, 1.0, 0.0],
            ]
        )
        record.arrays["matrix_a"] = adjacency
        record.arrays["matrix_b"] = adjacency.copy()
        record.arrays["sync_indices"] = np.asarray([2, 2])
        figures = create_detail_figures(record)
        graph_figure = figures[0]
        edges, nodes = graph_figure.data
        self.assertEqual(len(edges.x), 12)
        self.assertEqual(len(nodes.x), 4)
        self.assertEqual(nodes.marker.color[2], "#cf563f")

    def test_high_dimension_zero_state_has_finite_color_range(self) -> None:
        figures = create_detail_figures(self.make_record(16, zero=True))
        for figure in figures:
            for trace in figure.data:
                if isinstance(trace, go.Heatmap):
                    self.assertLess(trace.zmin, trace.zmax)


if __name__ == "__main__":
    unittest.main()