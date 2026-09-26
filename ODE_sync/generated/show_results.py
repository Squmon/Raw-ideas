"""Generate a clickable seed-by-dimension MSE table and detail pages."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import html
import json
import os
from pathlib import Path
import re

import networkx as nx
import numpy as np
import plotly.graph_objects as go

MAX_ANIMATION_FRAMES = 120


@dataclass(frozen=True)
class ResultRecord:
    """Loaded result data and the grid coordinates used to index it."""

    path: Path
    seed: int
    dimension: int
    final_mse: float
    arrays: dict[str, np.ndarray]


def _read_seed(path: Path, arrays: dict[str, np.ndarray]) -> int:
    if "seed" in arrays:
        return int(arrays["seed"])
    match = re.search(r"solution_seed_(-?\d+)", path.stem)
    if match:
        return int(match.group(1))
    raise ValueError(
        f"Cannot identify a seed in {path.name}; regenerate it with the current "
        "pipeline, which stores seed metadata in each NPZ archive."
    )


def load_results(directory: Path) -> list[ResultRecord]:
    """Load result archives, deriving final MSE and grid coordinates."""
    paths = sorted(directory.glob("solution_*.npz"))
    if not paths:
        raise FileNotFoundError(f"No solution_*.npz files found in {directory}")

    records = []
    for path in paths:
        with np.load(path, allow_pickle=False) as archive:
            arrays = {name: archive[name].copy() for name in archive.files}
        state_a = arrays["state_a"]
        state_b = arrays["state_b"]
        if state_a.shape != state_b.shape or state_a.ndim != 2:
            raise ValueError(f"Invalid state array shapes in {path.name}")
        difference = state_a[-1] - state_b[-1]
        records.append(
            ResultRecord(
                path=path,
                seed=_read_seed(path, arrays),
                dimension=int(state_a.shape[1]),
                final_mse=float(np.mean(np.square(difference))),
                arrays=arrays,
            )
        )

    return records


def create_detail_figures(record: ResultRecord) -> list[go.Figure]:
    """Build independent Plotly charts so each panel has a local legend."""
    arrays = record.arrays
    ts = arrays["ts"]
    state_a = arrays["state_a"]
    state_b = arrays["state_b"]
    difference = state_a - state_b
    hub_a, hub_b = map(int, arrays["sync_indices"])
    config = json.loads(str(arrays["config_json"])) if "config_json" in arrays else {}
    coupling = float(arrays.get("coupling", config.get("coupling", np.nan)))
    mse = np.mean(np.square(difference), axis=1)
    use_heatmaps = record.dimension > 15

    def component_figure(values: np.ndarray, title: str) -> go.Figure:
        if use_heatmaps:
            color_limit = max(float(np.max(np.abs(values))), np.finfo(np.float32).eps)
            figure = go.Figure(
                go.Heatmap(
                    x=ts,
                    y=np.arange(record.dimension),
                    z=values.T,
                    colorscale="RdBu",
                    zmin=-color_limit,
                    zmax=color_limit,
                    zmid=0,
                    colorbar={"title": "Value"},
                    hovertemplate=(
                        "Time: %{x:.5g}<br>Component: %{y}<br>Value: %{z:.5g}"
                        "<extra></extra>"
                    ),
                )
            )
            y_title = "Component"
        else:
            figure = go.Figure()
            for component in range(record.dimension):
                figure.add_trace(
                    go.Scatter(
                        x=ts,
                        y=values[:, component],
                        mode="lines",
                        name=f"{title}[{component}]",
                    )
                )
            y_title = "State"
        figure.update_layout(
            title=title,
            template="plotly_dark",
            height=460,
            hovermode="x unified",
            legend={"title": {"text": title}, "groupclick": "togglegroup"},
        )
        figure.update_xaxes(title_text="Time")
        figure.update_yaxes(title_text=y_title)
        return figure
    states_a_figure = component_figure(state_a, "System a states")
    states_b_figure = component_figure(state_b, "System b states")
    difference_figure = component_figure(difference, "State differences (a - b)")

    hub_figure = go.Figure(
        [
            go.Scatter(x=ts, y=state_a[:, hub_a], mode="lines", name=f"a hub [{hub_a}]"),
            go.Scatter(
                x=ts,
                y=state_b[:, hub_b],
                mode="lines",
                name=f"b hub [{hub_b}]",
                line={"dash": "dash"},
            ),
        ]
    )
    hub_figure.update_layout(
        title=f"Coupled hub states (a[{hub_a}], b[{hub_b}])",
        template="plotly_dark",
        height=380,
        hovermode="x unified",
        legend={"title": {"text": "Hub state"}},
    )
    hub_figure.update_xaxes(title_text="Time")
    hub_figure.update_yaxes(title_text="State")

    error_figure = go.Figure(
        go.Scatter(x=ts, y=mse, mode="lines", name="Mean squared error")
    )
    error_figure.update_layout(
        title="Synchronization error",
        template="plotly_dark",
        height=380,
        hovermode="x unified",
        legend={"title": {"text": "Error metric"}},
    )
    error_figure.update_xaxes(title_text="Time")
    error_figure.update_yaxes(
        title_text="MSE", type="log" if np.all(mse > 0) else "linear"
    )
    graph_a = _graph_from_matrix(arrays["matrix_a"])
    graph_b = _graph_from_matrix(arrays["matrix_b"])
    positions_a = _graph_positions(graph_a)
    positions_b = _graph_positions(graph_b)
    graph_a_figure = create_graph_figure(
        arrays["matrix_a"], f"System A graph | seed {record.seed}", hub_a,
        graph=graph_a, positions=positions_a,
    )
    graph_b_figure = create_graph_figure(
        arrays["matrix_b"], f"System B graph | seed {record.seed}", hub_b,
        graph=graph_b, positions=positions_b,
    )
    animated_graph_figures = [
        create_animated_graph_figure(
            arrays["matrix_a"], state_a, ts, f"System A state | seed {record.seed}", hub_a,
            graph=graph_a, positions=positions_a,
        ),
        create_animated_graph_figure(
            arrays["matrix_b"], state_b, ts, f"System B state | seed {record.seed}", hub_b,
            graph=graph_b, positions=positions_b,
        ),
        create_animated_graph_figure(
            arrays["matrix_a"], difference, ts,
            f"State difference (A - B) | seed {record.seed}", hub_a,
            colorscale="RdBu", symmetric_scale=True, graph=graph_a, positions=positions_a,
        ),
        create_animated_graph_figure(
            arrays["matrix_a"], np.abs(difference), ts,
            f"Absolute state difference | seed {record.seed}", hub_a,
            graph=graph_a, positions=positions_a,
        ),
    ]
    return [
        graph_a_figure,
        graph_b_figure,
        *animated_graph_figures,
        states_a_figure,
        states_b_figure,
        hub_figure,
        error_figure,
        difference_figure,
    ]


def _graph_from_matrix(matrix: np.ndarray) -> nx.Graph:
    matrix = np.asarray(matrix)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("graph matrix must be square")
    graph = nx.Graph()
    graph.add_nodes_from(range(matrix.shape[0]))
    for source in range(matrix.shape[0]):
        for target in range(source + 1, matrix.shape[0]):
            weight = max(
                abs(float(matrix[source, target])),
                abs(float(matrix[target, source])),
            )
            if weight > 0:
                graph.add_edge(source, target, weight=weight)
    return graph


def _graph_positions(graph: nx.Graph) -> dict[int, np.ndarray]:
    if graph.number_of_edges():
        return nx.kamada_kawai_layout(graph, weight=None)
    return nx.circular_layout(graph)


def _graph_edge_coordinates(
    graph: nx.Graph,
    positions: dict[int, np.ndarray],
) -> tuple[list[float | None], list[float | None]]:
    edge_x: list[float | None] = []
    edge_y: list[float | None] = []
    for source, target in graph.edges:
        edge_x.extend((positions[source][0], positions[target][0], None))
        edge_y.extend((positions[source][1], positions[target][1], None))
    return edge_x, edge_y


def create_animated_graph_figure(
    matrix: np.ndarray,
    values_by_time: np.ndarray,
    times: np.ndarray,
    title: str,
    hub_index: int,
    *,
    colorscale: str = "Viridis",
    symmetric_scale: bool = False,
    max_frames: int = MAX_ANIMATION_FRAMES,
    graph: nx.Graph | None = None,
    positions: dict[int, np.ndarray] | None = None,
) -> go.Figure:
    """Animate node colors over time while keeping layout and edges fixed."""
    matrix = np.asarray(matrix)
    values_by_time = np.asarray(values_by_time)
    times = np.asarray(times)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("graph matrix must be square")
    if values_by_time.ndim != 2 or values_by_time.shape[1] != matrix.shape[0]:
        raise ValueError("values_by_time must have shape (n_times, graph_dimension)")
    if values_by_time.shape[0] != len(times):
        raise ValueError("times length must match values_by_time")
    if max_frames < 2:
        raise ValueError("max_frames must be at least 2")

    frame_indices = np.unique(
        np.linspace(
            0,
            len(times) - 1,
            min(max_frames, len(times)),
            dtype=np.int64,
        )
    )
    frame_times = times[frame_indices]
    frame_values = values_by_time[frame_indices]

    graph = graph if graph is not None else _graph_from_matrix(matrix)
    positions = positions if positions is not None else _graph_positions(graph)
    node_ids = list(graph.nodes)
    node_x = [positions[node][0] for node in node_ids]
    node_y = [positions[node][1] for node in node_ids]
    degrees = [int(graph.degree[node]) for node in node_ids]
    weighted_degrees = [
        float(sum(data["weight"] for _, _, data in graph.edges(node, data=True)))
        for node in node_ids
    ]
    edge_x, edge_y = _graph_edge_coordinates(graph, positions)
    max_abs_value = max(float(np.max(np.abs(values_by_time))), np.finfo(np.float32).eps)
    if symmetric_scale:
        color_min, color_max = -max_abs_value, max_abs_value
    else:
        color_min = min(0.0, float(np.min(values_by_time)))
        color_max = max_abs_value

    initial_values = frame_values[0]
    initial_customdata = np.column_stack((degrees, weighted_degrees, initial_values))
    figure = go.Figure(
        data=[
            go.Scatter(
                x=edge_x,
                y=edge_y,
                mode="lines",
                line={"color": "#536273", "width": 1.5},
                hoverinfo="skip",
                name="connections",
            ),
            go.Scatter(
                x=node_x,
                y=node_y,
                mode="markers+text",
                text=[str(node) for node in node_ids],
                textposition="top center",
                marker={
                    "size": 15,
                    "color": initial_values,
                    "colorscale": colorscale,
                    "cmin": color_min,
                    "cmax": color_max,
                    "showscale": True,
                    "colorbar": {"title": title.split("|")[0].strip()},
                    "line": {
                        "color": ["#ffcf70" if node == hub_index else "#dce4ed" for node in node_ids],
                        "width": [3 if node == hub_index else 1 for node in node_ids],
                    },
                },
                customdata=initial_customdata,
                hovertemplate=(
                    "Node %{text}<br>Degree %{customdata[0]}"
                    "<br>Weighted degree %{customdata[1]:.4g}"
                    "<br>Value %{customdata[2]:.5g}<extra></extra>"
                ),
                name="vertices",
            ),
        ],
        frames=[
            go.Frame(
                name=str(frame_index),
                data=[
                    go.Scatter(
                        marker={"color": frame_values_at_time},
                        customdata=np.column_stack(
                            (degrees, weighted_degrees, frame_values_at_time)
                        ),
                    )
                ],
                traces=[1],
                layout={"title": f"{title}<br><sup>t = {time_value:.5g}</sup>"},
            )
            for frame_index, (time_value, frame_values_at_time) in enumerate(
                zip(frame_times, frame_values, strict=True)
            )
        ],
    )
    frame_names = [str(frame_index) for frame_index in range(len(frame_times))]
    figure.update_layout(
        title=f"{title}<br><sup>t = {frame_times[0]:.5g}</sup>",
        template="plotly_dark",
        height=560,
        showlegend=False,
        margin={"l": 20, "r": 30, "t": 80, "b": 80},
        updatemenus=[
            {
                "type": "buttons",
                "showactive": False,
                "x": 0.02,
                "y": -0.12,
                "buttons": [
                    {
                        "label": "Play",
                        "method": "animate",
                        "args": [
                            None,
                            {"frame": {"duration": 120, "redraw": True}, "fromcurrent": True},
                        ],
                    },
                    {
                        "label": "Pause",
                        "method": "animate",
                        "args": [[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate"}],
                    },
                ],
            }
        ],
        sliders=[
            {
                "active": 0,
                "x": 0.20,
                "len": 0.78,
                "y": -0.12,
                "currentvalue": {"prefix": "Time: "},
                "steps": [
                    {
                        "label": f"{time_value:.4g}",
                        "method": "animate",
                        "args": [
                            [frame_name],
                            {"mode": "immediate", "frame": {"duration": 0, "redraw": True}, "transition": {"duration": 0}},
                        ],
                    }
                    for frame_name, time_value in zip(
                        frame_names, frame_times, strict=True
                    )
                ],
            }
        ],
    )
    figure.update_xaxes(visible=False, scaleanchor="y", scaleratio=1)
    figure.update_yaxes(visible=False)
    return figure


def create_graph_figure(
    matrix: np.ndarray,
    title: str,
    hub_index: int,
    *,
    graph: nx.Graph | None = None,
    positions: dict[int, np.ndarray] | None = None,
) -> go.Figure:
    """Visualize a system matrix as an undirected weighted graph."""
    graph = graph if graph is not None else _graph_from_matrix(matrix)
    positions = positions if positions is not None else _graph_positions(graph)

    edge_x = []
    edge_y = []
    for source, target in graph.edges:
        edge_x.extend((positions[source][0], positions[target][0], None))
        edge_y.extend((positions[source][1], positions[target][1], None))

    node_x = [positions[node][0] for node in graph.nodes]
    node_y = [positions[node][1] for node in graph.nodes]
    node_colors = ["#cf563f" if node == hub_index else "#267c72" for node in graph.nodes]
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=edge_x,
            y=edge_y,
            mode="lines",
            line={"color": "#a7b4af", "width": 1.5},
            hoverinfo="skip",
            name="connections",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=node_x,
            y=node_y,
            mode="markers+text",
            text=[str(node) for node in graph.nodes],
            textposition="top center",
            marker={"size": 13, "color": node_colors, "line": {"color": "white", "width": 1}},
            customdata=[
                [int(graph.degree[node]), float(sum(data["weight"] for _, _, data in graph.edges(node, data=True)))]
                for node in graph.nodes
            ],
            hovertemplate=(
                "Node %{text}<br>Degree %{customdata[0]}"
                "<br>Weighted degree %{customdata[1]:.4g}<extra></extra>"
            ),
            name="nodes",
        )
    )
    figure.update_layout(
        title=title,
        template="plotly_dark",
        height=470,
        showlegend=False,
        margin={"l": 20, "r": 20, "t": 60, "b": 20},
    )
    figure.update_xaxes(visible=False, scaleanchor="y", scaleratio=1)
    figure.update_yaxes(visible=False)
    return figure


def create_index_figure(
    records: list[ResultRecord],
    output_path: Path,
    details_directory: Path,
) -> go.Figure:
    """Create the clickable Plotly heatmap indexed by seed and dimension."""
    seeds = sorted({record.seed for record in records})
    dimensions = sorted({record.dimension for record in records})
    by_coordinate: dict[tuple[int, int], list[ResultRecord]] = {}
    for record in records:
        by_coordinate.setdefault((record.seed, record.dimension), []).append(record)
    z_values = []
    customdata = []
    labels = []
    for dimension in dimensions:
        z_row = []
        link_row = []
        label_row = []
        for seed in seeds:
            coordinate_records = by_coordinate.get((seed, dimension), [])
            if not coordinate_records:
                z_row.append(np.nan)
                link_row.append(["", 0])
                label_row.append("")
                continue
            mean_mse = float(np.mean([record.final_mse for record in coordinate_records]))
            detail_path = details_directory / f"seed_{seed}_dimension_{dimension}.html"
            relative_target = os.path.relpath(detail_path, start=output_path.parent)
            link_row.append([Path(relative_target).as_posix(), len(coordinate_records)])
            z_row.append(mean_mse)
            label_row.append(f"{mean_mse:.3g}")
        z_values.append(z_row)
        customdata.append(link_row)
        labels.append(label_row)

    positive_errors = [value for row in z_values for value in row if np.isfinite(value) and value > 0]
    zmin = min(positive_errors) if positive_errors else 0.0
    zmax = max(positive_errors) if positive_errors else 1.0
    if zmin == zmax:
        zmax = zmin * 1.01 if zmin > 0 else 1.0
    colorbar = {"title": "Final MSE"}
    figure = go.Figure(
        go.Heatmap(
            x=[str(seed) for seed in seeds],
            y=[str(dimension) for dimension in dimensions],
            z=z_values,
            zmin=zmin,
            zmax=zmax,
            zauto=False,
            colorscale="Viridis",
            colorbar=colorbar,
            customdata=customdata,
            text=labels,
            texttemplate="%{text}",
            hovertemplate=(
                "seed=%{x}<br>dimension=%{y}<br>mean final MSE=%{z:.6g}"
                "<br>runs=%{customdata[1]}"
                "<extra>Click to open system plots</extra>"
            ),
            xgap=3,
            ygap=3,
        )
    )
    figure.update_layout(
        title="Final synchronization MSE by seed and dimension",
        template="plotly_dark",
        height=max(430, 90 * len(dimensions) + 180),
        margin={"l": 90, "r": 50, "t": 90, "b": 80},
        clickmode="event",
        xaxis={"title": "Random seed", "side": "bottom"},
        yaxis={"title": "System dimension", "autorange": "reversed", "scaleanchor": "x", "scaleratio": 1},
    )
    return figure


def _html_page(title: str, content: str, extra_script: str = "") -> str:
    safe_title = html.escape(title)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{safe_title}</title>
  <style>
    :root {{ color-scheme: dark; font-family: system-ui, sans-serif; color: #e4ebf3; background: #111820; }}
    body {{ margin: 0; background: #111820; color: #e4ebf3; }}
    header {{ padding: 18px 24px; border-bottom: 1px solid #2d3947; background: #19232e; }}
    header a {{ color: #88d2bf; font-weight: 650; text-decoration: none; }}
    main {{ max-width: 1500px; margin: 0 auto; padding: 12px 18px 36px; }}
    .plotly-graph-div {{ width: 100%; }}
    section {{ margin: 0 0 28px; padding: 16px; border-bottom: 1px solid #2d3947; }}
    h1, h2, p, summary {{ color: #e4ebf3; }}
    code {{ color: #b7d5f2; overflow-wrap: anywhere; }}
    a {{ color: #88d2bf; }}
    @media (max-width: 640px) {{ header {{ padding: 14px 16px; }} main {{ padding: 8px 4px 24px; }} }}
  </style>
</head>
<body>
{content}
{extra_script}
</body>
</html>
"""


def generate_batch_table(directory: Path, output_path: Path | None = None) -> Path:
    """Generate an index page and one linked Plotly page for each archive."""
    result_directory = directory.resolve()
    records = load_results(result_directory)
    index_path = output_path or result_directory / "batch_table.html"
    index_path = index_path.resolve()
    index_path.parent.mkdir(parents=True, exist_ok=True)
    details_directory = index_path.parent / f"{index_path.stem}_details"
    details_directory.mkdir(parents=True, exist_ok=True)

    grouped_records: dict[tuple[int, int], list[ResultRecord]] = {}
    for record in records:
        grouped_records.setdefault((record.seed, record.dimension), []).append(record)

    for (seed, dimension), coordinate_records in grouped_records.items():
        detail_sections = []
        include_plotly = True
        for index, record in enumerate(coordinate_records):
            config = (
                json.loads(str(record.arrays["config_json"]))
                if "config_json" in record.arrays
                else {}
            )
            plot_sections = []
            for plot_index, detail_figure in enumerate(create_detail_figures(record)):
                detail_plot = detail_figure.to_html(
                    full_html=False,
                    include_plotlyjs="cdn" if include_plotly else False,
                    div_id=f"system-plots-{index}-{plot_index}",
                )
                include_plotly = False
                plot_sections.append(detail_plot)
            settings = html.escape(json.dumps(config, sort_keys=True))
            detail_sections.append(
                f"<section><h2>Run {index + 1}: {html.escape(record.path.name)}</h2>"
                f"<p>Final MSE: {record.final_mse:.6g} | Configuration: <code>{settings}</code></p>"
                f"{''.join(plot_sections)}</section>"
            )
        average_mse = float(np.mean([record.final_mse for record in coordinate_records]))
        detail_content = (
            '<header><a href="../' + html.escape(index_path.name) + '">Back to results table</a></header>'
            f"<main><h1>Seed {seed}, dimension {dimension}</h1>"
            f"<p>{len(coordinate_records)} run(s); mean final MSE: {average_mse:.6g}</p>"
            f"{''.join(detail_sections)}</main>"
        )
        detail_page = _html_page(
            f"Seed {seed}, dimension {dimension}", detail_content
        )
        (details_directory / f"seed_{seed}_dimension_{dimension}.html").write_text(
            detail_page, encoding="utf-8"
        )

    index_figure = create_index_figure(records, index_path, details_directory)
    click_script = """
<script>
const grid = document.getElementById('seed-dimension-grid');
grid.on('plotly_click', (event) => {
    const target = event.points[0]?.customdata?.[0];
  if (typeof target === 'string' && target.length > 0) {
    window.location.href = target;
  }
});
</script>
"""
    plot_html = index_figure.to_html(
        full_html=False,
        include_plotlyjs="cdn",
        div_id="seed-dimension-grid",
    )
    links = "".join(
        f'<li><a href="{html.escape(details_directory.name)}/seed_{seed}_dimension_{dimension}.html">'
        f"seed {seed}, dimension {dimension}: mean final MSE "
        f"{np.mean([record.final_mse for record in coordinate_records]):.6g} "
        f"({len(coordinate_records)} runs)</a></li>"
        for (seed, dimension), coordinate_records in sorted(grouped_records.items())
    )
    content = (
        "<header><strong>Synchronization batch results</strong></header>"
        f"<main><p>{len(records)} systems across {len(grouped_records)} seed/dimension cells. "
        "Cell colors show mean final MSE across matching runs. Click a square to inspect every run in that cell.</p>"
        f"{plot_html}<details><summary>Accessible result links</summary><ul>{links}</ul></details></main>"
    )
    index_path.write_text(
        _html_page("Synchronization batch results", content, click_script),
        encoding="utf-8",
    )
    return index_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "directory",
        nargs="?",
        type=Path,
        default=Path(__file__).parent / "results",
        help="Directory containing solution_*.npz archives.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Index HTML path (defaults to <directory>/batch_table.html).",
    )
    args = parser.parse_args()
    output_path = generate_batch_table(args.directory, args.output)
    print(f"Generated clickable results table: {output_path}")


if __name__ == "__main__":
    main()