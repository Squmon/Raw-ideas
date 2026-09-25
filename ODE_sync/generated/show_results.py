"""Generate a clickable seed-by-dimension MSE table and detail pages."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import html
import json
import os
from pathlib import Path
import re

import numpy as np
import plotly.graph_objects as go


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
            template="plotly_white",
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
        template="plotly_white",
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
        template="plotly_white",
        height=380,
        hovermode="x unified",
        legend={"title": {"text": "Error metric"}},
    )
    error_figure.update_xaxes(title_text="Time")
    error_figure.update_yaxes(
        title_text="MSE", type="log" if np.all(mse > 0) else "linear"
    )
    return [states_a_figure, states_b_figure, hub_figure, error_figure, difference_figure]


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
        template="plotly_white",
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
    :root {{ color-scheme: light; font-family: system-ui, sans-serif; color: #17211f; background: #f4f7f5; }}
    body {{ margin: 0; }}
    header {{ padding: 18px 24px; border-bottom: 1px solid #d6dfda; background: #fff; }}
    header a {{ color: #176b55; font-weight: 650; text-decoration: none; }}
    main {{ max-width: 1500px; margin: 0 auto; padding: 12px 18px 36px; }}
    .plotly-graph-div {{ width: 100%; }}
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