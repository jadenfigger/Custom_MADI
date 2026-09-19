"""Interactive slicing explorer for the MADI model manifold.

    python -m tools.manifold_explorer.app            # then open the printed URL

What you are looking at
-----------------------
The library is a cloud of ~18,800 points, one per (rho, V, k_io) triple, living
in a 31,125-dimensional prediction space whose axes are acquisition columns
(delta, Delta, b).  Choose some columns as MEASURED and a reference point, and
the app keeps the entries whose predictions on those columns agree with the
reference to within the noise you set.  The survivors are what a real
measurement at those columns could not tell apart:

* spread out in (rho, V)  ->  those parameters are not identifiable there;
* tight on an unmeasured DISPLAY column  ->  that column is already implied.

Adding a second Delta to the measured set and watching the (rho, V) spread
shrink is the experiment this tool exists for.
"""

from __future__ import annotations

import argparse
import functools
import time
import webbrowser
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from dash import Dash, Input, Output, State, callback_context, dcc, html, no_update

from . import slice as slicing
from .build_column_cache import DEFAULT_LIBRARY
from .columns import cache_is_valid, load_labels, open_reader

PARAMETER_CHOICES = {
    "rho": "rho (1/uL)",
    "V": "V (pL)",
    "k_io": "k_io (1/s)",
    "vi": "rho*V (v_i)",
}
CONTROL_STYLE = {"marginBottom": "10px"}
PANEL_STYLE = {"width": "330px", "padding": "12px", "overflowY": "auto",
               "height": "97vh", "borderRight": "1px solid #ddd",
               "fontFamily": "system-ui, sans-serif", "fontSize": "13px"}


class ExplorerData:
    """Labels plus a small memoised window onto the library's columns."""

    def __init__(self, library_path: Path, cache_dir: Path | None = None,
                 prefer_cache: bool = True):
        self.library_path = Path(library_path)
        self.labels = load_labels(self.library_path)
        self.reader = open_reader(self.library_path, cache_dir, prefer_cache)
        self.cache_ok, self.cache_note = cache_is_valid(self.library_path, cache_dir)

    @functools.lru_cache(maxsize=64)
    def _read(self, columns: tuple[int, ...]) -> np.ndarray:
        return self.reader.read(np.asarray(columns, dtype=np.int64))

    @functools.lru_cache(maxsize=64)
    def _read_variance(self, columns: tuple[int, ...]) -> np.ndarray:
        return self.reader.read_variance(np.asarray(columns, dtype=np.int64))

    def block(self, columns) -> np.ndarray:
        """(n_entries, len(columns)) S/S0, memoised on the column set."""
        return self._read(tuple(int(c) for c in columns))

    def variance(self, columns) -> np.ndarray:
        return self._read_variance(tuple(int(c) for c in columns))

    def nearest_entry(self, rho: float, V: float, kio: float,
                      eligible: np.ndarray) -> int:
        """Library row closest to a requested parameter triple.

        Distance is taken in (log rho, log V, k_io) because the rho and V grids
        are log-spaced; k_io is linear, scaled by its range so it counts
        comparably.
        """
        labels = self.labels
        target = np.array([np.log(max(rho, 1e-12)), np.log(max(V, 1e-12)), kio])
        grid = np.column_stack([
            np.log(np.maximum(labels.nominal_rhos[eligible], 1e-12)),
            np.log(np.maximum(labels.nominal_Vs[eligible], 1e-12)),
            labels.kios[eligible],
        ])
        scale = np.array([1.0, 1.0, 1.0 / max(np.ptp(labels.kios[eligible]), 1e-9)])
        distance = np.sum(((grid - target) * scale) ** 2, axis=1)
        return int(eligible[int(np.argmin(distance))])


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def _row_ids(rows: np.ndarray) -> list[int]:
    """Library row indices as a plain list, for Plotly `customdata`.

    Plotly 6 base64-encodes numpy arrays into {dtype, bdata}, and plotly.js
    does not expand that back into per-point customdata, so a numpy array here
    silently produces click events with no customdata and clicking a point
    does nothing.  A plain list round-trips correctly.
    """
    return [int(row) for row in np.asarray(rows).ravel()]


def _colour_values(labels, rows: np.ndarray, parameter: str) -> np.ndarray:
    if parameter == "rho":
        return np.log10(np.maximum(labels.nominal_rhos[rows], 1e-12))
    if parameter == "V":
        return np.log10(np.maximum(labels.nominal_Vs[rows], 1e-12))
    if parameter == "k_io":
        return labels.kios[rows]
    return labels.vis[rows]


def _colour_title(parameter: str) -> str:
    return {"rho": "log10 rho", "V": "log10 V", "k_io": "k_io",
            "vi": "v_i = rho*V"}[parameter]


def _hover_text(labels, rows: np.ndarray) -> list[str]:
    return [
        f"row {row}<br>rho={labels.nominal_rhos[row]:.3g}"
        f"<br>V={labels.nominal_Vs[row]:.3g}"
        f"<br>k_io={labels.kios[row]:.3g}"
        f"<br>v_i={labels.vis[row]:.3f}"
        for row in rows
    ]


def prediction_figure(data: ExplorerData, display_columns, display_block,
                      eligible, survivors, reference_row, colour_by):
    """Scatter of every entry on 2 or 3 chosen prediction axes."""
    labels = data.labels
    names = [labels.column_label(int(c)) for c in display_columns]
    survivor_rows = eligible[survivors]
    figure = go.Figure()

    if len(display_columns) >= 3:
        figure.add_trace(go.Scatter3d(
            x=display_block[:, 0], y=display_block[:, 1], z=display_block[:, 2],
            mode="markers", name="all entries",
            marker=dict(size=1.6, color="#bbbbbb", opacity=0.35),
            customdata=_row_ids(eligible), hoverinfo="skip",
        ))
        figure.add_trace(go.Scatter3d(
            x=display_block[survivors, 0], y=display_block[survivors, 1],
            z=display_block[survivors, 2], mode="markers", name="slice survivors",
            marker=dict(size=3.4, color=_colour_values(labels, survivor_rows, colour_by),
                        colorscale="Viridis", showscale=True,
                        colorbar=dict(title=_colour_title(colour_by), thickness=12)),
            customdata=_row_ids(survivor_rows), text=_hover_text(labels, survivor_rows),
            hovertemplate="%{text}<extra></extra>",
        ))
        if reference_row is not None:
            where = int(np.flatnonzero(eligible == reference_row)[0])
            figure.add_trace(go.Scatter3d(
                x=[display_block[where, 0]], y=[display_block[where, 1]],
                z=[display_block[where, 2]], mode="markers", name="reference",
                marker=dict(size=7, color="red", symbol="x"),
            ))
        figure.update_layout(scene=dict(xaxis_title=names[0], yaxis_title=names[1],
                                        zaxis_title=names[2]))
    else:
        figure.add_trace(go.Scattergl(
            x=display_block[:, 0], y=display_block[:, 1], mode="markers",
            name="all entries", marker=dict(size=3, color="#cccccc"),
            customdata=_row_ids(eligible), text=_hover_text(labels, eligible),
            hovertemplate="%{text}<extra></extra>",
        ))
        figure.add_trace(go.Scattergl(
            x=display_block[survivors, 0], y=display_block[survivors, 1],
            mode="markers", name="slice survivors",
            marker=dict(size=6, color=_colour_values(labels, survivor_rows, colour_by),
                        colorscale="Viridis", showscale=True,
                        colorbar=dict(title=_colour_title(colour_by), thickness=12)),
            customdata=_row_ids(survivor_rows), text=_hover_text(labels, survivor_rows),
            hovertemplate="%{text}<extra></extra>",
        ))
        if reference_row is not None:
            where = int(np.flatnonzero(eligible == reference_row)[0])
            figure.add_trace(go.Scattergl(
                x=[display_block[where, 0]], y=[display_block[where, 1]],
                mode="markers", name="reference",
                marker=dict(size=14, color="red", symbol="x-thin",
                            line=dict(width=2.5, color="red")),
            ))
        figure.update_layout(xaxis_title=names[0], yaxis_title=names[1])

    figure.update_layout(
        title="Prediction space (display columns)", height=430,
        margin=dict(l=55, r=10, t=40, b=45), showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    return figure


def parameter_figure(data: ExplorerData, eligible, survivors, reference_row,
                     x_name: str, y_name: str, colour_by: str):
    """Survivors in a parameter plane, log axes where the grid is log-spaced."""
    labels = data.labels
    axes = {
        "rho": (labels.nominal_rhos, True, "rho (1/uL)"),
        "V": (labels.nominal_Vs, True, "V (pL)"),
        "k_io": (labels.kios, False, "k_io (1/s)"),
    }
    x_values, x_log, x_title = axes[x_name]
    y_values, y_log, y_title = axes[y_name]
    survivor_rows = eligible[survivors]

    figure = go.Figure()
    figure.add_trace(go.Scattergl(
        x=x_values[eligible], y=y_values[eligible], mode="markers",
        name="all entries", marker=dict(size=3, color="#dddddd"),
        customdata=_row_ids(eligible), hoverinfo="skip",
    ))
    figure.add_trace(go.Scattergl(
        x=x_values[survivor_rows], y=y_values[survivor_rows], mode="markers",
        name="survivors",
        marker=dict(size=6, color=_colour_values(labels, survivor_rows, colour_by),
                    colorscale="Viridis"),
        customdata=_row_ids(survivor_rows), text=_hover_text(labels, survivor_rows),
        hovertemplate="%{text}<extra></extra>",
    ))
    if reference_row is not None:
        figure.add_trace(go.Scattergl(
            x=[x_values[reference_row]], y=[y_values[reference_row]], mode="markers",
            name="reference",
            marker=dict(size=14, color="red", symbol="x-thin",
                        line=dict(width=2.5, color="red")),
        ))
    figure.update_layout(
        title=f"{x_title} vs {y_title}", height=340, showlegend=False,
        margin=dict(l=60, r=10, t=35, b=45),
        xaxis=dict(title=x_title, type="log" if x_log else "linear"),
        yaxis=dict(title=y_title, type="log" if y_log else "linear"),
    )
    return figure


def widths_figure(curve: list[dict]) -> go.Figure:
    """How the parameter spreads shrink as measured columns are added."""
    figure = go.Figure()
    counts = [step["n_measured"] for step in curve]
    figure.add_trace(go.Scatter(x=counts, y=[s["rho_log_width"] for s in curve],
                                name="rho max/min", mode="lines+markers"))
    figure.add_trace(go.Scatter(x=counts, y=[s["V_log_width"] for s in curve],
                                name="V max/min", mode="lines+markers"))
    figure.add_trace(go.Scatter(x=counts, y=[s["n_survivors"] for s in curve],
                                name="survivors", mode="lines+markers", yaxis="y2"))
    figure.update_layout(
        title="Slice width vs number of measured columns (added in selection order)",
        height=320, margin=dict(l=60, r=60, t=40, b=45),
        xaxis_title="measured columns used",
        yaxis=dict(title="parameter width (max/min)", type="log"),
        yaxis2=dict(title="survivors", overlaying="y", side="right", type="log"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    return figure


def readout_table(result: slicing.SliceResult, n_eligible: int,
                  n_measured: int, elapsed_ms: float) -> html.Div:
    """Counts and widths: the numbers the plots are a picture of."""
    def row(cells, header=False):
        tag = html.Th if header else html.Td
        style = {"padding": "2px 8px", "borderBottom": "1px solid #eee",
                 "textAlign": "right"}
        return html.Tr([tag(cell, style=style) for cell in cells])

    head = row(["quantity", "n", "min", "max", "std", "max/min"], header=True)
    body = []
    for name, stats in result.parameter_stats.items():
        body.append(row([
            name, stats["n"], f"{stats['min']:.4g}", f"{stats['max']:.4g}",
            f"{stats['std']:.4g}",
            "-" if not np.isfinite(stats["log_width"]) else f"{stats['log_width']:.3f}",
        ]))
    for name, stats in result.display_stats.items():
        body.append(row([
            f"col {name}", stats["n"], f"{stats['min']:.5g}", f"{stats['max']:.5g}",
            f"{stats['std']:.3g}", "-",
        ]))

    weight = float(np.exp(-result.threshold / 2.0))
    summary = (
        f"{result.n_survivors} survivors of {n_eligible} eligible entries "
        f"({100.0 * result.n_survivors / max(n_eligible, 1):.2f}%)  |  "
        f"{n_measured} measured columns, chi2 <= {result.threshold:.3g} "
        f"(relative bayes weight >= {weight:.3g})  |  sigma = "
        f"{result.sigma_measurement:.4g}, S0 {result.s0_mode}  |  {elapsed_ms:.0f} ms"
    )
    return html.Div([
        html.P(summary, style={"fontWeight": "600", "margin": "6px 0"}),
        html.Table([html.Thead(head), html.Tbody(body)],
                   style={"borderCollapse": "collapse", "fontSize": "12px"}),
    ])


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------

def build_layout(data: ExplorerData) -> html.Div:
    labels = data.labels
    deltas = [{"label": f"{d:g}", "value": float(d)} for d in labels.deltas()]
    b_options = [{"label": f"{b:g}", "value": float(b)} for b in labels.b_values]
    default_delta = 20.0 if 20.0 in set(labels.deltas()) else float(labels.deltas()[0])
    default_Deltas = labels.Deltas_for_delta(default_delta)
    default_Delta = float(default_Deltas[-1])
    # The options must exist in the served layout: a Dropdown whose value is
    # not among its options renders blank and reports None to callbacks.
    Delta_options = [{"label": f"{D:g}", "value": float(D)} for D in default_Deltas]

    # Open on a working slice rather than an empty page: one (delta, Delta),
    # b up to 6000, with three of those columns as the plot axes.
    seed_columns = labels.columns_for_pair(default_delta, default_Delta,
                                           b_max=6000.0).tolist()
    seed_display = [seed_columns[i] for i in (2, 6, len(seed_columns) - 1)][:3]
    display_options = [{"label": labels.column_label(c), "value": int(c)}
                       for c in seed_columns]

    status = (f"reader: {data.reader.kind}"
              + ("" if data.cache_ok else f"  ({data.cache_note})"))

    def labelled(text, control):
        return html.Div([html.Label(text, style={"fontWeight": "600"}), control],
                        style=CONTROL_STYLE)

    controls = html.Div([
        html.H3("MADI manifold slicer", style={"marginTop": 0}),
        html.Div(status, style={"color": "#666", "fontSize": "11px"}),
        html.Div(f"{labels.n_entries} entries x {labels.n_columns} columns",
                 style={"color": "#666", "fontSize": "11px", "marginBottom": "10px"}),

        html.Hr(),
        html.H4("1. Measured columns"),
        labelled("delta (ms)", dcc.Dropdown(id="pick-delta", options=deltas,
                                            value=default_delta, clearable=False)),
        labelled("Delta (ms)", dcc.Dropdown(id="pick-Delta", options=Delta_options,
                                            value=default_Delta, clearable=False)),
        labelled("b-values (empty = all)",
                 dcc.Dropdown(id="pick-b", options=b_options, multi=True, value=[])),
        html.Div([
            html.Button("add these", id="add-columns", n_clicks=0),
            html.Button("add one b across all Delta", id="add-b-sweep", n_clicks=0,
                        style={"marginLeft": "6px"}),
            html.Button("clear", id="clear-columns", n_clicks=0,
                        style={"marginLeft": "6px"}),
        ], style=CONTROL_STYLE),
        html.Div(id="measured-summary", style={"fontSize": "11px", "color": "#444"}),

        html.Hr(),
        html.H4("2. Display columns (plot axes)"),
        labelled("axes", dcc.Dropdown(id="display-columns", multi=True,
                                      value=seed_display, options=display_options,
                                      placeholder="pick 2 or 3 columns")),
        html.Div("Choices come from the (delta, Delta) selected above plus any "
                 "measured columns.", style={"fontSize": "11px", "color": "#666"}),

        html.Hr(),
        html.H4("3. Reference point"),
        labelled("mode", dcc.RadioItems(
            id="reference-mode",
            options=[{"label": " click / parameters", "value": "entry"},
                     {"label": " pasted signal values", "value": "paste"}],
            value="entry")),
        html.Div([
            html.Div([html.Span("rho "), dcc.Input(id="ref-rho", type="number",
                                                   value=3.0e5, style={"width": "90px"})]),
            html.Div([html.Span("V   "), dcc.Input(id="ref-V", type="number",
                                                   value=2.0, style={"width": "90px"})]),
            html.Div([html.Span("k_io"), dcc.Input(id="ref-kio", type="number",
                                                   value=20.0, style={"width": "90px"})]),
            html.Button("snap to nearest entry", id="snap-reference", n_clicks=0,
                        style={"marginTop": "5px"}),
        ], style=CONTROL_STYLE),
        labelled("pasted S/S0 for the measured columns (comma or space separated)",
                 dcc.Textarea(id="ref-paste", value="", style={"width": "100%",
                                                               "height": "60px"})),
        html.Div(id="reference-summary", style={"fontSize": "11px", "color": "#444"}),

        html.Hr(),
        html.H4("4. Slice"),
        labelled("sigma (S/S0 units)", html.Div([
            dcc.Slider(id="sigma-exponent", min=-6, max=0, step=0.1, value=np.log10(0.02),
                       marks={-6: "1e-6", -4: "1e-4", -2: "1e-2", 0: "1"}),
            html.Div(id="sigma-readout", style={"fontSize": "11px"}),
        ])),
        labelled("reduced chi2 threshold (chi2 / n_measured)",
                 dcc.Slider(id="reduced-threshold", min=0.0, max=10.0, step=0.1, value=1.0,
                            marks={0: "0", 1: "1", 5: "5", 10: "10"})),
        labelled("S0 handling", dcc.RadioItems(
            id="s0-mode",
            options=[{"label": " fixed (data already S/S0)", "value": "fixed"},
                     {"label": " marginalised (free amplitude)", "value": "free"}],
            value="fixed")),
        dcc.Checklist(
            id="slice-options",
            options=[{"label": " add library MC variance in quadrature",
                      "value": "variance"},
                     {"label": " include the free-water atom", "value": "free_water"}],
            value=["variance"], style=CONTROL_STYLE),
        labelled("v_i band", html.Div([
            dcc.Input(id="vi-min", type="number", value=0.40, step=0.01,
                      style={"width": "70px"}),
            dcc.Input(id="vi-max", type="number", value=0.99, step=0.01,
                      style={"width": "70px", "marginLeft": "6px"}),
        ])),
        labelled("colour by", dcc.Dropdown(
            id="colour-by",
            options=[{"label": text, "value": key}
                     for key, text in PARAMETER_CHOICES.items()],
            value="k_io", clearable=False)),
        dcc.Checklist(id="widths-toggle",
                      options=[{"label": " compute widths vs column count",
                                "value": "on"}],
                      value=[], style=CONTROL_STYLE),
    ], style=PANEL_STYLE)

    plots = html.Div([
        html.Div(id="readout", style={"fontFamily": "system-ui, sans-serif",
                                      "padding": "6px 12px"}),
        dcc.Graph(id="prediction-plot"),
        html.Div([
            html.Div(dcc.Graph(id="rho-V-plot"), style={"flex": "1"}),
            html.Div(dcc.Graph(id="V-kio-plot"), style={"flex": "1"}),
        ], style={"display": "flex"}),
        dcc.Graph(id="widths-plot"),
    ], style={"flex": "1", "height": "97vh", "overflowY": "auto"})

    return html.Div([
        dcc.Store(id="measured-columns", data=seed_columns),
        dcc.Store(id="reference-row", data=None),
        controls, plots,
    ], style={"display": "flex"})


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------

def register_callbacks(app: Dash, data: ExplorerData) -> None:
    labels = data.labels

    @app.callback(Output("pick-Delta", "options"), Output("pick-Delta", "value"),
                  Input("pick-delta", "value"), State("pick-Delta", "value"))
    def _delta_options(delta, current):
        """Delta >= delta, so the offered Deltas change with delta."""
        available = labels.Deltas_for_delta(float(delta))
        options = [{"label": f"{D:g}", "value": float(D)} for D in available]
        keep = current if current is not None and float(current) in set(available) \
            else float(available[-1])
        return options, keep

    @app.callback(Output("sigma-readout", "children"), Input("sigma-exponent", "value"))
    def _sigma_readout(exponent):
        return f"sigma = {10.0 ** float(exponent):.4g}"

    @app.callback(
        Output("measured-columns", "data"),
        Input("add-columns", "n_clicks"), Input("add-b-sweep", "n_clicks"),
        Input("clear-columns", "n_clicks"),
        State("pick-delta", "value"), State("pick-Delta", "value"),
        State("pick-b", "value"), State("measured-columns", "data"),
        prevent_initial_call=True,
    )
    def _edit_columns(add, sweep, clear, delta, Delta, b_values, current):
        trigger = callback_context.triggered_id
        if trigger == "clear-columns":
            return []
        if delta is None or Delta is None:
            return no_update
        chosen = list(current or [])
        if trigger == "add-columns":
            if b_values:
                new = [labels.column_index(float(delta), float(Delta), float(b))
                       for b in b_values]
            else:
                new = labels.columns_for_pair(float(delta), float(Delta)).tolist()
        else:
            # One b at this delta, swept across every stored Delta.  This is the
            # "does a second diffusion time help?" control.
            picked = b_values[0] if b_values else float(labels.b_values[2])
            new = labels.columns_for_b(float(picked), delta=float(delta)).tolist()
        chosen.extend(int(column) for column in new)
        return sorted(dict.fromkeys(chosen))

    @app.callback(
        Output("measured-summary", "children"),
        Output("display-columns", "options"),
        Input("measured-columns", "data"),
        Input("pick-delta", "value"), Input("pick-Delta", "value"),
        State("display-columns", "value"),
    )
    def _summarise_columns(measured, delta, Delta, already_displayed):
        measured = measured or []
        pairs = sorted({labels.column_triple(c)[:2] for c in measured})
        summary = (f"{len(measured)} measured columns over {len(pairs)} (delta, Delta) "
                   f"pairs: " + ", ".join(f"({d:g},{D:g})" for d, D in pairs[:8])
                   + (" ..." if len(pairs) > 8 else "")) if measured else \
            "no measured columns yet"
        nearby = ([] if delta is None or Delta is None
                  else labels.columns_for_pair(float(delta), float(Delta)).tolist())
        # Keep whatever is already an axis on the menu, or changing (delta,
        # Delta) would silently drop the current display columns.
        offered = sorted(set(measured) | set(nearby)
                         | {int(c) for c in (already_displayed or [])})
        options = [{"label": labels.column_label(c), "value": int(c)} for c in offered]
        return summary, options

    @app.callback(
        Output("reference-row", "data"), Output("reference-summary", "children"),
        Input("snap-reference", "n_clicks"),
        Input("prediction-plot", "clickData"), Input("rho-V-plot", "clickData"),
        Input("V-kio-plot", "clickData"),
        State("ref-rho", "value"), State("ref-V", "value"), State("ref-kio", "value"),
        State("vi-min", "value"), State("vi-max", "value"),
        State("slice-options", "value"),
    )
    def _set_reference(_clicks, prediction_click, rho_V_click, V_kio_click,
                       rho, V, kio, vi_min, vi_max, options):
        eligible = np.flatnonzero(slicing.candidate_mask(
            labels, float(vi_min), float(vi_max),
            include_free_water="free_water" in (options or [])))
        trigger = callback_context.triggered_id
        row = None
        if trigger in {"prediction-plot", "rho-V-plot", "V-kio-plot"}:
            click = {"prediction-plot": prediction_click, "rho-V-plot": rho_V_click,
                     "V-kio-plot": V_kio_click}[trigger]
            point = (click or {}).get("points", [{}])[0]
            if point.get("customdata") is not None:
                row = int(np.ravel(point["customdata"])[0])
        if row is None:
            if rho is None or V is None or kio is None:
                return no_update, "pick a reference"
            row = data.nearest_entry(float(rho), float(V), float(kio), eligible)
        text = (f"reference row {row}: rho={labels.nominal_rhos[row]:.4g}, "
                f"V={labels.nominal_Vs[row]:.4g}, k_io={labels.kios[row]:.4g}, "
                f"v_i={labels.vis[row]:.3f}")
        return row, text

    @app.callback(
        Output("prediction-plot", "figure"), Output("rho-V-plot", "figure"),
        Output("V-kio-plot", "figure"), Output("widths-plot", "figure"),
        Output("readout", "children"),
        Input("measured-columns", "data"), Input("display-columns", "value"),
        Input("reference-row", "data"), Input("sigma-exponent", "value"),
        Input("reduced-threshold", "value"), Input("s0-mode", "value"),
        Input("slice-options", "value"), Input("vi-min", "value"),
        Input("vi-max", "value"), Input("colour-by", "value"),
        Input("widths-toggle", "value"), Input("reference-mode", "value"),
        Input("ref-paste", "value"),
    )
    def _render(measured, display, reference_row, sigma_exponent, reduced_threshold,
                s0_mode, options, vi_min, vi_max, colour_by, widths_on,
                reference_mode, pasted):
        started = time.time()
        measured = [int(c) for c in (measured or [])]
        display = [int(c) for c in (display or [])][:3]
        options = options or []
        empty = go.Figure()

        if not measured or len(display) < 2:
            return (empty, empty, empty, empty,
                    html.P("Add at least one measured column and pick 2 or 3 "
                           "display columns."))

        eligible = np.flatnonzero(slicing.candidate_mask(
            labels, float(vi_min), float(vi_max),
            include_free_water="free_water" in options))

        measured_block = data.block(measured)[eligible]
        display_block = data.block(display)[eligible]
        variance_block = (data.variance(measured)[eligible]
                          if "variance" in options else None)

        reference = _reference_signal(measured_block, eligible, reference_row,
                                      reference_mode, pasted, len(measured))
        if reference is None:
            return (empty, empty, empty, empty,
                    html.P("Reference not usable: paste exactly one value per "
                           "measured column, or switch back to entry mode."))

        sigma = 10.0 ** float(sigma_exponent)
        threshold = float(reduced_threshold) * len(measured)
        result = slicing.slice_manifold(
            measured_block, reference, labels, eligible, threshold=threshold,
            sigma_measurement=sigma, variance_block=variance_block,
            s0_mode=s0_mode, display_block=display_block,
            display_columns=np.asarray(display),
        )

        marked = reference_row if reference_mode == "entry" else None
        if marked is not None and marked not in set(eligible.tolist()):
            marked = None

        widths = empty
        if "on" in (widths_on or []):
            curve = slicing.widths_versus_measured_count(
                measured_block, reference, labels, eligible,
                sigma_measurement=sigma, variance_block=variance_block,
                s0_mode=s0_mode, reduced_threshold=float(reduced_threshold))
            widths = widths_figure(curve)

        elapsed_ms = 1000.0 * (time.time() - started)
        return (
            prediction_figure(data, display, display_block, eligible,
                              result.survivors, marked, colour_by),
            parameter_figure(data, eligible, result.survivors, marked,
                             "rho", "V", colour_by),
            parameter_figure(data, eligible, result.survivors, marked,
                             "V", "k_io", colour_by),
            widths,
            readout_table(result, len(eligible), len(measured), elapsed_ms),
        )


def _reference_signal(measured_block, eligible, reference_row, mode, pasted,
                      n_measured):
    """The point being sliced around: a library entry's prediction, or pasted."""
    if mode == "paste":
        text = (pasted or "").replace(",", " ").split()
        if len(text) != n_measured:
            return None
        try:
            return np.array([float(value) for value in text])
        except ValueError:
            return None
    if reference_row is None:
        return None
    where = np.flatnonzero(eligible == int(reference_row))
    if where.size == 0:
        return None
    return measured_block[int(where[0])]


def create_app(library: Path, cache_dir: Path | None = None,
               prefer_cache: bool = True) -> tuple[Dash, ExplorerData]:
    data = ExplorerData(library, cache_dir, prefer_cache)
    app = Dash(__name__, title="MADI manifold slicer")
    app.layout = build_layout(data)
    register_callbacks(app, data)
    return app, data


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--library", type=Path, default=DEFAULT_LIBRARY)
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--no-cache", action="store_true",
                        help="read columns from the .npz instead of the cache (slow)")
    parser.add_argument("--port", type=int, default=8050)
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    app, data = create_app(args.library, args.cache_dir, prefer_cache=not args.no_cache)
    url = f"http://127.0.0.1:{args.port}"
    print(f"library : {data.library_path}")
    print(f"reader  : {data.reader.kind} ({data.cache_note})")
    print(f"serving : {url}")
    if not args.no_browser:
        webbrowser.open(url)
    app.run(port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
