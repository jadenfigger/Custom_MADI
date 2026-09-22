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

The slice is the EXACT region.  The Fisher view puts its LOCAL QUADRATIC
approximation on the same axes, so where the two disagree you are seeing the
curvature of the degeneracy that a Cramer-Rao bound cannot express.

Layout
------
Left: controls, in the order you use them -- what was measured, what the plot
axes are, where the reference sits, how the slice is cut, how it is coloured,
the Fisher options, and workspace import/export. Each section folds away.

Right: one summary line that is always visible, a view selector, and four
views (Slice, Fisher / CRLB, Widths, Inspect / next acquisition). Every view stays in the page, so
clicking a point in any of them re-centres every other.
"""

from __future__ import annotations

import argparse
import copy
import time
import uuid
import webbrowser
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from dash import (ALL, MATCH, Dash, Input, Output, State, callback_context, dcc,
                  html, no_update)

from . import axes as axis_groups
from . import fisher as fisher_tools
from . import figures as fig
from . import slice as slicing
from .build_column_cache import DEFAULT_LIBRARY
from .data import ExplorerData
from . import analysis as exploration
from . import session
from .workspace import register_workspace

N_AXIS_SLOTS = 3
CONTROL_STYLE = {"marginBottom": "10px"}
SLOT_STYLE = {"border": "1px solid #e2e2e2", "borderRadius": "4px",
              "padding": "6px", "marginBottom": "8px"}
PANEL_STYLE = {"width": "370px", "padding": "12px", "overflowY": "auto",
               "height": "98vh", "borderRight": "1px solid #ccc",
               "fontFamily": "system-ui, sans-serif", "fontSize": "13px"}
SECTION_STYLE = {"marginBottom": "6px", "borderBottom": "1px solid #eee",
                 "paddingBottom": "6px"}
SUMMARY_STYLE = {"fontWeight": "700", "cursor": "pointer", "padding": "4px 0"}
VIEWS = (("slice", "Slice"), ("fisher", "Fisher / CRLB"), ("widths", "Widths"),
         ("inspect", "Inspect / next acquisition"))


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------

def _section(number: int, title: str, children, open_by_default: bool = True):
    """One foldable control group, numbered in the order you use it."""
    return html.Details([
        html.Summary(f"{number}. {title}", style=SUMMARY_STYLE),
        html.Div(children, style={"paddingLeft": "2px"}),
    ], open=open_by_default, style=SECTION_STYLE)


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

    # Open on a working slice rather than an empty page.
    seed_columns = labels.columns_for_pair(default_delta, default_Delta,
                                           b_max=6000.0).tolist()
    seed_slots = [
        {"delta": 4.0, "Delta": Delta, "b": [1000.0, 2000.0, 3000.0, 4000.0]}
        for Delta in (20.0, 30.0, 40.0)
    ]
    for slot in seed_slots:
        if not np.any(np.isclose(labels.Deltas_for_delta(slot["delta"]), slot["Delta"])):
            slot["delta"], slot["Delta"] = default_delta, default_Delta
        slot["b"] = [b for b in slot["b"] if b in set(labels.b_values)]

    coverage = data.stencil_coverage[1]

    def labelled(text, control):
        return html.Div([html.Label(text, style={"fontWeight": "600"}), control],
                        style=CONTROL_STYLE)

    def axis_slot(index: int) -> html.Div:
        """One plot axis: a (delta, Delta) plus b-values, or arbitrary columns."""
        seed = seed_slots[index]
        slot_Deltas = labels.Deltas_for_delta(seed["delta"])
        return html.Div([
            html.Div(f"axis {index + 1}", style={"fontWeight": "600",
                                                 "fontSize": "12px"}),
            dcc.RadioItems(
                id={"type": "group-mode", "index": index},
                options=[{"label": " (delta, Delta) + b", "value": "pair"},
                         {"label": " any columns", "value": "any"},
                         {"label": " off", "value": "off"}],
                value="pair", inline=True, style={"fontSize": "11px"}),
            html.Div([
                dcc.Dropdown(id={"type": "group-delta", "index": index},
                             options=deltas, value=seed["delta"], clearable=False,
                             style={"fontSize": "11px"}),
                dcc.Dropdown(id={"type": "group-Delta", "index": index},
                             options=[{"label": f"{D:g}", "value": float(D)}
                                      for D in slot_Deltas],
                             value=seed["Delta"], clearable=False,
                             style={"fontSize": "11px"}),
                dcc.Dropdown(id={"type": "group-b", "index": index},
                             options=b_options, value=seed["b"], multi=True,
                             placeholder="b-values (empty = all stored b)",
                             style={"fontSize": "11px"}),
            ], id={"type": "group-pair-box", "index": index}),
            html.Div([
                dcc.Dropdown(id={"type": "group-any", "index": index},
                             options=[], value=[], multi=True,
                             placeholder="columns (Mean only)",
                             style={"fontSize": "11px"}),
            ], id={"type": "group-any-box", "index": index},
                style={"display": "none"}),
        ], style=SLOT_STYLE)

    controls = html.Div([
        html.Div("MADI manifold slicer",
                 style={"fontWeight": "700", "fontSize": "15px"}),
        html.Div(f"{labels.n_entries} entries x {labels.n_columns} columns  |  "
                 f"reader: {data.reader.kind}"
                 + ("" if data.cache_ok else f"  ({data.cache_note})"),
                 style={"color": "#666", "fontSize": "11px",
                        "marginBottom": "8px"}),

        _section(1, "Measured columns  (what the slice uses)", [
            labelled("delta (ms)", dcc.Dropdown(id="pick-delta", options=deltas,
                                                value=default_delta,
                                                clearable=False)),
            labelled("Delta (ms)", dcc.Dropdown(id="pick-Delta",
                                                options=Delta_options,
                                                value=default_Delta,
                                                clearable=False)),
            labelled("b-values (empty = all)",
                     dcc.Dropdown(id="pick-b", options=b_options, multi=True,
                                  value=[])),
            html.Div([
                html.Button("add these", id="add-columns", n_clicks=0),
                html.Button("add one b across all Delta", id="add-b-sweep",
                            n_clicks=0, style={"marginLeft": "6px"}),
                html.Button("clear", id="clear-columns", n_clicks=0,
                            style={"marginLeft": "6px"}),
            ], style=CONTROL_STYLE),
            html.Div(id="measured-summary",
                     style={"fontSize": "11px", "color": "#444"}),
        ]),

        _section(2, "Display axes  (plot coordinates only)", [
            labelled("collapse each group by", dcc.Dropdown(
                id="collapse-method",
                options=[{"label": text, "value": key}
                         for key, text in axis_groups.COLLAPSE_METHODS.items()],
                value="mean", clearable=False)),
            html.Div(id="collapse-note",
                     style={"fontSize": "11px", "color": "#a33"}),
            html.Div([axis_slot(index) for index in range(N_AXIS_SLOTS)]),
            dcc.Checklist(id="groups-as-measured",
                          options=[{"label": " use display groups as measured "
                                             "columns", "value": "on"}],
                          value=[], style=CONTROL_STYLE),
            html.Div("Collapsing is display only - the slice always uses the "
                     "individual measured columns.",
                     style={"fontSize": "11px", "color": "#666"}),
        ]),

        _section(3, "Reference point", [
            dcc.RadioItems(
                id="reference-mode",
                options=[{"label": " click / parameters", "value": "entry"},
                         {"label": " pasted signal values", "value": "paste"}],
                value="entry", style=CONTROL_STYLE),
            html.Div([
                html.Div([html.Span("rho "),
                          dcc.Input(id="ref-rho", type="number", value=3.0e5,
                                    style={"width": "90px"})]),
                html.Div([html.Span("V   "),
                          dcc.Input(id="ref-V", type="number", value=2.0,
                                    style={"width": "90px"})]),
                html.Div([html.Span("k_io"),
                          dcc.Input(id="ref-kio", type="number", value=20.0,
                                    style={"width": "90px"})]),
                html.Button("snap to nearest entry", id="snap-reference",
                            n_clicks=0, style={"marginTop": "5px"}),
            ], style=CONTROL_STYLE),
            labelled("pasted S/S0, one per measured column",
                     dcc.Textarea(id="ref-paste", value="",
                                  style={"width": "100%", "height": "50px"})),
            html.Div(id="reference-summary",
                     style={"fontSize": "11px", "color": "#444"}),
        ]),

        _section(4, "Slice", [
            labelled("sigma (S/S0 units)", html.Div([
                dcc.Slider(id="sigma-exponent", min=-6, max=0, step=0.1,
                           value=np.log10(0.02),
                           marks={-6: "1e-6", -4: "1e-4", -2: "1e-2", 0: "1"}),
                html.Div(id="sigma-readout", style={"fontSize": "11px"}),
            ])),
            labelled("reduced chi2 threshold (chi2 / n_measured)",
                     dcc.Slider(id="reduced-threshold", min=0.0, max=10.0,
                                step=0.1, value=1.0,
                                marks={0: "0", 1: "1", 5: "5", 10: "10"})),
            labelled("S0 handling", dcc.RadioItems(
                id="s0-mode",
                options=[{"label": " fixed (data already S/S0)", "value": "fixed"},
                         {"label": " marginalised (free amplitude)",
                          "value": "free"}],
                value="fixed")),
            dcc.Checklist(
                id="slice-options",
                options=[{"label": " add library MC variance in quadrature",
                          "value": "variance"},
                         {"label": " include the free-water atom",
                          "value": "free_water"}],
                value=["variance"], style=CONTROL_STYLE),
            labelled("v_i band", html.Div([
                dcc.Input(id="vi-min", type="number", value=0.40, step=0.01,
                          style={"width": "70px"}),
                dcc.Input(id="vi-max", type="number", value=0.99, step=0.01,
                          style={"width": "70px", "marginLeft": "6px"}),
            ])),
            labelled("maximum rho (empty = no cap)",
                     dcc.Input(id="rho-max", type="number", min=0, value=None, debounce=True)),
        ]),

        _section(5, "Colour", [
            dcc.Dropdown(id="colour-by", value="k_io", clearable=False,
                         options=[{"label": text, "value": key}
                                  for text, key in
                                  [(t, k) for k, t in
                                   {**fig.PARAMETER_CHOICES, **fig.REFERENCE_HUES,
                                    **fig.FISHER_HUES}.items()]],
                         style=CONTROL_STYLE),
            dcc.RadioItems(id="colour-scale",
                           options=[{"label": " linear", "value": "linear"},
                                    {"label": " log", "value": "log"}],
                           value="linear", inline=True),
            html.Div(id="colour-note",
                     style={"fontSize": "11px", "color": "#666"}),
        ]),

        _section(6, "Fisher / CRLB", [
            labelled("stencil half-width", dcc.RadioItems(
                id="fisher-width",
                options=[{"label": " k=1 (pre-registered)", "value": "1"},
                         {"label": " k=2", "value": "2"},
                         {"label": " Richardson (4J1-J2)/3", "value": "r"}],
                value="1")),
            labelled("ellipse", dcc.RadioItems(
                id="ellipse-mode",
                options=[{"label": " k_io profiled out (marginal)",
                          "value": "profiled"},
                         {"label": " k_io known (conditional)",
                          "value": "conditional"}],
                value="profiled")),
            dcc.Checklist(
                id="fisher-options",
                options=[{"label": " Monte-Carlo debias of the derivatives",
                          "value": "debias"},
                         {"label": " draw sloppy / stiff directions",
                          "value": "axes"},
                         {"label": " overlay the ellipse on the slice view",
                          "value": "overlay"}],
                value=["axes"], style=CONTROL_STYLE),
            dcc.Checklist(id="widths-toggle",
                          options=[{"label": " compute widths vs column count",
                                    "value": "on"}],
                          value=[], style=CONTROL_STYLE),
            html.Div(f"Central stencils exist at {coverage['complete']} of "
                     f"{coverage['total']} entries ({coverage['fraction']:.1%}) "
                     f"at k=1. Entries without one report no Fisher matrix.",
                     style={"fontSize": "11px", "color": "#666"}),
            html.Div(id="fisher-note",
                     style={"fontSize": "11px", "color": "#a33"}),
        ]),
        _section(7, "Workspace / export", [
            html.Div([
                html.Button("Undo", id="undo-state", n_clicks=0),
                html.Button("Redo", id="redo-state", n_clicks=0),
                html.Button("Save session", id="save-session", n_clicks=0),
            ]),
            html.Div(id="history-note", style={"fontSize": "11px"}),
            dcc.Upload(id="load-session", children=html.Button("Load session JSON"),
                       accept=".json", max_size=session.MAX_UPLOAD_BYTES, multiple=False),
            dcc.Upload(id="load-signal", children=html.Button("Import reference CSV"),
                       accept=".csv", max_size=session.MAX_UPLOAD_BYTES, multiple=False),
            html.Div("CSV: column_id,signal or delta,Delta,b,signal. File order becomes measured order.",
                     style={"fontSize": "11px"}),
            dcc.Dropdown(id="export-format", value="survivors", clearable=False, options=[
                {"label": text, "value": value} for value, text in [
                    ("survivors", "Survivor table (CSV)"), ("all", "All candidates (CSV)"),
                    ("npz", "Signals + transformed axes + session (NumPy NPZ)"),
                    ("reference", "Reference signals (CSV / import template)"),
                    ("html", "Interactive plot report (HTML)")]]),
            html.Button("Export current result", id="export-result", n_clicks=0),
            html.Div(id="workspace-note", role="status"),
            html.Div(id="export-note", role="status"),
            labelled("maximum plotted entries (analysis always uses all)",
                     dcc.Dropdown(id="point-budget", clearable=False, value=5000,
                         options=[{"label": "All entries" if n == 1000000 else str(n), "value": n}
                                  for n in (1000, 5000, 20000, 1000000)])),
            html.Div("Camera buttons save 3600 × 2400 PNGs. Ctrl/⌘+S saves the session; "
                     "Alt+Z / Alt+Shift+Z undo/redo; Alt+1–4 switch views; Alt+R snaps reference.",
                     style={"fontSize": "11px"}),
        ]),
    ], style=PANEL_STYLE)

    def graph(id):
        return dcc.Graph(id=id, config={"displaylogo": False,
            "toImageButtonOptions": {"format": "png", "width": 1800, "height": 1200, "scale": 2}})

    def view(key, children):
        return html.Div(children, id=f"view-{key}",
                        style={} if key == "slice" else {"display": "none"})

    plots = html.Div([
        html.Div(id="job-status", role="status", style={"padding": "4px 12px"}),
        html.Button("Recompute", id="recompute", n_clicks=0),
        html.Div(id="readout", style={"padding": "4px 12px",
                                      "borderBottom": "1px solid #eee"}),
        dcc.RadioItems(id="view", options=[{"label": f"  {name}  ", "value": key}
                                           for key, name in VIEWS],
                       value="slice", inline=True,
                       style={"padding": "6px 12px", "fontWeight": "600"}),
        view("slice", [
            graph(id="prediction-plot"),
            html.Div([
                html.Div(graph(id="rho-V-plot"), style={"flex": "1"}),
                html.Div(graph(id="V-kio-plot"), style={"flex": "1"}),
            ], style={"display": "flex"}),
        ]),
        view("fisher", [
            html.Div(id="fisher-panel"),
            graph(id="fisher-plane-plot"),
            graph(id="crlb-curve-plot"),
        ]),
        view("widths", [graph(id="widths-plot")]),
        view("inspect", [
            html.P("Hover a point in any scatter plot to inspect it without moving the reference. "
                   "Clicks still re-centre the slice. Enter a library row here for a fixed inspection."),
            dcc.Input(id="inspect-row", type="number", min=0, max=labels.n_entries - 1,
                      step=1, value=None, debounce=True, placeholder="row id (empty follows hover)"),
            html.Div(id="inspector-details"),
            graph(id="residual-plot"),
            dcc.Checklist(id="rank-options", value=[], options=[{
                "label": " Rank unmeasured columns at the measured-column picker's (delta, Delta)",
                "value": "on"}]),
            html.P("Ranking = standard deviation of survivor predictions / noise. "
                   "All survivors have equal weight. This describes raw S/S0 disagreement; "
                   "it is not expected information gain or an amplitude-marginal design score."),
            html.Div(id="ranking-table"),
            dcc.Dropdown(id="rank-column", options=[], placeholder="Choose a ranked acquisition"),
            html.Button("Add ranked column to measurements", id="add-ranked", n_clicks=0),
            html.Div(id="ranking-note"),
        ]),
    ], style={"flex": "1", "height": "98vh", "overflowY": "auto",
              "fontFamily": "system-ui, sans-serif"})

    return html.Div([
        dcc.Store(id="browser-session", data=uuid.uuid4().hex),
        dcc.Store(id="analysis-request"), dcc.Store(id="analysis-complete"),
        dcc.Store(id="session-current"), dcc.Store(id="session-restore"),
        dcc.Store(id="history-event"), dcc.Store(id="restore-event"), dcc.Store(id="hover-row"),
        dcc.Interval(id="job-poll", interval=700, disabled=True),
        dcc.Download(id="session-download"), dcc.Download(id="result-download"),
        dcc.Store(id="measured-columns", data=seed_columns),
        dcc.Store(id="reference-row", data=None),
        controls, plots,
    ], style={"display": "flex"})


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------

def build_groups(labels, modes, deltas, Deltas, b_lists, any_lists):
    """Turn the three axis slots into AxisGroup objects, skipping empty ones."""
    groups = []
    for index in range(len(modes or [])):
        mode = modes[index]
        if mode == "off":
            continue
        if mode == "any":
            columns = any_lists[index] or []
            if columns:
                groups.append(axis_groups.any_group(columns))
            continue
        delta, Delta = deltas[index], Deltas[index]
        if delta is None or Delta is None:
            continue
        groups.append(axis_groups.pair_group(labels, float(delta), float(Delta),
                                             b_lists[index] or []))
    return groups


def register_callbacks(app: Dash, data: ExplorerData) -> None:
    labels = data.labels

    @app.callback(
        Output({"type": "group-Delta", "index": MATCH}, "options"),
        Output({"type": "group-Delta", "index": MATCH}, "value"),
        Input({"type": "group-delta", "index": MATCH}, "value"),
        State({"type": "group-Delta", "index": MATCH}, "value"),
    )
    def _group_delta_options(delta, current):
        available = labels.Deltas_for_delta(float(delta))
        options = [{"label": f"{D:g}", "value": float(D)} for D in available]
        keep = current if current is not None and float(current) in set(available) \
            else float(available[-1])
        return options, keep

    @app.callback(
        Output({"type": "group-pair-box", "index": MATCH}, "style"),
        Output({"type": "group-any-box", "index": MATCH}, "style"),
        Input({"type": "group-mode", "index": MATCH}, "value"),
    )
    def _group_mode_boxes(mode):
        hidden, shown = {"display": "none"}, {}
        if mode == "any":
            return hidden, shown
        if mode == "off":
            return hidden, hidden
        return shown, hidden

    @app.callback(Output("pick-Delta", "options"), Output("pick-Delta", "value"),
                  Input("pick-delta", "value"), State("pick-Delta", "value"))
    def _delta_options(delta, current):
        """Delta >= delta, so the offered Deltas change with delta."""
        available = labels.Deltas_for_delta(float(delta))
        options = [{"label": f"{D:g}", "value": float(D)} for D in available]
        keep = current if current is not None and float(current) in set(available) \
            else float(available[-1])
        return options, keep

    @app.callback(Output("sigma-readout", "children"),
                  Input("sigma-exponent", "value"))
    def _sigma_readout(exponent):
        return f"sigma = {10.0 ** float(exponent):.4g}"

    @app.callback(
        [Output(f"view-{key}", "style") for key, _ in VIEWS],
        Input("view", "value"),
    )
    def _switch_view(chosen):
        """Every view stays in the page; only its visibility changes.

        Keeping the graphs mounted means one render callback can update them
        all, and a click in a hidden view still carries its selection.
        """
        return [{} if key == chosen else {"display": "none"} for key, _ in VIEWS]

    @app.callback(
        Output("measured-columns", "data"),
        Input("add-columns", "n_clicks"), Input("add-b-sweep", "n_clicks"),
        Input("clear-columns", "n_clicks"),
        Input("groups-as-measured", "value"),
        Input({"type": "group-mode", "index": ALL}, "value"),
        Input({"type": "group-delta", "index": ALL}, "value"),
        Input({"type": "group-Delta", "index": ALL}, "value"),
        Input({"type": "group-b", "index": ALL}, "value"),
        Input({"type": "group-any", "index": ALL}, "value"),
        State("pick-delta", "value"), State("pick-Delta", "value"),
        State("pick-b", "value"), State("measured-columns", "data"),
        prevent_initial_call=True,
    )
    def _edit_columns(add, sweep, clear, follow_groups, modes, group_deltas,
                      group_Deltas, group_bs, group_anys, delta, Delta,
                      b_values, current):
        """Maintain the measured column set.

        While "use display groups as measured columns" is on, the groups are
        the single source of truth and the buttons stand down -- otherwise the
        two would fight over the same store.
        """
        trigger = callback_context.triggered_id
        if "on" in (follow_groups or []):
            groups = build_groups(labels, modes, group_deltas, group_Deltas,
                                  group_bs, group_anys)
            return sorted({int(column) for group in groups
                           for column in group.columns})
        if isinstance(trigger, dict):
            return no_update          # a group changed but the toggle is off
        if trigger == "groups-as-measured":
            return no_update          # just switched off; keep what is there
        if trigger == "clear-columns":
            return []
        if delta is None or Delta is None:
            return no_update
        chosen = list(current or [])
        if trigger == "add-columns":
            if b_values:
                new_columns = [labels.column_index(float(delta), float(Delta),
                                                   float(b)) for b in b_values]
            else:
                new_columns = labels.columns_for_pair(float(delta),
                                                      float(Delta)).tolist()
        else:
            # One b at this delta, swept across every stored Delta.  This is the
            # "does a second diffusion time help?" control.
            picked = b_values[0] if b_values else float(labels.b_values[2])
            new_columns = labels.columns_for_b(float(picked),
                                               delta=float(delta)).tolist()
        chosen.extend(int(column) for column in new_columns)
        return list(dict.fromkeys(chosen))

    @app.callback(
        Output("measured-summary", "children"),
        Output({"type": "group-any", "index": ALL}, "options"),
        Input("measured-columns", "data"),
        Input("pick-delta", "value"), Input("pick-Delta", "value"),
        Input("groups-as-measured", "value"),
        State({"type": "group-any", "index": ALL}, "value"),
    )
    def _summarise_columns(measured, delta, Delta, follow_groups, any_values):
        measured = measured or []
        pairs = sorted({labels.column_triple(c)[:2] for c in measured})
        if measured:
            summary = (f"{len(measured)} measured columns over {len(pairs)} "
                       f"(delta, Delta) pairs: "
                       + ", ".join(f"({d:g},{D:g})" for d, D in pairs[:8])
                       + (" ..." if len(pairs) > 8 else ""))
        else:
            summary = "no measured columns yet"
        if "on" in (follow_groups or []):
            summary += "  [following the display groups; buttons disabled]"

        nearby = ([] if delta is None or Delta is None
                  else labels.columns_for_pair(float(delta), float(Delta)).tolist())
        already = {int(c) for values in (any_values or []) for c in (values or [])}
        offered = sorted(set(measured) | set(nearby) | already)
        options = [{"label": labels.column_label(c), "value": int(c)}
                   for c in offered]
        return summary, [options] * len(any_values or [])

    @app.callback(
        Output("collapse-note", "children"),
        Output("collapse-method", "options"),
        Input({"type": "group-mode", "index": ALL}, "value"),
        Input({"type": "group-delta", "index": ALL}, "value"),
        Input({"type": "group-Delta", "index": ALL}, "value"),
        Input({"type": "group-b", "index": ALL}, "value"),
        Input({"type": "group-any", "index": ALL}, "value"),
    )
    def _collapse_availability(modes, group_deltas, group_Deltas, group_bs,
                               group_anys):
        """ADC needs every axis to be one (delta, Delta) with >= 2 b-values."""
        groups = build_groups(labels, modes, group_deltas, group_Deltas,
                              group_bs, group_anys)
        refusals = [group.adc_refusal() for group in groups
                    if not group.supports_adc]
        options = [{"label": text, "value": key}
                   for key, text in axis_groups.COLLAPSE_METHODS.items()]
        if refusals:
            options[1]["disabled"] = True
            return f"ADC unavailable: {refusals[0]}.", options
        return "", options

    @app.callback(
        Output("reference-row", "data"), Output("reference-summary", "children"),
        Input("snap-reference", "n_clicks"),
        Input("prediction-plot", "clickData"), Input("rho-V-plot", "clickData"),
        Input("V-kio-plot", "clickData"), Input("fisher-plane-plot", "clickData"),
        State("ref-rho", "value"), State("ref-V", "value"), State("ref-kio", "value"),
        State("vi-min", "value"), State("vi-max", "value"),
        State("slice-options", "value"),
        State("rho-max", "value"),
    )
    def _set_reference(_clicks, prediction_click, rho_V_click, V_kio_click,
                       fisher_click, rho, V, kio, vi_min, vi_max, options, rho_max):
        try:
            eligible = data.eligible(float(vi_min), float(vi_max),
                                     "free_water" in (options or []), rho_max)
        except (TypeError, ValueError) as exc:
            return no_update, str(exc)
        trigger = callback_context.triggered_id
        clicks = {"prediction-plot": prediction_click, "rho-V-plot": rho_V_click,
                  "V-kio-plot": V_kio_click, "fisher-plane-plot": fisher_click}
        row = None
        if trigger in clicks:
            point = (clicks[trigger] or {}).get("points", [{}])[0]
            if point.get("customdata") is not None:
                row = int(np.ravel(point["customdata"])[0])
        if row is None:
            if rho is None or V is None or kio is None:
                return no_update, "pick a reference"
            try:
                row = data.nearest_entry(float(rho), float(V), float(kio), eligible)
            except ValueError as exc:
                return no_update, str(exc)
        text = (f"reference row {row}: rho={labels.nominal_rhos[row]:.4g}, "
                f"V={labels.nominal_Vs[row]:.4g}, k_io={labels.kios[row]:.4g}, "
                f"v_i={labels.vis[row]:.3f}")
        return row, text

    render_outputs = [
        Output("prediction-plot", "figure"), Output("rho-V-plot", "figure"),
        Output("V-kio-plot", "figure"), Output("widths-plot", "figure"),
        Output("fisher-plane-plot", "figure"), Output("crlb-curve-plot", "figure"),
        Output("fisher-panel", "children"), Output("readout", "children"),
        Output("colour-by", "options"), Output("colour-note", "children"),
        Output("fisher-note", "children"),
    ]
    render_inputs = [
        Input("measured-columns", "data"),
        Input("collapse-method", "value"),
        Input({"type": "group-mode", "index": ALL}, "value"),
        Input({"type": "group-delta", "index": ALL}, "value"),
        Input({"type": "group-Delta", "index": ALL}, "value"),
        Input({"type": "group-b", "index": ALL}, "value"),
        Input({"type": "group-any", "index": ALL}, "value"),
        Input("reference-row", "data"), Input("sigma-exponent", "value"),
        Input("reduced-threshold", "value"), Input("s0-mode", "value"),
        Input("slice-options", "value"), Input("vi-min", "value"),
        Input("vi-max", "value"), Input("colour-by", "value"),
        Input("colour-scale", "value"), Input("widths-toggle", "value"),
        Input("reference-mode", "value"), Input("ref-paste", "value"),
        Input("fisher-width", "value"), Input("ellipse-mode", "value"),
        Input("fisher-options", "value"),
        Input("rho-max", "value"), Input("point-budget", "value"),
        Input("rank-options", "value"), Input("pick-delta", "value"),
        Input("pick-Delta", "value"), Input("view", "value"),
    ]
    def _render(measured, method, modes, group_deltas, group_Deltas, group_bs,
                group_anys, reference_row, sigma_exponent, reduced_threshold,
                s0_mode, options, vi_min, vi_max, colour_by, colour_scale,
                widths_on, reference_mode, pasted, fisher_width, ellipse_mode,
                fisher_options, rho_max, point_budget, rank_options, pick_delta,
                pick_Delta, active_view, checkpoint=lambda: None):
        started = time.perf_counter()
        measured = [int(c) for c in (measured or [])]
        options = options or []
        fisher_options = fisher_options or []

        has_reference = reference_row is not None or reference_mode == "paste"
        hue_options = [{"label": text, "value": key}
                       for key, text in fig.PARAMETER_CHOICES.items()]
        hue_options += [{"label": text, "value": key, "disabled": not has_reference}
                        for key, text in fig.REFERENCE_HUES.items()]
        hue_options += [{"label": text, "value": key}
                        for key, text in fig.FISHER_HUES.items()]
        if colour_by is None or (colour_by in fig.REFERENCE_HUES
                                 and not has_reference):
            colour_by = "k_io"

        widths_hint = ("Tick the widths checkbox in section 6 to draw how the "
                       "parameter spreads shrink as columns are added.")

        def bail(message: str):
            raise ValueError(message)

        groups = build_groups(labels, modes, group_deltas, group_Deltas,
                              group_bs, group_anys)
        if len(groups) < 2:
            return bail("Pick at least 2 display axes; set an axis to 'off' "
                        "only when you want a 2-D plot.")
        groups = groups[:3]
        if method == "adc" and any(not group.supports_adc for group in groups):
            refusal = next(group.adc_refusal() for group in groups
                           if not group.supports_adc)
            return bail(f"ADC unavailable: {refusal}. Switch the collapse "
                        "method back to Mean.")
        if not measured:
            return bail("Add at least one measured column.")

        sigma = 10.0 ** float(sigma_exponent)
        threshold = float(reduced_threshold) * len(measured)
        analysis = exploration.compute(
            data, measured, groups, method, reference_row, reference_mode, pasted,
            sigma, float(reduced_threshold), s0_mode, options, float(vi_min),
            float(vi_max), rho_max, checkpoint)
        result = analysis.result
        eligible = result.eligible_rows
        measured_block, variance_block = analysis.block, analysis.variance
        reference, marked = result.reference_signal, analysis.reference_row
        display_values, valid_all, names = analysis.display, analysis.valid, analysis.names
        checkpoint()

        # --- Fisher at the reference, and for the hue if one is selected ----
        report, fisher_note, truncation = None, "", None
        width_choice = str(fisher_width or "1")
        widths_wanted = (1, 2) if width_choice == "r" else (int(width_choice),)
        node = data.node_grid.node_of_row.get(int(marked)) if marked is not None else None
        debias = "debias" in fisher_options
        if debias and width_choice == "r":
            raise ValueError("Richardson MC debias needs cross-stencil covariance. Choose k=1 or k=2, or disable debias.")
        crn = data.crn_subset(measured) if debias else None

        if node is not None:
            full_block = data.block(measured)
            jacobians = []
            missing = []
            for width in widths_wanted:
                J_w, missing_w = fisher_tools.jacobian(data.node_grid, full_block,
                                                       node, width)
                if J_w is None:
                    missing = missing_w
                    jacobians = []
                    break
                jacobians.append(J_w)
            if jacobians:
                J = (fisher_tools.richardson(*jacobians) if width_choice == "r"
                     else jacobians[0])
                if len(widths_wanted) == 1 and width_choice != "r":
                    other = fisher_tools.jacobian(data.node_grid, full_block, node,
                                                  2 if widths_wanted[0] == 1 else 1)[0]
                    if other is not None:
                        truncation = fisher_tools.truncation_estimate(
                            *((J, other) if widths_wanted[0] == 1 else (other, J)))
                variance_for_J = None
                if debias:
                    variance_for_J = fisher_tools.derivative_variance(
                        data.node_grid, data.variance(measured), node,
                        widths_wanted[0], crn)
                report = fisher_tools.fisher_report(
                    J, full_block[int(marked)], sigma, labels.kios[int(marked)],
                    variance_for_J, s0_mode)
            else:
                fisher_note = ("No central stencil on " + ", ".join(missing)
                               + " at this node (v_i band edge or k_io endpoint).")
        elif marked is None:
            fisher_note = ("The Fisher matrix needs a library entry as the "
                           "reference; pasted signals have no grid node.")

        stencil_note = fisher_note
        debias_note = ""
        if debias:
            debias_note = (f"Monte-Carlo debias on; the CRN covariance term "
                           f"covers {crn.n_covered} of {len(measured)} columns "
                           f"(the rest use the conservative form).")

        # --- the hue ------------------------------------------------------
        fisher_values = None
        if colour_by in fig.FISHER_HUES:
            rows, J_batch = data.batch_jacobian(measured, widths_wanted[0])
            if width_choice == "r":
                rows2, J2 = data.batch_jacobian(measured, 2)
                rows, i1, i2 = np.intersect1d(rows, rows2, return_indices=True)
                J_batch = fisher_tools.richardson(J_batch[i1], J2[i2])
            batch = fisher_tools.batch_quantities(J_batch, rows, sigma, labels.kios)
            key = {"crlb_log_rho": "crlb_log_rho", "crlb_log_vi": "crlb_log_vi",
                   "kappa": "condition_number",
                   "lambda3": "smallest_eigenvalue",
                   "profiled_angle": "profiled_angle_deg"}[colour_by]
            whole = np.full(labels.n_entries, np.nan)
            whole[rows] = batch[key]
            fisher_values = whole[eligible]

        colour = fig.build_colour(
            labels, eligible, colour_by, colour_scale == "log",
            measured_block=measured_block, reference=reference,
            chi2=result.chi2, threshold=threshold, fisher_values=fisher_values)
        colour_note = ("Fisher colours use fixed S0, measurement noise only, and no MC debias. "
                       if colour_by in fig.FISHER_HUES else "")
        if colour_by in fig.REFERENCE_HUES and colour_scale == "log":
            colour_note = (f"log colour floored at {fig.COLOUR_LOG_FLOOR:g}; the "
                           "reference entry itself is exactly 0.")
        if colour.n_undefined:
            colour_note += (f"  {colour.n_undefined} entries have no value for "
                            "this hue and are drawn in pale grey.")

        reference_point, reference_note = _collapsed_reference(
            data, groups, method, measured, reference, eligible, marked)

        excluded = int((~valid_all).sum())
        if excluded:
            print(f"[manifold_explorer] {excluded} entries excluded from the "
                  f"prediction plot (S <= 0 inside an ADC group); library rows: "
                  f"{eligible[~valid_all].tolist()}", flush=True)

        # --- the three views ----------------------------------------------
        widths = fig.placeholder_figure(widths_hint)
        if "on" in (widths_on or []) and active_view == "widths":
            curve = slicing.widths_versus_measured_count(
                measured_block, reference, labels, eligible,
                sigma_measurement=sigma, variance_block=variance_block,
                s0_mode=s0_mode, reduced_threshold=float(reduced_threshold),
                checkpoint=checkpoint)
            predicted = None
            if report is not None:
                predicted = fisher_tools.crlb_versus_measured_count(
                    J, data.block(measured)[int(marked)], sigma,
                    labels.kios[int(marked)], variance=variance_for_J, s0_mode=s0_mode,
                    reduced_threshold=float(reduced_threshold))
            widths = fig.widths_figure(curve, predicted)

        checkpoint()
        if report is not None and active_view == "fisher":
            crlb_curve = fig.crlb_curve_figure(
                fisher_tools.crlb_versus_measured_count(
                    J, data.block(measured)[int(marked)], sigma,
                    labels.kios[int(marked)], variance=variance_for_J, s0_mode=s0_mode,
                    reduced_threshold=float(reduced_threshold)))
        else:
            crlb_curve = fig.placeholder_figure(
                fisher_note or "Open Fisher / CRLB to compute the column-count curve.")

        overlay = report if "overlay" in fisher_options else None
        notes = [note for note in (reference_note, colour_note) if note]
        checkpoint()
        ranking = (exploration.rank_acquisitions(
            data, analysis, labels.columns_for_pair(float(pick_delta), float(pick_Delta)), checkpoint)
            if "on" in (rank_options or []) else [])
        indices = exploration.sample_indices(eligible, result.survivors, int(point_budget), marked)
        plotted = eligible[indices]
        survivors = result.survivors[indices]
        # Preserve the full-library colour range while sampling only the geometry.
        sampled_colour = copy.copy(colour)
        sampled_colour.raw = colour.raw[indices]
        sampled_colour.values = colour.values[indices]
        sampled_colour.defined = colour.defined[indices]
        if len(plotted) < len(eligible):
            notes.append(f"Plot preview: {len(plotted)} / {len(eligible)} entries. Counts, widths, ranking and exports use all entries.")
        outputs = (
            fig.prediction_figure(data, names, display_values[indices], valid_all[indices], plotted,
                                  survivors, reference_point, sampled_colour, method),
            fig.parameter_figure(data, plotted, survivors, marked,
                                 "rho", "V", sampled_colour, ellipse=overlay,
                                 ellipse_threshold=threshold,
                                 ellipse_mode=ellipse_mode),
            fig.parameter_figure(data, plotted, survivors, marked,
                                 "V", "k_io", sampled_colour),
            widths,
            fig.fisher_plane_figure(labels, plotted, survivors, marked,
                                    sampled_colour, report, threshold, ellipse_mode,
                                    "axes" in fisher_options),
            crlb_curve,
            fig.fisher_panel(report, node, labels, marked, stencil_note,
                             truncation, debias_note),
            None,  # Filled after all figures are built so the timing covers rendering.
            hue_options,
            " ".join(notes),
            fisher_note,
        )
        for index, figure in enumerate(outputs[:6]):
            figure.update_layout(uirevision=(f"{method}:{names}" if index == 0 else "parameters"))
        elapsed_ms = 1000.0 * (time.perf_counter() - started)
        outputs = (*outputs[:7], fig.readout_table(result, len(eligible), len(measured),
                                                 elapsed_ms, excluded), *outputs[8:])
        return {"outputs": outputs, "analysis": analysis, "ranking": ranking,
                "fisher_report": report}

    register_workspace(app, data, render_inputs, render_outputs, _render)



def _collapsed_reference(data, groups, method, measured, reference, eligible,
                         reference_row):
    """The reference point in collapsed axis coordinates, or None with a reason.

    A reference that is a library entry has a prediction at every column, so it
    always collapses.  A pasted reference only supplies the measured columns,
    so a group reaching outside them cannot be collapsed and no marker is drawn.
    """
    if reference_row is not None:
        point = []
        for group in groups:
            block = data.block(group.columns)[int(reference_row)][None, :]
            values, valid = axis_groups.collapse(block, group, method)
            if not bool(valid[0]):
                return None, ("No reference marker: the reference entry has "
                              "S <= 0 inside an ADC group.")
            point.append(float(values[0]))
        return np.asarray(point), ""

    by_column = {int(column): float(value)
                 for column, value in zip(measured, reference)}
    point = []
    for group in groups:
        value = axis_groups.collapse_reference(by_column, group, method)
        if value is None:
            return None, ("No reference marker: the pasted values do not cover "
                          f"every column of '{group.label(method)}'.")
        point.append(value)
    return np.asarray(point), ""


def _reference_signal(measured_block, eligible, reference_row, mode, pasted,
                      n_measured):
    """The point being sliced around: a library entry's prediction, or pasted."""
    if mode == "paste":
        text = (pasted or "").replace(",", " ").split()
        if len(text) != n_measured:
            return None
        try:
            return session.parse_signal(pasted, n_measured)
        except ValueError:
            return None
    if reference_row is None:
        return None
    where = np.flatnonzero(eligible == int(reference_row))
    if where.size == 0:
        return None
    return measured_block[int(where[0])]


def create_app(library: Path, cache_dir: Path | None = None,
               prefer_cache: bool = True, cache_mb: int = 256) -> tuple[Dash, ExplorerData]:
    data = ExplorerData(library, cache_dir, prefer_cache, cache_mb)
    app = Dash(__name__, title="MADI manifold slicer",
               assets_folder=str(Path(__file__).parent / "assets"))
    app.layout = lambda: build_layout(data)
    register_callbacks(app, data)
    return app, data


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--library", type=Path, default=DEFAULT_LIBRARY)
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--no-cache", action="store_true",
                        help="read columns from the .npz instead of the cache (slow)")
    parser.add_argument("--memory-cache-mb", type=int, default=256, help="array cache budget in MiB")
    parser.add_argument("--port", type=int, default=8050)
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    if args.memory_cache_mb < 0:
        parser.error("--memory-cache-mb must be nonnegative")
    app, data = create_app(args.library, args.cache_dir, prefer_cache=not args.no_cache,
                           cache_mb=args.memory_cache_mb)
    url = f"http://127.0.0.1:{args.port}"
    print(f"library : {data.library_path}")
    print(f"reader  : {data.reader.kind} ({data.cache_note})")
    print(f"serving : {url}")
    if not args.no_browser:
        webbrowser.open(url)
    try:
        app.run(port=args.port, debug=args.debug)
    finally:
        app.explorer_jobs.close()
        data.reader.close()


if __name__ == "__main__":
    main()
