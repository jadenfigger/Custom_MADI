"""Dash workspace adapters: jobs, sessions, downloads and point inspection."""

from __future__ import annotations

import html as html_escape
import hashlib
import json
import logging
import uuid

import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
from dash import Input, Output, State, callback_context, dcc, html, no_update

from . import analysis, figures, session
from .runtime import Jobs

LOG = logging.getLogger(__name__)


def _table(records, fields):
    if not records:
        return html.P("No rows to display.")
    return html.Table([
        html.Thead(html.Tr([html.Th(label) for _, label in fields])),
        html.Tbody([html.Tr([html.Td(
            f"{row[key]:.6g}" if isinstance(row[key], float) else str(row[key]))
            for key, _ in fields]) for row in records]),
    ], style={"borderSpacing": "12px 3px", "fontSize": "12px"})


def register_workspace(app, data, render_inputs, render_outputs, render):
    jobs = Jobs()
    app.explorer_jobs = jobs
    library = session.identity(data)
    bindings = [{"id": component, "property": prop, "key": session.key(component)}
                for component, prop in session.BINDINGS]
    app.clientside_callback(
        "function() { return window.manifoldWorkspace.record(" + json.dumps(bindings)
        + ", Array.from(arguments)); }",
        Output("session-current", "data"),
        [Input(component, prop) for component, prop in session.BINDINGS])
    app.clientside_callback(
        "function(undo, redo) { return window.manifoldWorkspace.history(); }",
        Output("history-event", "data"), Input("undo-state", "n_clicks"),
        Input("redo-state", "n_clicks"), prevent_initial_call=True)
    app.clientside_callback(
        "function(value) { return window.manifoldWorkspace.load(value); }",
        Output("restore-event", "data"), Input("session-restore", "data"),
        prevent_initial_call=True)

    def get_args(controls):
        values = []
        for dependency in render_inputs:
            component = dependency.component_id
            if isinstance(component, dict):
                values.append([controls[f"{component['type']}:{index}"] for index in range(3)])
            else:
                values.append(controls[component])
        return values

    @app.callback(Output("analysis-request", "data"),
                  Input("session-current", "data"), Input("recompute", "n_clicks"),
                  State("browser-session", "data"), State("analysis-request", "data"))
    def submit(controls, _clicks, browser, previous):
        if not controls:
            return no_update
        try:
            session.validate_controls(controls, data.labels)
            arguments = get_args(controls)
            signature = hashlib.sha256(json.dumps(arguments, sort_keys=True).encode()).hexdigest()
            if (previous or {}).get("signature") == signature and callback_context.triggered_id != "recompute":
                return no_update

            def work(checkpoint):
                try:
                    result = render(*arguments, checkpoint=checkpoint)
                    result["session"] = session.session_document(controls, library)
                    return result
                except (ValueError, RuntimeError, KeyError) as exc:
                    return {"error": str(exc)}
                except Exception:
                    LOG.exception("Exploration failed")
                    raise

            token = jobs.submit(browser, work)
            return {"token": token, "signature": signature}
        except (ValueError, TypeError, KeyError) as exc:
            # Invalidate any old result even when the new controls are invalid.
            jobs.invalidate(browser)
            return {"token": uuid.uuid4().hex, "error": str(exc)}

    extra_outputs = [Output("analysis-complete", "data"), Output("job-status", "children"),
                     Output("ranking-table", "children"), Output("rank-column", "options"),
                     Output("job-poll", "disabled")]

    @app.callback(render_outputs + extra_outputs,
                  Input("analysis-request", "data"), Input("job-poll", "n_intervals"),
                  State("browser-session", "data"), State("analysis-complete", "data"))
    def poll(request, _ticks, browser, completed):
        unchanged = [no_update] * (len(render_outputs) + len(extra_outputs))
        if not request or request.get("token") == (completed or {}).get("token"):
            return unchanged
        status, result = jobs.poll(browser, request["token"]) if "error" not in request else ("error", request["error"])
        if status in {"queued", "running"}:
            unchanged[len(render_outputs) + 1] = (
                f"{status.capitalize()} — plots show the previous result until this calculation finishes.")
            unchanged[-1] = False
            return unchanged
        error = (result.get("error") if status == "complete" else
                 result if status == "error" else "Result expired. Press Recompute.")
        if error:
            blank = figures.placeholder_figure(str(error))
            outputs = [blank] * 6 + [html.P(str(error)), html.P(str(error)), no_update, "", ""]
            return outputs + [{"token": request["token"], "error": str(error)},
                              str(error), "", [], True]
        ranking = result["ranking"]
        rank_table = (_table(ranking, [("acquisition", "Acquisition"),
                    ("spread_noise", "Spread / noise"), ("std", "Signal std"),
                    ("q05", "5%"), ("median", "Median"), ("q95", "95%")])
                    if ranking else html.P("Enable ranking; it needs at least two survivors and an unmeasured column in the picker pair."))
        options = [{"label": f"{r['acquisition']} — spread/noise {r['spread_noise']:.3g}",
                    "value": r["column_id"]} for r in ranking]
        cache_mb = data.arrays.bytes / 1024 ** 2
        return list(result["outputs"]) + [{"token": request["token"]},
            f"Ready · array cache {cache_mb:.1f} / {data.arrays.max_bytes / 1024 ** 2:g} MiB", rank_table, options, True]

    def current_result(browser, request, completed):
        if not request or request.get("token") != (completed or {}).get("token") or "error" in (completed or {}):
            raise ValueError("Wait for a successful current calculation before inspecting or exporting.")
        status, result = jobs.poll(browser, request["token"])
        if status != "complete" or not isinstance(result, dict) or "analysis" not in result:
            raise ValueError("The result is no longer available. Press Recompute.")
        return result

    @app.callback(Output("session-download", "data"), Output("session-restore", "data"),
                  Output("workspace-note", "children"), Input("save-session", "n_clicks"),
                  Input("load-session", "contents"), Input("load-signal", "contents"),
                  State("session-current", "data"), prevent_initial_call=True)
    def session_io(_clicks, session_contents, signal_contents, controls):
        try:
            trigger = callback_context.triggered_id
            if trigger == "save-session":
                session.validate_controls(controls, data.labels)
                document = session.session_document(controls, library)
                return dcc.send_string(json.dumps(document, indent=2, allow_nan=False),
                                       "manifold-session.json"), no_update, "Session saved."
            if trigger == "load-session":
                restored = session.read_session(session.decode_upload(session_contents), library, data.labels)
                message = "Session restored. Undo returns to the previous workspace."
            else:
                columns, pasted = session.read_signal_csv(session.decode_upload(signal_contents), data.labels)
                restored = dict(controls)
                restored.update({"measured-columns": columns, "ref-paste": pasted,
                                 "reference-mode": "paste", "groups-as-measured": []})
                session.validate_controls(restored, data.labels)
                message = f"Imported {len(columns)} signals in file order; reference mode is pasted values."
            return no_update, {"controls": restored, "nonce": uuid.uuid4().hex}, message
        except (ValueError, TypeError, KeyError) as exc:
            return no_update, no_update, f"Import/export: {exc}"

    @app.callback(Output("result-download", "data"), Output("export-note", "children"),
                  Input("export-result", "n_clicks"), State("export-format", "value"),
                  State("browser-session", "data"), State("analysis-request", "data"),
                  State("analysis-complete", "data"), prevent_initial_call=True)
    def export(_clicks, format, browser, request, completed):
        try:
            result = current_result(browser, request, completed)
            snapshot = result["analysis"]
            if format in {"survivors", "all"}:
                download = dcc.send_string(analysis.export_csv(data, snapshot, format == "survivors"),
                                           f"manifold-{format}.csv")
            elif format == "reference":
                download = dcc.send_string(analysis.export_reference_csv(data, snapshot), "manifold-reference.csv")
            elif format == "npz":
                download = dcc.send_bytes(analysis.export_npz(data, snapshot, result["session"],
                    result["fisher_report"], result["ranking"]), "manifold-analysis.npz")
            elif format == "html":
                metadata = html_escape.escape(json.dumps(result["session"], indent=2, allow_nan=False))
                parts = ["<!doctype html><html><head><meta charset='utf-8'><title>Manifold exploration</title></head><body>",
                         "<h1>Manifold exploration</h1><p>Plots may be sampled; use NPZ or CSV for full numeric data.</p>"]
                plots = [figure for figure in result["outputs"][:6] if len(figure.data)]
                for index, figure in enumerate(plots):
                    parts.append(pio.to_html(figure, full_html=False, include_plotlyjs=index == 0))
                parts.append(f"<details><summary>Session / provenance</summary><pre>{metadata}</pre></details></body></html>")
                download = dcc.send_string("\n".join(parts), "manifold-report.html")
            else:
                raise ValueError("Unsupported export format.")
            return download, "Exported the current completed calculation."
        except (ValueError, RuntimeError) as exc:
            return no_update, str(exc)

    @app.callback(Output("hover-row", "data"),
                  Input("prediction-plot", "hoverData"), Input("rho-V-plot", "hoverData"),
                  Input("V-kio-plot", "hoverData"), Input("fisher-plane-plot", "hoverData"))
    def remember_hover(prediction, rhoV, Vkio, fisher):
        hover = {"prediction-plot": prediction, "rho-V-plot": rhoV,
                 "V-kio-plot": Vkio, "fisher-plane-plot": fisher}.get(callback_context.triggered_id)
        points = (hover or {}).get("points", [])
        if points and points[0].get("customdata") is not None:
            return int(np.ravel(points[0]["customdata"])[0])
        return no_update

    @app.callback(Output("inspector-details", "children"), Output("residual-plot", "figure"),
                  Input("inspect-row", "value"), Input("analysis-complete", "data"),
                  Input("hover-row", "data"),
                  State("browser-session", "data"), State("analysis-request", "data"))
    def inspect_row(explicit_row, completed, hovered_row, browser, request):
        try:
            result = current_result(browser, request, completed)
            snapshot = result["analysis"]
            row = explicit_row if explicit_row is not None else hovered_row
            if row is None:
                row = snapshot.reference_row
            if row is None:
                row = int(snapshot.result.eligible_rows[np.argmin(snapshot.result.chi2)])
            if not isinstance(row, (int, float)) or not np.isfinite(row) or row != int(row):
                raise ValueError("Enter an integer library row.")
            row = int(row)
            detail = analysis.inspect(snapshot, row)
            labels = data.labels
            description = (
                f"Row {row} · {'survivor' if detail['survives'] else 'outside slice'} · "
                f"nominal rho={labels.nominal_rhos[row]:.6g}, V={labels.nominal_Vs[row]:.6g}, "
                f"k_io={labels.kios[row]:.6g}, v_i={labels.vis[row]:.6g}. "
                f"Realised rho={labels.rhos[row]:.6g}, V={labels.Vs[row]:.6g}. "
                f"chi²={detail['chi2']:.6g}, reduced chi²={detail['reduced_chi2']:.6g}, "
                f"raw RMSE={detail['rmse']:.6g}, fitted amplitude={detail['amplitude']:.6g}.")
            coordinate_text = "; ".join(f"{name} = {value:.6g}" for name, value in zip(snapshot.names, detail["display"]))
            if not detail["display_valid"]:
                coordinate_text += " (undefined ADC coordinate: nonpositive signal)"
            graph = go.Figure()
            names = [labels.column_label(c) for c in snapshot.measured]
            graph.add_trace(go.Scatter(x=names, y=detail["standardized"], mode="markers",
                                      name="standardized residual"))
            graph.add_hline(y=0)
            graph.add_hline(y=1, line_dash="dot")
            graph.add_hline(y=-1, line_dash="dot")
            graph.update_layout(title="Residuals by measured acquisition",
                                yaxis_title="(a·signal − reference) / (a·sigma)", height=350,
                                uirevision=str(snapshot.measured))
            worst = np.argsort(-np.nan_to_num(np.abs(detail["standardized"]), nan=np.inf))[:20]
            records = [{"column": names[j], "signal": float(detail["signal"][j]),
                        "reference": float(snapshot.result.reference_signal[j]),
                        "sigma": float(detail["sigma"][j]), "z": float(detail["standardized"][j])}
                       for j in worst]
            return [html.P(description), html.P(coordinate_text), html.P("Largest residuals (up to 20 columns)"),
                    _table(records, [("column", "Acquisition"), ("signal", "Model S/S0"),
                                     ("reference", "Reference"), ("sigma", "Sigma"), ("z", "Residual / sigma")])], graph
        except (ValueError, IndexError, TypeError) as exc:
            return str(exc), figures.placeholder_figure(str(exc))

    @app.callback(Output("measured-columns", "data", allow_duplicate=True),
                  Output("ranking-note", "children"), Input("add-ranked", "n_clicks"),
                  State("rank-column", "value"), State("measured-columns", "data"),
                  State("groups-as-measured", "value"), State("reference-mode", "value"),
                  State("browser-session", "data"), State("analysis-request", "data"),
                  State("analysis-complete", "data"), prevent_initial_call=True)
    def add_ranked(_clicks, column, measured, follow, mode, browser, request, completed):
        try:
            result = current_result(browser, request, completed)
            if "on" in (follow or []):
                raise ValueError("Turn off 'use display groups as measured columns' before adding a ranked column.")
            if mode == "paste":
                raise ValueError("Import a CSV containing the additional measured signal to extend a pasted reference.")
            if column not in {r["column_id"] for r in result["ranking"]}:
                raise ValueError("Choose a column from the current ranking.")
            return list(dict.fromkeys([*(measured or []), column])), "Added acquisition."
        except ValueError as exc:
            return no_update, str(exc)

    @app.callback(Output("reference-summary", "children", allow_duplicate=True),
                  Input("reference-row", "data"), prevent_initial_call=True)
    def reference_summary(row):
        if row is None:
            return "No reference entry selected."
        labels = data.labels
        return (f"reference row {row}: rho={labels.nominal_rhos[row]:.4g}, "
                f"V={labels.nominal_Vs[row]:.4g}, k_io={labels.kios[row]:.4g}, v_i={labels.vis[row]:.3f}")
