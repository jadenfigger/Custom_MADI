"""Every Plotly figure the explorer draws, and the colour scale they share.

Split out of ``app.py`` so that file holds only layout and callbacks.  Nothing
here touches the library: the callbacks hand these functions arrays that are
already in memory.
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from dash import html

from . import axes as axis_groups
from . import fisher as fisher_tools
from . import slice as slicing

# A log colour scale cannot show the reference entry itself, whose RMSE and chi
# are exactly 0.  Values are floored here before taking the logarithm, and the
# colorbar says so.
COLOUR_LOG_FLOOR = 1e-6

# Survivor counts share a log axis on the widths plot, where 0 has no place.
# An empty slice is drawn at this floor with the true count in its hover text
# rather than being dropped.
WIDTHS_COUNT_FLOOR = 0.5

# The free-water atom carries rho = V = 0, which cannot sit on a log axis.  It
# is drawn in its own band this many decades below the smallest positive grid
# value instead of being silently dropped.
FREE_WATER_BAND_DECADES = 1.0

PARAMETER_CHOICES = {
    "rho": "rho (1/uL)",
    "V": "V (pL)",
    "k_io": "k_io (1/s)",
    "vi": "rho*V (v_i)",
}
# Hues measured from the reference point rather than read off the entry's
# labels.  They mean nothing until a reference exists, so the UI disables them.
REFERENCE_HUES = {
    "rmse": "RMSE to reference (S/S0)",
    "chi": "chi distance to reference",
}
# Hues that need a Fisher matrix at every node, so they are undefined wherever
# a central stencil is missing.
FISHER_HUES = {
    "crlb_log_rho": "CRLB on log rho",
    "crlb_log_vi": "CRLB on log v_i",
    "kappa": "condition number of D F D",
    "lambda3": "smallest eigenvalue of D F D",
    "profiled_angle": "sloppy angle from constant-v_i (deg)",
}


def _row_ids(rows: np.ndarray) -> list[int]:
    """Library row indices as a plain list, for Plotly `customdata`.

    Plotly 6 base64-encodes numpy arrays into {dtype, bdata}, and plotly.js
    does not expand that back into per-point customdata, so a numpy array here
    silently produces click events with no customdata and clicking a point
    does nothing.  A plain list round-trips correctly.
    """
    return [int(row) for row in np.asarray(rows).ravel()]


class Colour:
    """One hue, evaluated once per eligible entry and shared by every plot.

    Parameter hues are read from the entry's labels.  Reference hues (RMSE and
    chi) are distances from the reference point, so they are computed over the
    measured columns and are undefined until a reference exists.  Both kinds
    end up as one array indexed the same way as ``eligible``, which keeps the
    plotting code from caring which is which.
    """

    def __init__(self, values: np.ndarray, title: str, log_scale: bool = False,
                 threshold: float | None = None, paint_background: bool = False,
                 undefined_note: str = ""):
        self.raw = np.asarray(values, dtype=float)
        # Fisher hues are NaN wherever a node has no central stencil.  Those
        # entries are real and still plotted, in their own grey trace, rather
        # than dropped -- a missing point must never look like a missing entry.
        self.defined = np.isfinite(self.raw)
        self.undefined_note = undefined_note
        self.log_scale = bool(log_scale)
        # A distance-from-reference hue is a field over the whole library, so
        # the faint background entries are coloured too and every plot shares
        # one scale.  Then the slice edge falls inside the colorbar instead of
        # sitting at its very top, where it says nothing.
        self.paint_background = bool(paint_background)
        if self.log_scale:
            self.values = np.log10(np.maximum(self.raw, COLOUR_LOG_FLOOR))
            self.title = f"log10 {title}<br>(floor {COLOUR_LOG_FLOOR:g})"
        else:
            self.values = self.raw
            self.title = title
        self.threshold = threshold
        finite = self.values[np.isfinite(self.values)]
        self.cmin = float(finite.min()) if finite.size else 0.0
        self.cmax = float(finite.max()) if finite.size else 1.0

    @property
    def n_undefined(self) -> int:
        return int((~self.defined).sum())

    def colorbar(self) -> dict:
        """Colorbar spec, with the slice edge marked when there is one."""
        bar = {"title": self.title, "thickness": 12}
        if self.threshold is None or not np.isfinite(self.threshold):
            return bar
        edge = (np.log10(max(self.threshold, COLOUR_LOG_FLOOR))
                if self.log_scale else self.threshold)
        if not (self.cmin <= edge <= self.cmax):
            return bar
        ticks = list(np.linspace(self.cmin, self.cmax, 5))
        labels = [f"{value:.3g}" for value in ticks]
        ticks.append(float(edge))
        labels.append("slice edge")
        order = np.argsort(ticks)
        bar["tickvals"] = [float(ticks[i]) for i in order]
        bar["ticktext"] = [labels[i] for i in order]
        return bar

    def subset(self, mask: np.ndarray) -> np.ndarray:
        return self.values[mask]

    def marker(self, values, show_scale: bool) -> dict:
        """Marker colour options that keep every trace on one shared scale."""
        return dict(color=values, colorscale="Viridis", cmin=self.cmin,
                    cmax=self.cmax, showscale=show_scale,
                    colorbar=self.colorbar() if show_scale else None)


def build_colour(labels, eligible: np.ndarray, choice: str, log_scale: bool,
                 measured_block: np.ndarray | None = None,
                 reference: np.ndarray | None = None,
                 chi2: np.ndarray | None = None,
                 threshold: float | None = None,
                 fisher_values: np.ndarray | None = None) -> Colour:
    """The hue array for one dropdown choice, over the eligible entries."""
    if choice == "rho":
        return Colour(np.log10(np.maximum(labels.nominal_rhos[eligible], 1e-12)),
                      "log10 rho")
    if choice == "V":
        return Colour(np.log10(np.maximum(labels.nominal_Vs[eligible], 1e-12)),
                      "log10 V")
    if choice == "k_io":
        return Colour(labels.kios[eligible], "k_io")
    if choice == "vi":
        return Colour(labels.vis[eligible], "v_i = rho*V")
    if choice == "rmse":
        return Colour(slicing.rmse_to_reference(measured_block, reference),
                      "RMSE (S/S0)", log_scale=log_scale, paint_background=True)
    if choice == "chi":
        # chi = sqrt(chi2) with the slice's own sigma, variance term and S0
        # convention, so chi <= sqrt(threshold) is exactly the surviving set.
        edge = None if threshold is None else float(np.sqrt(max(threshold, 0.0)))
        return Colour(np.sqrt(np.maximum(chi2, 0.0)), "chi = sqrt(chi2)",
                      log_scale=log_scale, threshold=edge, paint_background=True)
    if choice in FISHER_HUES:
        if fisher_values is None:
            raise ValueError(f"{choice} needs Fisher quantities")
        note = ("no central stencil (band edge or k_io endpoint)"
                if not np.all(np.isfinite(fisher_values)) else "")
        return Colour(fisher_values, FISHER_HUES[choice], log_scale=log_scale,
                      paint_background=True, undefined_note=note)
    raise ValueError(f"unknown colour choice {choice!r}")


def _hover_text(labels, rows: np.ndarray) -> list[str]:
    return [
        f"row {row}<br>rho={labels.nominal_rhos[row]:.3g}"
        f"<br>V={labels.nominal_Vs[row]:.3g}"
        f"<br>k_io={labels.kios[row]:.3g}"
        f"<br>v_i={labels.vis[row]:.3f}"
        for row in rows
    ]


def prediction_figure(data, names, values, valid, eligible,
                      survivors, reference_point, colour: "Colour", method: str):
    """Scatter of every entry on 2 or 3 collapsed group axes.

    ``values`` is (n_eligible, n_axes) already collapsed -- a mean S/S0 or an
    ADC per group -- and ``valid`` marks the entries every axis could be
    computed for.  Invalid entries (an ADC group containing S <= 0) are left
    out of the plot rather than clipped.
    """
    labels = data.labels
    n_axes = values.shape[1]
    keep = np.asarray(valid, dtype=bool)
    shown = values[keep]
    shown_rows = eligible[keep]
    shown_survivors = np.asarray(survivors, dtype=bool)[keep]
    hue = colour.values[keep]

    figure = go.Figure()
    scatter = go.Scatter3d if n_axes >= 3 else go.Scattergl
    faint = (dict(size=2.2, color="#8f8f8f", opacity=0.55) if n_axes >= 3
             else dict(size=3, color="#cccccc"))
    bright = dict(size=3.4) if n_axes >= 3 else dict(size=6)
    coords = lambda block: ({"x": block[:, 0], "y": block[:, 1]}
                            if n_axes < 3 else
                            {"x": block[:, 0], "y": block[:, 1], "z": block[:, 2]})

    if colour.paint_background:
        faint = dict(faint)
        faint.pop("color", None)
        faint.update(colour.marker(hue, show_scale=True))
        faint["opacity"] = 0.5
    figure.add_trace(scatter(
        **coords(shown), mode="markers", name="all entries", marker=faint,
        customdata=_row_ids(shown_rows), text=_hover_text(labels, shown_rows),
        hovertemplate="%{text}<extra></extra>",
    ))
    figure.add_trace(scatter(
        **coords(shown[shown_survivors]), mode="markers", name="slice survivors",
        marker=dict(**bright,
                    **colour.marker(hue[shown_survivors],
                                    show_scale=not colour.paint_background)),
        customdata=_row_ids(shown_rows[shown_survivors]),
        text=_hover_text(labels, shown_rows[shown_survivors]),
        hovertemplate="%{text}<extra></extra>",
    ))
    if reference_point is not None:
        marker = (dict(size=7, color="red", symbol="x") if n_axes >= 3 else
                  dict(size=14, color="red", symbol="x-thin",
                       line=dict(width=2.5, color="red")))
        figure.add_trace(scatter(
            **coords(np.asarray(reference_point, dtype=float)[None, :]),
            mode="markers", name="reference", marker=marker,
        ))

    if n_axes >= 3:
        # Group axes are strongly correlated -- three means of the same
        # b-values at nearby diffusion times differ only slightly -- so the
        # cloud lies close to the cube's main diagonal.  The default camera
        # looks straight down that diagonal and foreshortens the whole
        # manifold into a few pixels, which reads as an empty plot.  Look at
        # it from off the diagonal instead.
        figure.update_layout(scene=dict(
            xaxis_title=names[0], yaxis_title=names[1], zaxis_title=names[2],
            camera=dict(eye=dict(x=2.0, y=0.55, z=0.45)),
        ))
    else:
        figure.update_layout(xaxis_title=names[0], yaxis_title=names[1])

    dropped = int((~keep).sum())
    title = f"Prediction space ({axis_groups.COLLAPSE_METHODS[method]} of each group)"
    if dropped:
        title += f" - {dropped} entries excluded (S <= 0 in an ADC group)"
    figure.update_layout(
        title=title, height=430, margin=dict(l=55, r=10, t=40, b=45),
        showlegend=True, legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    return figure


def _log_axis_floor(values: np.ndarray) -> float:
    """A decade below the smallest positive value, for the free-water band."""
    positive = values[values > 0.0]
    if positive.size == 0:
        return 1e-12
    return float(positive.min()) / (10.0 ** FREE_WATER_BAND_DECADES)


def parameter_figure(data, eligible, survivors, reference_row,
                     x_name: str, y_name: str, colour: "Colour",
                     ellipse: dict | None = None,
                     ellipse_threshold: float = 0.0,
                     ellipse_mode: str = "profiled"):
    """Survivors in a parameter plane, log axes where the grid is log-spaced.

    The free-water atom has rho = V = 0, which a log axis cannot show.  Rather
    than letting it disappear, it is drawn in its own band a decade below the
    smallest positive grid value and labelled, so "no free-water marker" always
    means "free water is excluded", never "it fell off the axis".
    """
    labels = data.labels
    axes = {
        "rho": (labels.nominal_rhos, True, "rho (1/uL)"),
        "V": (labels.nominal_Vs, True, "V (pL)"),
        "k_io": (labels.kios, False, "k_io (1/s)"),
    }
    x_values, x_log, x_title = axes[x_name]
    y_values, y_log, y_title = axes[y_name]
    survivor_mask = np.asarray(survivors, dtype=bool)

    def placed(values, is_log, floor):
        """Grid values with non-positives moved onto the log axis's floor band."""
        if not is_log:
            return values
        return np.where(values > 0.0, values, floor)

    x_floor = _log_axis_floor(x_values[eligible]) if x_log else 0.0
    y_floor = _log_axis_floor(y_values[eligible]) if y_log else 0.0
    off_axis = np.zeros(len(eligible), dtype=bool)
    if x_log:
        off_axis |= x_values[eligible] <= 0.0
    if y_log:
        off_axis |= y_values[eligible] <= 0.0

    x_all = placed(x_values[eligible], x_log, x_floor)
    y_all = placed(y_values[eligible], y_log, y_floor)

    figure = go.Figure()
    background = (dict(size=3, **colour.marker(colour.subset(~off_axis),
                                               show_scale=False))
                  if colour.paint_background else dict(size=3, color="#dddddd"))
    figure.add_trace(go.Scattergl(
        x=x_all[~off_axis], y=y_all[~off_axis], mode="markers",
        name="all entries", marker=background,
        customdata=_row_ids(eligible[~off_axis]), text=_hover_text(labels, eligible[~off_axis]),
        hovertemplate="%{text}<extra></extra>",
    ))
    figure.add_trace(go.Scattergl(
        x=x_all[survivor_mask & ~off_axis], y=y_all[survivor_mask & ~off_axis],
        mode="markers", name="survivors",
        marker=dict(size=6, **colour.marker(
            colour.subset(survivor_mask & ~off_axis), show_scale=False)),
        customdata=_row_ids(eligible[survivor_mask & ~off_axis]),
        text=_hover_text(labels, eligible[survivor_mask & ~off_axis]),
        hovertemplate="%{text}<extra></extra>",
    ))
    undefined = ~colour.defined & ~off_axis
    if colour.paint_background and np.any(undefined):
        figure.add_trace(go.Scattergl(
            x=x_all[undefined], y=y_all[undefined], mode="markers",
            name=f"hue undefined ({int(undefined.sum())})",
            marker=dict(size=3, color="#e8e8e8"),
            customdata=_row_ids(eligible[undefined]), text=_hover_text(labels, eligible[undefined]),
            hovertemplate="%{text}<extra></extra>",
        ))
    if np.any(off_axis):
        # The free-water band: real entries, drawn where a log axis has no zero.
        figure.add_trace(go.Scattergl(
            x=x_all[off_axis], y=y_all[off_axis], mode="markers",
            name="free water (value 0, shown on the floor)",
            marker=dict(size=11, symbol="diamond-open", color="#1b6ca8",
                        line=dict(width=2, color="#1b6ca8")),
            customdata=_row_ids(eligible[off_axis]),
            text=_hover_text(labels, eligible[off_axis]),
            hovertemplate="%{text}<br>(plotted on the log-axis floor)<extra></extra>",
        ))
    if reference_row is not None:
        where = np.flatnonzero(eligible == reference_row)
        if where.size:
            figure.add_trace(go.Scattergl(
                x=[x_all[int(where[0])]], y=[y_all[int(where[0])]], mode="markers",
                name="reference",
                marker=dict(size=14, color="red", symbol="x-thin",
                            line=dict(width=2.5, color="red")),
            ))
    # The Fisher prediction of the same region, when asked for and when this
    # plane is the (rho, V) one the 2x2 block describes.
    if (ellipse is not None and reference_row is not None
            and {x_name, y_name} == {"rho", "V"}):
        centre = np.array([np.log(labels.nominal_rhos[reference_row]),
                           np.log(labels.nominal_Vs[reference_row])])
        curve = fisher_tools.confidence_ellipse(ellipse, ellipse_threshold,
                                                ellipse_mode)
        if curve is not None:
            points = np.exp(centre[None, :] + curve)
            order = (0, 1) if x_name == "rho" else (1, 0)
            figure.add_trace(go.Scattergl(
                x=points[:, order[0]], y=points[:, order[1]], mode="lines",
                name="Fisher ellipse",
                line=dict(color="#d62728", width=2)))

    figure.update_layout(
        title=f"{x_title} vs {y_title}", height=340,
        showlegend=bool(np.any(off_axis)) or ellipse is not None,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, font=dict(size=9)),
        margin=dict(l=60, r=10, t=35, b=45),
        xaxis=dict(title=x_title, type="log" if x_log else "linear"),
        yaxis=dict(title=y_title, type="log" if y_log else "linear"),
    )
    return figure


def placeholder_figure(message: str, height: int = 320) -> go.Figure:
    """An empty plot that says why it is empty.

    A bare `go.Figure()` renders as a blank axes box with no explanation, which
    reads as a broken plot rather than a switched-off one.
    """
    figure = go.Figure()
    figure.add_annotation(text=message, showarrow=False, xref="paper", yref="paper",
                          x=0.5, y=0.5, font=dict(size=13, color="#666"))
    figure.update_layout(
        height=height, margin=dict(l=40, r=20, t=30, b=30),
        xaxis=dict(visible=False), yaxis=dict(visible=False),
        plot_bgcolor="#fafafa",
    )
    return figure


def widths_figure(curve: list[dict], predicted: list[dict] | None = None) -> go.Figure:
    """How the parameter spreads shrink as measured columns are added.

    With ``predicted`` the Fisher bound is drawn beside the measured width, in
    the same max/min units: a `+/- sqrt(T) * CRLB` interval in `log rho` is a
    ratio of `exp(2 sqrt(T) CRLB)`.  Where the two separate, the quadratic
    approximation has stopped describing the real slice.
    """
    figure = go.Figure()
    counts = [step["n_measured"] for step in curve]
    figure.add_trace(go.Scatter(x=counts, y=[s["rho_log_width"] for s in curve],
                                name="rho max/min", mode="lines+markers"))
    figure.add_trace(go.Scatter(x=counts, y=[s["V_log_width"] for s in curve],
                                name="V max/min", mode="lines+markers"))

    # Survivor counts share a log axis, where an empty slice (0) has no place.
    # Draw those at a floor and keep the true number in the hover text.
    survivors = [s["n_survivors"] for s in curve]
    plotted = [value if value > 0 else WIDTHS_COUNT_FLOOR for value in survivors]
    figure.add_trace(go.Scatter(
        x=counts, y=plotted, name="survivors", mode="lines+markers", yaxis="y2",
        text=[f"{value} survivors" + (" (plotted on the floor)" if value == 0 else "")
              for value in survivors],
        hovertemplate="%{text}<extra></extra>",
    ))

    if predicted:
        figure.add_trace(go.Scatter(
            x=[s["n_measured"] for s in predicted],
            y=[s["predicted_rho_width"] for s in predicted],
            name="rho, Fisher bound", mode="lines+markers",
            line=dict(dash="dash"), marker=dict(symbol="diamond-open")))
        figure.add_trace(go.Scatter(
            x=[s["n_measured"] for s in predicted],
            y=[s["predicted_V_width"] for s in predicted],
            name="V, Fisher bound", mode="lines+markers",
            line=dict(dash="dash"), marker=dict(symbol="diamond-open")))

    skipped = sum(step.get("n_nonpositive_rho", 0) for step in curve)
    title = "Slice width vs number of measured columns (added in selection order)"
    if skipped:
        title += "<br><sub>widths taken over positive rho/V only; the free-water atom has no log width</sub>"
    figure.update_layout(
        title=title, height=320, margin=dict(l=60, r=60, t=50, b=45),
        xaxis_title="measured columns used",
        yaxis=dict(title="parameter width (max/min)", type="log"),
        yaxis2=dict(title="survivors", overlaying="y", side="right", type="log"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    return figure


def readout_table(result: slicing.SliceResult, n_eligible: int,
                  n_measured: int, elapsed_ms: float,
                  excluded: int = 0) -> html.Div:
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
            f"axis {name}", stats["n"], f"{stats['min']:.5g}",
            f"{stats['max']:.5g}", f"{stats['std']:.3g}", "-",
        ]))

    weight = float(np.exp(-result.threshold / 2.0))
    summary = (
        f"{result.n_survivors} survivors of {n_eligible} eligible entries "
        f"({100.0 * result.n_survivors / max(n_eligible, 1):.2f}%)  |  "
        f"{n_measured} measured columns, chi2 <= {result.threshold:.3g} "
        f"(relative bayes weight >= {weight:.3g})  |  sigma = "
        f"{result.sigma_measurement:.4g}, S0 {result.s0_mode}  |  {elapsed_ms:.0f} ms"
    )
    if excluded:
        summary += f"  |  {excluded} entries excluded from the plot (S <= 0 in an ADC group)"
    return html.Div([
        html.P(summary, style={"fontWeight": "600", "margin": "6px 0"}),
        html.Table([html.Thead(head), html.Tbody(body)],
                   style={"borderCollapse": "collapse", "fontSize": "12px"}),
    ])




# ---------------------------------------------------------------------------
# Fisher / CRLB
# ---------------------------------------------------------------------------

def fisher_plane_figure(labels, eligible, survivors, reference_row, colour,
                        report: dict | None, threshold: float,
                        ellipse_mode: str = "profiled",
                        show_axes: bool = True) -> go.Figure:
    """The (log rho, log V) plane with the slice AND its quadratic prediction.

    The survivors are the exact region; the ellipse is what the Fisher matrix
    at the reference predicts the same region should be.  They are drawn on one
    pair of axes precisely so the disagreement is visible: the slice follows the
    curved constant-`v_i` hyperbola, the ellipse is its tangent approximation.
    """
    rho = labels.nominal_rhos
    V = labels.nominal_Vs
    survivor_mask = np.asarray(survivors, dtype=bool)
    on_axis = (rho[eligible] > 0) & (V[eligible] > 0)

    figure = go.Figure()
    figure.add_trace(go.Scattergl(
        x=rho[eligible][on_axis], y=V[eligible][on_axis], mode="markers",
        name="all entries", marker=dict(size=3, color="#e0e0e0"),
        customdata=_row_ids(eligible[on_axis]), text=_hover_text(labels, eligible[on_axis]),
        hovertemplate="%{text}<extra></extra>"))
    keep = survivor_mask & on_axis
    figure.add_trace(go.Scattergl(
        x=rho[eligible][keep], y=V[eligible][keep], mode="markers",
        name="slice survivors",
        marker=dict(size=6, **colour.marker(colour.subset(keep), show_scale=True)),
        customdata=_row_ids(eligible[keep]),
        text=_hover_text(labels, eligible[keep]),
        hovertemplate="%{text}<extra></extra>"))

    note = ""
    if reference_row is not None and report is not None:
        centre = np.array([np.log(rho[reference_row]), np.log(V[reference_row])])
        ellipse = fisher_tools.confidence_ellipse(report, threshold, ellipse_mode)
        if ellipse is None:
            note = " - no ellipse: the Fisher matrix is not positive definite here"
        else:
            points = np.exp(centre[None, :] + ellipse)
            figure.add_trace(go.Scattergl(
                x=points[:, 0], y=points[:, 1], mode="lines",
                name=f"Fisher {ellipse_mode} ellipse, chi2 <= {threshold:.3g}",
                line=dict(color="#d62728", width=2)))
            if show_axes:
                segments = fisher_tools.eigenvector_segments(report, threshold,
                                                             ellipse_mode)
                for name, style in (("sloppy", "solid"), ("stiff", "dot")):
                    offset = segments[name]
                    ends = np.exp(np.stack([centre - offset, centre + offset]))
                    figure.add_trace(go.Scattergl(
                        x=ends[:, 0], y=ends[:, 1], mode="lines",
                        name=f"{name} direction",
                        line=dict(color="#d62728", width=1, dash=style)))
        figure.add_trace(go.Scattergl(
            x=[rho[reference_row]], y=[V[reference_row]], mode="markers",
            name="reference",
            marker=dict(size=14, color="red", symbol="x-thin",
                        line=dict(width=2.5, color="red"))))
    elif reference_row is None:
        note = " - pick a reference point to draw the ellipse"

    figure.update_layout(
        title=f"Slice vs Fisher prediction in (rho, V){note}",
        height=470, margin=dict(l=65, r=10, t=40, b=45),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, font=dict(size=10)),
        xaxis=dict(title="rho (1/uL)", type="log"),
        yaxis=dict(title="V (pL)", type="log"))
    return figure


def crlb_curve_figure(curve: list[dict]) -> go.Figure:
    """CRLB on each parameter as measured columns are added."""
    figure = go.Figure()
    counts = [step["n_measured"] for step in curve]
    for key, name in (("crlb_log_rho", "CRLB log rho"),
                      ("crlb_log_V", "CRLB log V"),
                      ("crlb_log_vi", "CRLB log v_i")):
        figure.add_trace(go.Scatter(x=counts, y=[step[key] for step in curve],
                                    name=name, mode="lines+markers"))
    figure.update_layout(
        title="Cramer-Rao bound vs number of measured columns"
              "<br><sub>gaps are column counts where the 3x3 matrix is not yet "
              "positive definite: fewer than three independent columns cannot "
              "bound three parameters</sub>",
        height=330, margin=dict(l=65, r=15, t=60, b=45),
        xaxis_title="measured columns used",
        yaxis=dict(title="CRLB (log units)", type="log"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02))
    return figure


def fisher_panel(report: dict | None, node, labels, row, stencil_note: str,
                 truncation: float | None, debias_note: str) -> html.Div:
    """The numbers behind the ellipse, as a table."""
    if report is None:
        return html.Div([
            html.P("No Fisher matrix at this reference.", style={"fontWeight": "600"}),
            html.P(stencil_note, style={"fontSize": "12px", "color": "#a33"}),
        ], style={"padding": "6px 12px"})

    def table(rows, header):
        style = {"padding": "2px 10px", "borderBottom": "1px solid #eee",
                 "textAlign": "right", "fontFamily": "monospace",
                 "fontSize": "12px"}
        head = html.Tr([html.Th(cell, style=style) for cell in header])
        body = [html.Tr([html.Td(cell, style=style) for cell in row])
                for row in rows]
        return html.Table([html.Thead(head), html.Tbody(body)],
                          style={"borderCollapse": "collapse",
                                 "marginRight": "24px"})

    names = ("log rho", "log V", "k_io")
    F = report["F"]
    matrix_rows = [[names[i]] + [f"{F[i, j]:.4g}" for j in range(3)]
                   for i in range(3)]
    bound_rows = [[names[i], f"{report['crlb'][i]:.4g}", f"{report['kappa'][i]:.4g}",
                   f"{report['eigenvalues'][i]:.4g}"] for i in range(3)]

    flags = []
    if not report["positive_definite"]:
        flags.append("F is NOT positive definite - the CRLB is undefined and is "
                     "reported as NaN rather than inverted anyway.")
    if truncation is not None and np.isfinite(truncation):
        flags.append(f"k=1 vs k=2 truncation estimate r_trunc = {truncation:.3g}.")
    if debias_note:
        flags.append(debias_note)
    if stencil_note:
        flags.append(stencil_note)

    summary = (
        f"row {row}, node {tuple(int(i) for i in node)}: "
        f"rho={labels.nominal_rhos[row]:.4g}, V={labels.nominal_Vs[row]:.4g}, "
        f"k_io={labels.kios[row]:g}, v_i={labels.vis[row]:.3f}  |  "
        f"CRLB on log v_i = {report['crlb_log_vi']:.4g}  |  "
        f"kappa(D F D) = {report['condition_number']:.4g}  |  "
        f"sloppy direction {report['sloppy_angle_deg']:.2f} deg from constant-v_i "
        f"in 3-D, {report['profiled_angle_deg']:.2f} deg in the "
        f"k_io-profiled plane"
    )
    return html.Div([
        html.P(summary, style={"fontWeight": "600", "margin": "6px 0",
                               "fontSize": "12px"}),
        html.Div([
            html.Div([html.Div("Fisher matrix F", style={"fontWeight": "600"}),
                      table(matrix_rows, ["", "log rho", "log V", "k_io"])]),
            html.Div([html.Div("bounds and spectrum", style={"fontWeight": "600"}),
                      table(bound_rows, ["", "CRLB", "kappa", "eig(D F D)"])]),
        ], style={"display": "flex", "flexWrap": "wrap"}),
        html.Ul([html.Li(flag, style={"fontSize": "11px"}) for flag in flags],
                style={"marginTop": "6px"}),
    ], style={"padding": "6px 12px"})
