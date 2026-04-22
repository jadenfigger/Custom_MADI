"""Matplotlib-embedded signal plot with multi-match overlay support.

One Δ is displayed at a time. The measured signal can be drawn in up to
three concurrent variants — per-Δ S0, averaged S0, and fitted-S0 — each
with a different marker so the user can see how the observed curve would
change under different S0 normalisations without leaving the viewer.

The canvas also supports drag-to-pan, scroll-to-zoom, double-click-to-
reset, and a fast (blitted) hover tooltip that identifies the nearest
data point. The legend is placed outside the axes and made draggable so
it never obscures the signal itself.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
from PyQt5.QtCore import pyqtSignal
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure

from ..constants import MATCH_COLORS, OBS_COLOR, BVALS_DISPLAY


# Marker styles for the three observed-signal variants
_OBS_STYLES = {
    "per_delta": dict(marker="o", s=72, edgecolors="white", linewidths=1.0,
                       label="Obs (S/S0_Δ)",      zorder=6),
    "avg_s0":    dict(marker="s", s=62, edgecolors="white", linewidths=1.0,
                       label="Obs (S/S0_avg)",    zorder=6),
    "fit_s0":    dict(marker="^", s=72, edgecolors="white", linewidths=1.0,
                       label="Obs (S/S0_fit)",    zorder=6),
}


class SignalCanvas(FigureCanvasQTAgg):

    # Emitted on hover: (b_value, y_value, annotation_text). All None means
    # the cursor left the plot.
    hoverInfo = pyqtSignal(object, object, object)

    def __init__(self, parent=None):
        self.fig = Figure(figsize=(6.2, 3.4), constrained_layout=True)
        super().__init__(self.fig)
        self.setParent(parent)
        self.ax = self.fig.add_subplot(111)

        # Interactive state
        self._home_xlim: Optional[tuple] = None
        self._home_ylim: Optional[tuple] = None
        self._pan_origin = None           # (xdata, ydata, xlim, ylim)
        self._press_xy = None             # for click-vs-drag distinction
        self._hover_annot = None          # text artist
        self._hover_pt = None              # highlight marker
        self._bg = None                    # cached background for blitting
        self._scatter_points: list[dict] = []   # for hover pick
        self._n_b_current = 0

        self.fig.canvas.mpl_connect("button_press_event",   self._on_press)
        self.fig.canvas.mpl_connect("button_release_event", self._on_release)
        self.fig.canvas.mpl_connect("motion_notify_event",  self._on_motion)
        self.fig.canvas.mpl_connect("scroll_event",         self._on_scroll)
        self.fig.canvas.mpl_connect("draw_event",           self._on_draw)
        self.fig.canvas.mpl_connect("figure_leave_event",   self._on_leave)

        self.setMouseTracking(True)
        self.clear_plot()

    # ----- single-profile API -----
    def plot_matches(self,
                     measured: np.ndarray,
                     matches: list,
                     selected_ranks: set[int],
                     delta_idx: int,
                     delta_ms: float,
                     n_b: int,
                     n_fit: int,
                     log_y: bool = False,
                     obs_label: str = "Observed",
                     s0_per_delta: Optional[np.ndarray] = None,
                     show_obs_per_delta: bool = True,
                     show_obs_avg_s0:   bool = False,
                     show_obs_fit_s0:   bool = False,
                     show_raw_signal:   bool = False,
                     show_s0_at_b0:     bool = False) -> None:
        self.ax.clear()
        self._scatter_points = []
        self._n_b_current = n_b

        obs = measured[delta_idx * n_b : (delta_idx + 1) * n_b]
        bvals = BVALS_DISPLAY[:len(obs)]

        # --- raw-signal mode: un-normalise by the per-Δ measured S0. ---
        # Each selected match gets its own rescale (pred * s0_fit) so the
        # predicted curve sits where the library would sit in raw units.
        raw_scale = None
        s0_measured = None
        if (show_raw_signal or show_s0_at_b0) \
                and s0_per_delta is not None and s0_per_delta.size \
                and 0 <= delta_idx < s0_per_delta.size:
            s0_measured = float(s0_per_delta[delta_idx])
        if show_raw_signal and s0_measured is not None:
            raw_scale = s0_measured   # applied to obs; pred uses its own s0_fit

        # --- match predictions (faded: unselected; bold: selected) ---
        # In raw-signal mode, each match's curve is scaled by its own
        # fit-S0 (or the measured S0 if s0_fit is unavailable) so the
        # curve lives in the same raw-signal units as the observed.
        def _pred_scale(m) -> float:
            if not show_raw_signal:
                return 1.0
            if m.s0_fit is not None and abs(m.s0_fit) > 1e-12:
                return float(m.s0_fit)
            return s0_measured if s0_measured is not None else 1.0

        # We draw faded first so selected curves are on top.
        for m in matches:
            if m.rank in selected_ranks:
                continue
            pred = m.pred[delta_idx * n_b : (delta_idx + 1) * n_b] * _pred_scale(m)
            c = MATCH_COLORS[(m.rank - 1) % len(MATCH_COLORS)]
            self.ax.plot(bvals[:len(pred)], pred, "o-",
                         color=c, ms=3, lw=0.8, alpha=0.22, zorder=2)
        for m in matches:
            if m.rank not in selected_ranks:
                continue
            pred = m.pred[delta_idx * n_b : (delta_idx + 1) * n_b] * _pred_scale(m)
            c = MATCH_COLORS[(m.rank - 1) % len(MATCH_COLORS)]
            lbl = (f"#{m.rank} "
                   f"kio={m.kio:.1f} "
                   f"ρ={m.rho/1e3:.0f}k "
                   f"V={m.V:.2f}")
            self.ax.plot(bvals[:len(pred)], pred, "o-",
                         color=c, ms=7, lw=2.2, alpha=0.95, zorder=5,
                         label=lbl)
            # register each point for hover picking
            for b, y in zip(bvals[:len(pred)], pred):
                self._scatter_points.append({
                    "x": float(b), "y": float(y),
                    "label": lbl, "color": c, "kind": "pred"})

        # --- observed variants ---
        # In raw-signal mode everything is multiplied by the measured S0
        # so points sit in the same units as the rescaled predictions.
        obs_mul = float(raw_scale) if raw_scale is not None else 1.0

        # per-Δ is the "raw" observed (measured is already S/S0_Δ).
        if show_obs_per_delta:
            self._scatter_obs(bvals, obs * obs_mul, OBS_COLOR, "per_delta",
                              tag=obs_label)

        # avg-S0: rescale by (S0_delta / S0_avg). Requires s0_per_delta.
        if show_obs_avg_s0 and s0_per_delta is not None and s0_per_delta.size:
            s0_avg = float(np.mean(s0_per_delta))
            if s0_avg > 1e-10 and 0 <= delta_idx < s0_per_delta.size:
                factor = float(s0_per_delta[delta_idx]) / s0_avg
                self._scatter_obs(bvals, obs * factor * obs_mul,
                                  "#0ea5e9", "avg_s0",
                                  tag=f"S0_avg factor ×{factor:.3f}")

        # fit-S0: observed / s0_fit of the best selected match.
        if show_obs_fit_s0 and matches and selected_ranks:
            # use lowest-rank selected (i.e. "best" selected), else rank 1
            sel_sorted = sorted(selected_ranks)
            target_rank = sel_sorted[0] if sel_sorted else 1
            best = next((m for m in matches if m.rank == target_rank),
                        matches[0])
            if best.s0_fit is not None and abs(best.s0_fit) > 1e-10:
                self._scatter_obs(bvals, obs / best.s0_fit * obs_mul,
                                  "#a855f7", "fit_s0",
                                  tag=f"using #{best.rank} s0_fit="
                                      f"{best.s0_fit:.3g}")

        # --- S0 at b=0 overlay ---
        # Measured S0 as a black diamond; each selected match's fitted S0
        # as a colored star on the same x=0 column. In normalised mode the
        # values are divided by the measured S0 so the ratio (which is
        # 1 for the measured anchor) is visually meaningful.
        if show_s0_at_b0 and s0_measured is not None:
            norm_div = 1.0 if show_raw_signal else max(s0_measured, 1e-12)
            y_meas = s0_measured / norm_div
            self.ax.scatter([0.0], [y_meas], c=OBS_COLOR,
                            marker="D", s=70, edgecolors="white",
                            linewidths=1.0, zorder=7,
                            label=f"S0_meas={s0_measured:.3g}")
            self._scatter_points.append({
                "x": 0.0, "y": float(y_meas),
                "label": f"S0 measured = {s0_measured:.4g}",
                "color": OBS_COLOR, "kind": "s0_meas"})
            for m in matches:
                if m.rank not in selected_ranks:
                    continue
                if m.s0_fit is None:
                    continue
                y_fit = float(m.s0_fit) / norm_div
                c = MATCH_COLORS[(m.rank - 1) % len(MATCH_COLORS)]
                self.ax.scatter([0.0], [y_fit], c=c,
                                marker="*", s=140, edgecolors="white",
                                linewidths=1.0, zorder=8,
                                label=f"#{m.rank} S0_fit={m.s0_fit:.3g}")
                self._scatter_points.append({
                    "x": 0.0, "y": y_fit,
                    "label": (f"#{m.rank} S0_fit = {m.s0_fit:.4g}"
                              f"  (ratio {m.s0_fit / max(s0_measured, 1e-12):.3f})"),
                    "color": c, "kind": "s0_fit"})

        # --- axes + legend ---
        self.ax.set_title(f"Δ = {delta_ms:.0f} ms   "
                          f"({delta_idx + 1}/{n_fit})",
                          color="#111827", pad=4)
        self.ax.set_xlabel("b-value  (s/mm²)")
        if show_raw_signal:
            self.ax.set_ylabel(r"$S(b)$   (raw signal)")
        else:
            self.ax.set_ylabel(r"$S(b)\,/\,S_0$")

        # Y scale / bounds — include every variant the user turned on so
        # the autoscale doesn't clip raw-mode curves / S0 markers.
        obs_scale = float(raw_scale) if raw_scale is not None else 1.0
        all_y = [obs * obs_scale]
        if show_obs_avg_s0 and s0_per_delta is not None and s0_per_delta.size:
            factor = (float(s0_per_delta[delta_idx]) /
                      max(float(np.mean(s0_per_delta)), 1e-12))
            all_y.append(obs * factor * obs_scale)
        if show_obs_fit_s0 and matches and selected_ranks:
            m = next((mm for mm in matches if mm.rank in selected_ranks),
                     matches[0])
            if m.s0_fit and abs(m.s0_fit) > 1e-10:
                all_y.append(obs / m.s0_fit * obs_scale)
        if show_raw_signal:
            # Include selected predictions' raw-scaled maxima so the view
            # breathes for high-S0 curves.
            for m in matches:
                if m.rank in selected_ranks:
                    all_y.append(
                        m.pred[delta_idx * n_b : (delta_idx + 1) * n_b]
                        * _pred_scale(m))
        if show_s0_at_b0 and s0_measured is not None:
            norm_div = 1.0 if show_raw_signal else max(s0_measured, 1e-12)
            all_y.append(np.array([s0_measured / norm_div]))
            for m in matches:
                if m.rank in selected_ranks and m.s0_fit is not None:
                    all_y.append(np.array([float(m.s0_fit) / norm_div]))
        all_y_arr = np.concatenate(
            [np.asarray(a).ravel() for a in all_y if a is not None])
        ymax = (float(np.nanmax(all_y_arr)) * 1.15
                if all_y_arr.size else 1.05)
        if show_raw_signal:
            ymin_default = -0.03 * max(ymax, 1.0)
        else:
            ymax = max(1.05, ymax)
            ymin_default = -0.03
        if log_y:
            self.ax.set_yscale("log")
            floor = max(1e-3, ymax * 1e-4) if show_raw_signal else 1e-3
            self.ax.set_ylim(floor, max(1.2, ymax))
        else:
            self.ax.set_yscale("linear")
            self.ax.set_ylim(ymin_default, ymax)
        # When the S0 marker is on, extend the x-axis to include b=0.
        xmin = -150 if show_s0_at_b0 else 600
        self.ax.set_xlim(xmin, 6800)

        leg = self.ax.legend(loc="upper left",
                              bbox_to_anchor=(1.01, 1.0),
                              frameon=True, fontsize=8,
                              borderaxespad=0.0)
        if leg is not None:
            leg.set_draggable(True)
            leg.set_title("Click & drag", prop={"size": 7})

        # Save "home" for reset view
        self._home_xlim = self.ax.get_xlim()
        self._home_ylim = self.ax.get_ylim()

        self._hover_annot = None
        self._hover_pt = None
        self._bg = None
        self.draw_idle()

    def _scatter_obs(self, bvals, yvals, color, kind: str, tag: str = ""):
        style = dict(_OBS_STYLES[kind])
        label = style.pop("label")
        self.ax.scatter(bvals[:len(yvals)], yvals, c=color,
                        label=label if tag == "" else f"{label}",
                        **style)
        for b, y in zip(bvals[:len(yvals)], yvals):
            self._scatter_points.append({
                "x": float(b), "y": float(y),
                "label": f"{label} · {tag}" if tag else label,
                "color": color, "kind": kind})

    # ----- compare-mode API -----
    def plot_compare(self, entries: list[dict],
                     delta_ms: float,
                     n_b: int,
                     log_y: bool = False):
        """Overlay observed + top-1 match for multiple profiles."""
        self.ax.clear()
        self._scatter_points = []
        self._n_b_current = n_b
        for i, e in enumerate(entries):
            obs = e["measured"][e["delta_idx"] * n_b : (e["delta_idx"] + 1) * n_b]
            pred = e["pred"][e["delta_idx"] * n_b : (e["delta_idx"] + 1) * n_b]
            c = e["color"]
            bvals = BVALS_DISPLAY[:len(pred)]
            self.ax.plot(bvals, pred, "-",
                         color=c, lw=2.2, alpha=0.9, zorder=4,
                         label=f"{e['label']} fit")
            self.ax.scatter(bvals[:len(obs)], obs,
                            facecolor="white", edgecolor=c, s=55, lw=1.2,
                            zorder=5, label=f"{e['label']} obs")
            for b, y in zip(bvals[:len(obs)], obs):
                self._scatter_points.append({
                    "x": float(b), "y": float(y),
                    "label": f"{e['label']} obs",
                    "color": c, "kind": "obs"})
        self.ax.set_title(f"Compare — Δ = {delta_ms:.0f} ms",
                          color="#111827", pad=4)
        self.ax.set_xlabel("b-value  (s/mm²)")
        self.ax.set_ylabel(r"$S(b)\,/\,S_0$")
        leg = self.ax.legend(loc="upper left",
                              bbox_to_anchor=(1.01, 1.0),
                              frameon=True, fontsize=7, ncol=1,
                              borderaxespad=0.0)
        if leg is not None:
            leg.set_draggable(True)
        self.ax.set_xlim(600, 6800)
        if log_y:
            self.ax.set_yscale("log"); self.ax.set_ylim(1e-3, 1.2)
        else:
            self.ax.set_yscale("linear"); self.ax.set_ylim(-0.03, 1.1)
        self._home_xlim = self.ax.get_xlim()
        self._home_ylim = self.ax.get_ylim()
        self._hover_annot = None
        self._hover_pt = None
        self._bg = None
        self.draw_idle()

    def clear_plot(self):
        self.ax.clear()
        self._scatter_points = []
        self.ax.set_xlabel("b-value  (s/mm²)")
        self.ax.set_ylabel(r"$S(b)\,/\,S_0$")
        self.ax.set_xlim(600, 6800)
        self.ax.set_ylim(-0.03, 1.05)
        self.ax.text(0.5, 0.5, "Select a voxel",
                     transform=self.ax.transAxes,
                     ha="center", va="center",
                     fontsize=11, color="#6b7280", style="italic")
        self._home_xlim = self.ax.get_xlim()
        self._home_ylim = self.ax.get_ylim()
        self._hover_annot = None
        self._hover_pt = None
        self._bg = None
        self.draw_idle()

    # ----- interactive plumbing -----
    def reset_view(self):
        if self._home_xlim is None:
            return
        self.ax.set_xlim(self._home_xlim)
        self.ax.set_ylim(self._home_ylim)
        self.draw_idle()

    def _on_press(self, event):
        if event.inaxes is not self.ax or event.xdata is None:
            return
        if event.dblclick:
            self.reset_view()
            return
        if event.button == 1:   # left = pan
            self._pan_origin = (event.xdata, event.ydata,
                                 self.ax.get_xlim(), self.ax.get_ylim())
            self._press_xy = (event.x, event.y)
        elif event.button == 3:  # right = also pan
            self._pan_origin = (event.xdata, event.ydata,
                                 self.ax.get_xlim(), self.ax.get_ylim())

    def _on_release(self, event):
        self._pan_origin = None
        self._press_xy = None

    def _on_motion(self, event):
        # Pan takes priority
        if self._pan_origin is not None and event.inaxes is self.ax \
                and event.xdata is not None:
            x0, y0, xlim0, ylim0 = self._pan_origin
            dx = event.xdata - x0
            dy = event.ydata - y0
            self.ax.set_xlim(xlim0[0] - dx, xlim0[1] - dx)
            self.ax.set_ylim(ylim0[0] - dy, ylim0[1] - dy)
            self.draw_idle()
            return
        # Hover
        if event.inaxes is not self.ax or event.xdata is None:
            self._clear_hover()
            self.hoverInfo.emit(None, None, None)
            return
        self._update_hover(event)

    def _on_scroll(self, event):
        if event.inaxes is not self.ax or event.xdata is None:
            return
        factor = 1.25 if event.button == "down" else (1.0 / 1.25)
        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()
        xd, yd = event.xdata, event.ydata
        nxlim = (xd - (xd - xlim[0]) * factor,
                 xd + (xlim[1] - xd) * factor)
        nylim = (yd - (yd - ylim[0]) * factor,
                 yd + (ylim[1] - yd) * factor)
        self.ax.set_xlim(nxlim)
        self.ax.set_ylim(nylim)
        self.draw_idle()

    def _on_draw(self, _event):
        # Cache the background after every normal draw so hover-blit works.
        self._bg = self.fig.canvas.copy_from_bbox(self.ax.bbox)
        self._hover_annot = None
        self._hover_pt = None

    def _on_leave(self, _event):
        self._clear_hover()
        self.hoverInfo.emit(None, None, None)

    def _clear_hover(self):
        if self._hover_annot is None and self._hover_pt is None:
            return
        if self._bg is not None:
            self.fig.canvas.restore_region(self._bg)
        self._hover_annot = None
        self._hover_pt = None
        self.fig.canvas.blit(self.ax.bbox)

    def _update_hover(self, event):
        """Find the nearest data point (in display space) and annotate it."""
        # Emit the cursor coords always
        self.hoverInfo.emit(float(event.xdata), float(event.ydata), None)
        if not self._scatter_points:
            return

        # nearest in *display* pixels so aspect doesn't distort
        tx, ty = self.ax.transData.transform((event.xdata, event.ydata))
        best = None
        best_d2 = 18 ** 2   # within 18 px
        for sp in self._scatter_points:
            px, py = self.ax.transData.transform((sp["x"], sp["y"]))
            d2 = (px - tx) ** 2 + (py - ty) ** 2
            if d2 < best_d2:
                best_d2 = d2
                best = sp

        if best is None:
            self._clear_hover()
            return

        txt = (f"b = {best['x']:.0f}\n"
               f"S/S0 = {best['y']:.4f}\n"
               f"{best['label']}")

        # emit a hover-info tuple with the text for the status bar
        self.hoverInfo.emit(float(event.xdata), float(event.ydata), txt)

        # Draw via blitting (fast even on big plots)
        if self._bg is None:
            self.draw()   # force a fresh cache
            return
        self.fig.canvas.restore_region(self._bg)
        if self._hover_pt is None:
            self._hover_pt, = self.ax.plot(
                [best["x"]], [best["y"]], "o",
                color=best["color"], markersize=14, mew=2,
                mfc="none", zorder=9, animated=True)
        else:
            self._hover_pt.set_data([best["x"]], [best["y"]])
            self._hover_pt.set_color(best["color"])
        self.ax.draw_artist(self._hover_pt)

        if self._hover_annot is None:
            self._hover_annot = self.ax.annotate(
                txt,
                xy=(best["x"], best["y"]),
                xytext=(12, 12), textcoords="offset points",
                bbox=dict(boxstyle="round,pad=0.4",
                          fc="#fef9c3", ec="#ca8a04", lw=0.8, alpha=0.95),
                fontsize=8, zorder=10, animated=True)
        else:
            self._hover_annot.set_text(txt)
            self._hover_annot.xy = (best["x"], best["y"])
        self.ax.draw_artist(self._hover_annot)
        self.fig.canvas.blit(self.ax.bbox)
