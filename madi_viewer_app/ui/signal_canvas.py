"""Matplotlib-embedded signal plot with multi-match overlay support.

One Δ is displayed at a time. Observed signal is plotted as black dots;
every library match in the ``selected_ranks`` set is drawn as a coloured
line. The plot also supports overlaying library predictions from multiple
profiles (compare mode) — each profile gets its own marker style.
"""
from __future__ import annotations

import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure

from ..constants import MATCH_COLORS, OBS_COLOR, BVALS_DISPLAY


class SignalCanvas(FigureCanvasQTAgg):

    def __init__(self, parent=None):
        self.fig = Figure(figsize=(6.0, 3.2), constrained_layout=True)
        super().__init__(self.fig)
        self.setParent(parent)
        self.ax = self.fig.add_subplot(111)
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
                     obs_label: str = "Observed") -> None:
        self.ax.clear()
        obs = measured[delta_idx * n_b : (delta_idx + 1) * n_b]
        # Faded: un-selected matches
        for m in matches:
            if m.rank in selected_ranks:
                continue
            pred = m.pred[delta_idx * n_b : (delta_idx + 1) * n_b]
            self.ax.plot(BVALS_DISPLAY[:len(pred)], pred, "o-",
                         color=MATCH_COLORS[(m.rank - 1) % len(MATCH_COLORS)],
                         ms=3, lw=0.8, alpha=0.22, zorder=2)
        # Bold: selected matches
        for m in matches:
            if m.rank not in selected_ranks:
                continue
            pred = m.pred[delta_idx * n_b : (delta_idx + 1) * n_b]
            c = MATCH_COLORS[(m.rank - 1) % len(MATCH_COLORS)]
            lbl = (f"#{m.rank} "
                   f"kio={m.kio:.1f} "
                   f"ρ={m.rho/1e3:.0f}k "
                   f"V={m.V:.2f}")
            self.ax.plot(BVALS_DISPLAY[:len(pred)], pred, "o-",
                         color=c, ms=8, lw=2.2, alpha=0.95, zorder=5,
                         label=lbl)
        # Observed
        self.ax.scatter(BVALS_DISPLAY[:len(obs)], obs, c=OBS_COLOR, s=70,
                        zorder=6, label=obs_label,
                        edgecolors="white", linewidths=1.0)

        self.ax.set_title(f"Δ = {delta_ms:.0f} ms   "
                          f"({delta_idx + 1}/{n_fit})",
                          color="#111827", pad=4)
        self.ax.set_xlabel("b-value  (s/mm²)")
        self.ax.set_ylabel(r"$S(b)\,/\,S_0$")
        self.ax.legend(loc="upper right", frameon=True, fontsize=8)
        self.ax.set_xlim(600, 6800)
        ymax = max(1.05,
                   float(np.nanmax(obs)) * 1.15 if obs.size else 1.05)
        if log_y:
            self.ax.set_yscale("log")
            self.ax.set_ylim(1e-3, max(1.2, ymax))
        else:
            self.ax.set_yscale("linear")
            self.ax.set_ylim(-0.03, max(1.05, ymax))
        self.draw_idle()

    # ----- compare-mode API -----
    def plot_compare(self, entries: list[dict],
                     delta_ms: float,
                     n_b: int,
                     log_y: bool = False):
        """Overlay observed + top-1 match for multiple profiles.

        Each entry: {"label":str, "color":str, "measured":ndarray, "pred":ndarray,
                     "delta_idx":int, "kio":float, "rho":float, "V":float}
        """
        self.ax.clear()
        for i, e in enumerate(entries):
            obs = e["measured"][e["delta_idx"] * n_b : (e["delta_idx"] + 1) * n_b]
            pred = e["pred"][e["delta_idx"] * n_b : (e["delta_idx"] + 1) * n_b]
            c = e["color"]
            self.ax.plot(BVALS_DISPLAY[:len(pred)], pred, "-",
                         color=c, lw=2.2, alpha=0.9, zorder=4,
                         label=f"{e['label']} fit")
            self.ax.scatter(BVALS_DISPLAY[:len(obs)], obs,
                            facecolor="white", edgecolor=c, s=55, lw=1.2,
                            zorder=5, label=f"{e['label']} obs")
        self.ax.set_title(f"Compare — Δ = {delta_ms:.0f} ms",
                          color="#111827", pad=4)
        self.ax.set_xlabel("b-value  (s/mm²)")
        self.ax.set_ylabel(r"$S(b)\,/\,S_0$")
        self.ax.legend(loc="upper right", frameon=True, fontsize=7, ncol=2)
        self.ax.set_xlim(600, 6800)
        if log_y:
            self.ax.set_yscale("log"); self.ax.set_ylim(1e-3, 1.2)
        else:
            self.ax.set_yscale("linear"); self.ax.set_ylim(-0.03, 1.1)
        self.draw_idle()

    def clear_plot(self):
        self.ax.clear()
        self.ax.set_xlabel("b-value  (s/mm²)")
        self.ax.set_ylabel(r"$S(b)\,/\,S_0$")
        self.ax.set_xlim(600, 6800)
        self.ax.set_ylim(-0.03, 1.05)
        self.ax.text(0.5, 0.5, "Select a voxel",
                     transform=self.ax.transAxes,
                     ha="center", va="center",
                     fontsize=11, color="#6b7280", style="italic")
        self.draw_idle()
