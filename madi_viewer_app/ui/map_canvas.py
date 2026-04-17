"""Matplotlib-embedded parametric-map / raw-DWI slice canvas."""
from __future__ import annotations

from typing import Optional

import numpy as np
from PyQt5.QtCore import pyqtSignal
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure

from ..constants import MAP_CMAPS, MAP_LABELS, DEFAULT_MARGIN_MM


class MapCanvas(FigureCanvasQTAgg):
    """2-D slice viewer with click + crosshair.

    Accepts either a parameter map (``set_map``) or a raw DWI volume-index
    slice (``set_raw``). Emits ``voxelClicked(vx, vy)`` when the user
    clicks inside the mask.
    """

    voxelClicked = pyqtSignal(int, int)

    def __init__(self, parent=None):
        self.fig = Figure(figsize=(5.5, 5.0), constrained_layout=False)
        super().__init__(self.fig)
        self.setParent(parent)

        self.ax = self.fig.add_axes([0.09, 0.13, 0.88, 0.82])
        self.cbar_ax = self.fig.add_axes([0.09, 0.07, 0.88, 0.028])

        self.fig.canvas.mpl_connect("button_press_event", self._on_click)
        self.fig.canvas.mpl_connect("motion_notify_event", self._on_motion)

        # Geometry
        self._zooms = np.array([1.0, 1.0, 1.0])
        self._axis = 2
        self._nx = 1
        self._ny = 1
        self._margin = DEFAULT_MARGIN_MM

        # Current frame
        self._mask_slice: Optional[np.ndarray] = None
        self._crosshair = None
        self._hover_cb = None  # optional external hover callback
        self._title = ""

    # --------------- public API ---------------
    def set_geometry(self, shape: tuple, zooms: np.ndarray,
                     axis: int, margin_mm: float = DEFAULT_MARGIN_MM):
        self._shape = shape
        self._zooms = np.asarray(zooms, dtype=float)
        self._axis = int(axis)
        self._margin = float(margin_mm)
        axes_2d = [i for i in range(3) if i != self._axis]
        self._zx = float(self._zooms[axes_2d[0]])
        self._zy = float(self._zooms[axes_2d[1]])
        # The "plane" dims
        self._nx = int(shape[axes_2d[0]])
        self._ny = int(shape[axes_2d[1]])
        self._extent = [0, self._nx * self._zx, 0, self._ny * self._zy]

    def _get_slice(self, vol, sl):
        s = [slice(None)] * 3
        s[self._axis] = sl
        return vol[tuple(s)]

    def set_map(self, map_name: str, map_vol: np.ndarray,
                mask: np.ndarray, sl: int,
                value_range: Optional[tuple] = None,
                keep_view: bool = False):
        sl_2d = self._get_slice(map_vol, sl)
        m_2d = self._get_slice(mask, sl)
        self._mask_slice = m_2d
        cmap = MAP_CMAPS.get(map_name, "viridis")
        label = MAP_LABELS.get(map_name, map_name)
        title = f"{map_name}  —  slice {sl}"
        self._draw(sl_2d, m_2d, cmap=cmap, label=label,
                   title=title, value_range=value_range, keep_view=keep_view)

    def set_raw(self, delta_ms: float, vol_idx: int,
                raw_slice_2d: np.ndarray, mask: np.ndarray, sl: int,
                keep_view: bool = False):
        m_2d = self._get_slice(mask, sl)
        self._mask_slice = m_2d
        self._draw(raw_slice_2d, m_2d, cmap="gray",
                   label=f"raw DWI (Δ={delta_ms:.0f}ms, vol {vol_idx})",
                   title=f"Δ={delta_ms:.0f}ms, vol {vol_idx}  —  slice {sl}",
                   value_range=None, keep_view=keep_view)

    def clear(self):
        self.ax.clear()
        self.cbar_ax.clear()
        self.ax.text(0.5, 0.5, "No data", ha="center", va="center",
                     transform=self.ax.transAxes, color="#6b7280")
        self.ax.set_axis_off()
        self.cbar_ax.set_axis_off()
        self.draw_idle()

    # --------------- coordinate helpers ---------------
    def disp_to_vx(self, xphys: float, yphys: float) -> tuple[int, int]:
        c = int(xphys / max(self._zx, 1e-9))
        r = int(yphys / max(self._zy, 1e-9))
        vx = int(np.clip(c, 0, self._nx - 1))
        vy = int(np.clip(self._ny - 1 - r, 0, self._ny - 1))
        return vx, vy

    def vx_to_disp(self, vx: int, vy: int) -> tuple[float, float]:
        xp = (vx + 0.5) * self._zx
        yp = (self._ny - 1 - vy + 0.5) * self._zy
        return xp, yp

    # --------------- crosshair ---------------
    def draw_crosshair(self, vx: int, vy: int):
        px, py = self.vx_to_disp(vx, vy)
        if self._crosshair is None:
            self._crosshair, = self.ax.plot(
                [px], [py], "+", color="white", ms=16, mew=2.2, zorder=10)
        else:
            self._crosshair.set_data([px], [py])
        self.draw_idle()

    def clear_crosshair(self):
        if self._crosshair is not None:
            self._crosshair.set_data([], [])
            self.draw_idle()

    # --------------- hover ---------------
    def set_hover_callback(self, cb):
        self._hover_cb = cb

    # --------------- internal draw ---------------
    def _draw(self, slice_2d, mask_2d, cmap, label, title,
              value_range=None, keep_view=False):
        xlim = self.ax.get_xlim() if keep_view else None
        ylim = self.ax.get_ylim() if keep_view else None
        self.ax.clear()

        rot = np.rot90(slice_2d)
        mask_rot = np.rot90(mask_2d) if mask_2d is not None else np.ones_like(rot, bool)
        bg = np.zeros_like(rot)
        self.ax.imshow(bg, cmap="gray", origin="lower", extent=self._extent,
                       aspect="equal", vmin=0, vmax=1)

        overlay_mask = ~mask_rot | (rot == 0)
        masked = np.ma.masked_where(overlay_mask, rot)

        if value_range is not None:
            vmin, vmax = value_range
        else:
            valid = masked.compressed()
            if valid.size:
                vmin, vmax = float(valid.min()), float(valid.max())
            else:
                vmin, vmax = 0.0, 1.0
            if vmin == vmax:
                vmax = vmin + 1e-6

        self._im = self.ax.imshow(
            masked, cmap=cmap, origin="lower", extent=self._extent,
            aspect="equal", interpolation="nearest", vmin=vmin, vmax=vmax)

        if keep_view and xlim is not None:
            self.ax.set_xlim(xlim); self.ax.set_ylim(ylim)
        else:
            self._auto_zoom(mask_rot)

        self.ax.set_title(title, color="#111827", pad=4)
        self.ax.set_xlabel("mm"); self.ax.set_ylabel("mm")
        self.ax.tick_params(labelsize=7, colors="#6b7280")

        # colorbar
        self.cbar_ax.clear()
        import matplotlib.pyplot as plt
        norm = plt.Normalize(vmin=vmin, vmax=vmax)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        cb = self.fig.colorbar(sm, cax=self.cbar_ax, orientation="horizontal")
        cb.ax.tick_params(labelsize=7, colors="#374151")
        cb.set_label(label, fontsize=8, color="#374151", labelpad=2)

        # re-add crosshair if any
        self._crosshair = None
        self.draw_idle()

    def _auto_zoom(self, mask_rot):
        rs, cs = np.where(mask_rot)
        if rs.size == 0:
            return
        cmin, cmax = cs.min(), cs.max()
        rmin, rmax = rs.min(), rs.max()
        xmin = cmin * self._zx - self._margin
        xmax = (cmax + 1) * self._zx + self._margin
        ymin = rmin * self._zy - self._margin
        ymax = (rmax + 1) * self._zy + self._margin
        self.ax.set_xlim(xmin, xmax)
        self.ax.set_ylim(ymin, ymax)

    # --------------- events ---------------
    def _on_click(self, event):
        if event.inaxes is not self.ax or event.xdata is None:
            return
        vx, vy = self.disp_to_vx(event.xdata, event.ydata)
        if self._mask_slice is not None and not self._mask_slice[vx, vy]:
            return
        self.voxelClicked.emit(vx, vy)

    def _on_motion(self, event):
        if self._hover_cb is None:
            return
        if event.inaxes is not self.ax or event.xdata is None:
            self._hover_cb(None, None)
            return
        vx, vy = self.disp_to_vx(event.xdata, event.ydata)
        self._hover_cb(vx, vy)
