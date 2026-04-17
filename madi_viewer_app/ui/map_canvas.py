"""Matplotlib-embedded parametric-map / raw-DWI slice canvas.

Left-click selects a voxel. Left-drag or right-drag pans the view,
scroll zooms around the cursor, and double-click restores the
default auto-zoom. A mask-aware hover callback forwards the voxel
index under the mouse to the parent widget.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
from PyQt5.QtCore import pyqtSignal
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure

from ..constants import MAP_CMAPS, MAP_LABELS, DEFAULT_MARGIN_MM


class MapCanvas(FigureCanvasQTAgg):
    """2-D slice viewer with click + crosshair."""

    voxelClicked = pyqtSignal(int, int)

    def __init__(self, parent=None):
        self.fig = Figure(figsize=(5.5, 5.0), constrained_layout=False)
        super().__init__(self.fig)
        self.setParent(parent)

        self.ax = self.fig.add_axes([0.09, 0.13, 0.88, 0.82])
        self.cbar_ax = self.fig.add_axes([0.09, 0.07, 0.88, 0.028])

        self.fig.canvas.mpl_connect("button_press_event",   self._on_press)
        self.fig.canvas.mpl_connect("button_release_event", self._on_release)
        self.fig.canvas.mpl_connect("motion_notify_event",  self._on_motion)
        self.fig.canvas.mpl_connect("scroll_event",         self._on_scroll)

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

        # Interactive state
        self._home_xlim: Optional[tuple] = None
        self._home_ylim: Optional[tuple] = None
        self._pan_origin = None   # (xdata, ydata, xlim, ylim)
        self._press_px:  Optional[tuple] = None  # pixel coords on press
        self._press_data: Optional[tuple] = None
        self._press_btn = None
        self._DRAG_PX = 4

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
        self._home_xlim = self.ax.get_xlim()
        self._home_ylim = self.ax.get_ylim()

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
        self._press_px = (event.x, event.y)
        self._press_data = (event.xdata, event.ydata)
        self._press_btn = event.button
        # Right button OR shift+left → pan directly
        if event.button == 3 or (event.key and "shift" in event.key):
            self._pan_origin = (event.xdata, event.ydata,
                                 self.ax.get_xlim(), self.ax.get_ylim())

    def _on_release(self, event):
        if self._press_px is None:
            return
        moved_px = 0
        if event.x is not None and event.y is not None:
            moved_px = max(abs(event.x - self._press_px[0]),
                           abs(event.y - self._press_px[1]))
        was_pan = self._pan_origin is not None
        self._pan_origin = None

        # Left-click (no drag) inside the mask → voxel pick
        if (not was_pan and self._press_btn == 1
                and moved_px <= self._DRAG_PX
                and event.inaxes is self.ax and event.xdata is not None):
            vx, vy = self.disp_to_vx(event.xdata, event.ydata)
            if self._mask_slice is None or self._mask_slice[vx, vy]:
                self.voxelClicked.emit(vx, vy)

        self._press_px = None
        self._press_data = None
        self._press_btn = None

    def _on_motion(self, event):
        # Pan handling (either initiated by right-click or by left-drag
        # after exceeding the click/drag threshold).
        if self._pan_origin is None and self._press_btn == 1 \
                and event.x is not None and self._press_px is not None:
            if max(abs(event.x - self._press_px[0]),
                   abs(event.y - self._press_px[1])) > self._DRAG_PX \
                    and self._press_data is not None:
                self._pan_origin = (self._press_data[0], self._press_data[1],
                                     self.ax.get_xlim(), self.ax.get_ylim())

        if self._pan_origin is not None and event.inaxes is self.ax \
                and event.xdata is not None:
            x0, y0, xlim0, ylim0 = self._pan_origin
            dx = event.xdata - x0
            dy = event.ydata - y0
            self.ax.set_xlim(xlim0[0] - dx, xlim0[1] - dx)
            self.ax.set_ylim(ylim0[0] - dy, ylim0[1] - dy)
            self.draw_idle()
            return

        # Hover callback
        if self._hover_cb is None:
            return
        if event.inaxes is not self.ax or event.xdata is None:
            self._hover_cb(None, None)
            return
        vx, vy = self.disp_to_vx(event.xdata, event.ydata)
        self._hover_cb(vx, vy)

    def _on_scroll(self, event):
        if event.inaxes is not self.ax or event.xdata is None:
            return
        factor = 1.25 if event.button == "down" else (1.0 / 1.25)
        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()
        xd, yd = event.xdata, event.ydata
        self.ax.set_xlim(xd - (xd - xlim[0]) * factor,
                          xd + (xlim[1] - xd) * factor)
        self.ax.set_ylim(yd - (yd - ylim[0]) * factor,
                          yd + (ylim[1] - yd) * factor)
        self.draw_idle()
