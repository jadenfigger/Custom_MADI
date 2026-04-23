"""Multi-profile comparison widget.

Each compared profile owns a ``ViewerWidget`` laid out side-by-side; a
master ``voxelChanged`` relay keeps slice + voxel synchronised across
them. Plus: a difference-map panel (A - B for the currently-shown map)
and a merged signal plot that overlays every profile's best fit at the
selected voxel.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QSplitter,
                              QLabel, QComboBox, QCheckBox, QPushButton,
                              QMessageBox)

from .viewer_widget import ViewerWidget
from .map_canvas import MapCanvas
from .signal_canvas import SignalCanvas
from ..app_state import OutputProfile
from ..constants import MAP_NAMES, N_SHELLS


class CompareWidget(QWidget):

    def __init__(self, profiles: list[OutputProfile], parent=None):
        super().__init__(parent)
        self.profiles = profiles
        self.viewers: list[ViewerWidget] = []
        self._sync = True
        self._handling_sync = False

        self._build_ui()

    def _build_ui(self):
        # Toolbar
        bar = QHBoxLayout()
        self.cb_sync = QCheckBox("Sync slice + voxel")
        self.cb_sync.setChecked(True)
        self.cb_sync.stateChanged.connect(
            lambda _: setattr(self, "_sync", self.cb_sync.isChecked()))
        bar.addWidget(self.cb_sync)

        self.cb_diff_map = QComboBox()
        self.cb_diff_map.addItems(MAP_NAMES)
        self.cb_diff_map.setCurrentText("kio")
        self.cb_diff_map.currentTextChanged.connect(lambda _: self._redraw_diff())
        bar.addWidget(QLabel("  Diff map:")); bar.addWidget(self.cb_diff_map)

        self.cb_a = QComboBox()
        self.cb_b = QComboBox()
        for i, p in enumerate(self.profiles):
            self.cb_a.addItem(f"{i+1}. {p.name}", i)
            self.cb_b.addItem(f"{i+1}. {p.name}", i)
        self.cb_a.setCurrentIndex(0)
        self.cb_b.setCurrentIndex(min(1, len(self.profiles)-1))
        self.cb_a.currentIndexChanged.connect(lambda _: self._redraw_diff())
        self.cb_b.currentIndexChanged.connect(lambda _: self._redraw_diff())
        bar.addWidget(QLabel("  A −")); bar.addWidget(self.cb_a)
        bar.addWidget(QLabel("B:")); bar.addWidget(self.cb_b)

        self.btn_refresh_diff = QPushButton("Refresh diff")
        self.btn_refresh_diff.clicked.connect(self._redraw_diff)
        bar.addWidget(self.btn_refresh_diff)

        bar.addStretch(1)

        # Viewers row
        viewers_split = QSplitter(Qt.Horizontal)
        for p in self.profiles:
            v = ViewerWidget(p, self)
            v.voxelChanged.connect(self._on_voxel_changed)
            self.viewers.append(v)
            viewers_split.addWidget(v)

        # Bottom: diff map + merged signal
        bottom_split = QSplitter(Qt.Horizontal)
        self.diff_canvas = MapCanvas(self)
        self.merged_signal = SignalCanvas(self)
        bottom_split.addWidget(self.diff_canvas)
        bottom_split.addWidget(self.merged_signal)
        bottom_split.setSizes([500, 500])

        main = QSplitter(Qt.Vertical)
        main.addWidget(viewers_split)
        main.addWidget(bottom_split)
        main.setStretchFactor(0, 3)
        main.setStretchFactor(1, 1)
        main.setSizes([700, 280])

        v = QVBoxLayout(self)
        v.setContentsMargins(4, 4, 4, 4)
        v.addLayout(bar)
        v.addWidget(main, 1)

    # ---------------- syncing ----------------
    def _on_voxel_changed(self, vx: int, vy: int, sl: int, axis: int):
        if self._handling_sync:
            return
        sender = self.sender()
        if self._sync:
            self._handling_sync = True
            try:
                for v in self.viewers:
                    if v is sender:
                        continue
                    v.sync_voxel(vx, vy, sl, axis)
            finally:
                self._handling_sync = False
        self._redraw_diff()
        self._redraw_merged_signal()

    # ---------------- diff map ----------------
    def _redraw_diff(self):
        ai = int(self.cb_a.currentIndex())
        bi = int(self.cb_b.currentIndex())
        if ai == bi or ai < 0 or bi < 0:
            self.diff_canvas.clear()
            return
        map_name = self.cb_diff_map.currentText()
        va = self.viewers[ai]; vb = self.viewers[bi]
        vol_a = va.pd.ensure_map(map_name)
        vol_b = vb.pd.ensure_map(map_name)
        if vol_a is None or vol_b is None:
            self.diff_canvas.clear()
            return
        if vol_a.shape != vol_b.shape:
            self.diff_canvas.clear()
            return
        diff = vol_a - vol_b
        mask = va.pd.mask & vb.pd.mask if va.pd.mask is not None and vb.pd.mask is not None \
            else (va.pd.mask if va.pd.mask is not None else vb.pd.mask)
        if mask is None:
            return
        self.diff_canvas.set_geometry(diff.shape, va.pd.zooms,
                                      va.settings.axis, va.settings.margin_mm)
        mx = float(np.nanmax(np.abs(diff[mask]))) if mask.any() else 1.0
        if mx == 0:
            mx = 1e-6
        # We adapt MapCanvas to plot the diff with a symmetric range
        import matplotlib.pyplot as plt
        self.diff_canvas.ax.clear()
        self.diff_canvas.cbar_ax.clear()
        sl = va.settings.slice_idx
        s = [slice(None)] * 3
        s[va.settings.axis] = sl
        diff_2d = diff[tuple(s)]
        mask_2d = mask[tuple(s)]
        rot = np.rot90(diff_2d)
        mrot = np.rot90(mask_2d)
        masked = np.ma.masked_where(~mrot, rot)
        self.diff_canvas.ax.imshow(np.zeros_like(rot), cmap="gray",
                                    origin="lower",
                                    extent=self.diff_canvas._extent,
                                    aspect="equal", vmin=0, vmax=1)
        im = self.diff_canvas.ax.imshow(
            masked, cmap="coolwarm", origin="lower",
            extent=self.diff_canvas._extent, aspect="equal",
            interpolation="nearest", vmin=-mx, vmax=mx)
        self.diff_canvas.ax.set_title(
            f"Δ{map_name}  =  {va.profile.name}  −  {vb.profile.name}",
            pad=4)
        self.diff_canvas.ax.set_xlabel("mm"); self.diff_canvas.ax.set_ylabel("mm")
        self.diff_canvas._auto_zoom(mrot)
        norm = plt.Normalize(vmin=-mx, vmax=mx)
        sm = plt.cm.ScalarMappable(cmap="coolwarm", norm=norm)
        cb = self.diff_canvas.fig.colorbar(sm,
                                            cax=self.diff_canvas.cbar_ax,
                                            orientation="horizontal")
        cb.ax.tick_params(labelsize=7, colors="#374151")
        cb.set_label(f"Δ {map_name}", fontsize=8, color="#374151", labelpad=2)
        self.diff_canvas.draw_idle()

    # ---------------- merged signal plot ----------------
    def _redraw_merged_signal(self):
        entries = []
        delta_ms = None
        for v in self.viewers:
            if not v._matches or v._measured is None:
                continue
            n_fit = len(v.pd.fit_deltas) or 1
            di = int(np.clip(v.settings.delta_idx, 0, n_fit - 1))
            delta_ms = v.pd.fit_deltas[di]
            entries.append({
                "label":    v.profile.name,
                "color":    v.profile.color,
                "measured": v._measured,
                "pred":     v._matches[0].pred,
                "delta_idx": di,
                "kio": v._matches[0].kio,
                "rho": v._matches[0].rho,
                "V":   v._matches[0].V,
            })
        if not entries:
            self.merged_signal.clear_plot()
            return
        n_b = int(self.viewers[0].pd.lib_bundle["meta"]["n_b"]) \
              if self.viewers[0].pd.lib_bundle else N_SHELLS
        self.merged_signal.plot_compare(entries, delta_ms or 0.0, n_b,
                                        log_y=self.viewers[0].settings.log_y)
