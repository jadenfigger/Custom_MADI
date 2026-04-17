"""Single-profile viewer: parametric map / raw DWI, signal plot, match table."""
from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QKeySequence
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel,
                              QSplitter, QPushButton, QFileDialog,
                              QMessageBox, QShortcut, QStatusBar)

from ..app_state import OutputProfile
from ..data import ProfileData
from ..matching import find_top_matches, MatchRow
from ..constants import (MAP_NAMES, N_SHELLS)

from .map_canvas import MapCanvas
from .signal_canvas import SignalCanvas
from .match_table import MatchTable
from .settings_panel import SettingsPanel, VizSettings


class ViewerWidget(QWidget):
    """A single-profile viewer. The main window opens one per tab."""

    # Emitted whenever the user selects a voxel / changes slice, so
    # other viewers (in compare mode) can sync.
    voxelChanged = pyqtSignal(int, int, int, int)   # vx, vy, sl, axis

    def __init__(self, profile: OutputProfile, parent=None):
        super().__init__(parent)
        self.profile = profile
        self.pd = ProfileData(profile)
        self.settings = VizSettings()
        self.settings.map_name = profile and "kio" or "kio"

        self._selected_vx: Optional[int] = None
        self._selected_vy: Optional[int] = None
        self._matches: list[MatchRow] = []
        self._measured: Optional[np.ndarray] = None
        self._hover_text = ""

        # Layout
        self._build_ui()
        self._install_shortcuts()

        self._load_profile()

    # ================================================================
    #  UI
    # ================================================================
    def _build_ui(self):
        self.canvas_map = MapCanvas(self)
        self.canvas_map.voxelClicked.connect(self._on_voxel_clicked)
        self.canvas_map.set_hover_callback(self._on_hover)

        self.canvas_signal = SignalCanvas(self)
        self.canvas_signal.hoverInfo.connect(self._on_signal_hover)
        self.table = MatchTable(self)
        self.table.ranksChanged.connect(self._on_selection_changed)

        # Right column (top = info, middle = signal, bottom = table)
        right = QSplitter(Qt.Vertical)
        self.info_label = QLabel("Click a voxel to begin.")
        self.info_label.setStyleSheet(
            "background:#f8fafc; padding:8px; border:1px solid #cbd5e1; "
            "font-family:monospace;")
        self.info_label.setWordWrap(True)
        self.info_label.setMinimumHeight(90)
        right.addWidget(self.info_label)
        right.addWidget(self.canvas_signal)
        right.addWidget(self.table)
        right.setStretchFactor(0, 0)
        right.setStretchFactor(1, 1)
        right.setStretchFactor(2, 1)
        right.setSizes([110, 280, 260])

        # Settings dock (hidden by default, toggled via button)
        self.settings_panel = SettingsPanel(self.settings, self)
        self.settings_panel.changed.connect(self._on_setting_changed)

        # Main layout
        center = QSplitter(Qt.Horizontal)
        center.addWidget(self.canvas_map)
        center.addWidget(right)
        center.addWidget(self.settings_panel)
        center.setStretchFactor(0, 2)
        center.setStretchFactor(1, 3)
        center.setStretchFactor(2, 0)
        center.setSizes([600, 700, 280])

        self.status = QLabel("")
        self.status.setStyleSheet("color:#6b7280;")

        top_bar = QHBoxLayout()
        self.btn_save_shot = QPushButton("Save screenshot")
        self.btn_save_shot.clicked.connect(self._save_screenshot)
        self.btn_reload = QPushButton("Reload")
        self.btn_reload.clicked.connect(self.reload)
        self.btn_reset_view = QPushButton("Reset zoom")
        self.btn_reset_view.clicked.connect(self._reset_zoom)
        top_bar.addWidget(QLabel(f"<b>{self.profile.name}</b>"))
        top_bar.addStretch(1)
        top_bar.addWidget(self.btn_save_shot)
        top_bar.addWidget(self.btn_reset_view)
        top_bar.addWidget(self.btn_reload)

        v = QVBoxLayout(self)
        v.setContentsMargins(4, 4, 4, 4)
        v.addLayout(top_bar)
        v.addWidget(center, 1)
        v.addWidget(self.status)

    def _install_shortcuts(self):
        def bind(seq: str, fn):
            s = QShortcut(QKeySequence(seq), self)
            s.setContext(Qt.WidgetWithChildrenShortcut)
            s.activated.connect(fn)
        bind("Left",  lambda: self._cycle_delta(-1))
        bind("Right", lambda: self._cycle_delta(1))
        bind("Up",    lambda: self._cycle_match(-1))
        bind("Down",  lambda: self._cycle_match(1))
        bind("PgUp",   lambda: self._nudge_slice(1))
        bind("PgDown", lambda: self._nudge_slice(-1))
        bind("M", self._cycle_map)
        bind("L", lambda: self._toggle_log_y())

    # ================================================================
    #  Profile loading
    # ================================================================
    def _load_profile(self):
        if not self.pd.ensure_mask():
            self.status.setText("; ".join(self.pd.load_errors) or
                                "Mask not available.")
            self.canvas_map.clear()
            return

        deltas = self.profile.delta_ms_list()
        self.pd.fit_deltas = deltas
        self.pd.ensure_library()
        # Pre-build subset matrix so matching is fast on first click
        self.pd.ensure_lib_sub(deltas)

        # Init settings based on available data
        shape = self.pd.mask.shape
        self.settings.axis = int(self.settings.axis)
        self.settings.slice_idx = shape[self.settings.axis] // 2
        self.settings_panel.update_slice_range(shape[self.settings.axis])
        self.settings_panel.update_raw_deltas(deltas)
        self.settings_panel.update_signal_deltas(deltas)
        # Raw volume range (all inputs have the same)
        if deltas:
            first_raw = self.pd.ensure_raw(deltas[0])
            if first_raw is not None:
                self.settings_panel.update_vol_range(first_raw.shape[-1])

        # Pick a sensible starting map
        avail_maps = self.pd.available_maps()
        if avail_maps:
            if self.settings.map_name not in avail_maps:
                self.settings.map_name = avail_maps[0]
            self._render_map()
        else:
            self.settings.display_mode = "raw"
            self.settings_panel.cb_mode.setCurrentIndex(1)
            self._render_raw()
            self.status.setText(
                "No parameter maps found on disk — showing raw DWI. "
                "Run fitting from the profile menu to generate maps.")

    # ================================================================
    #  Rendering
    # ================================================================
    def _render_map(self, keep_view=False):
        if not self.pd.ensure_mask():
            return
        vol = self.pd.ensure_map(self.settings.map_name)
        if vol is None:
            self.canvas_map.clear()
            self.status.setText(
                f"Map '{self.settings.map_name}' not found in "
                f"{self.profile.output_dir}.")
            return
        self.canvas_map.set_geometry(
            self.pd.mask.shape, self.pd.zooms,
            self.settings.axis, self.settings.margin_mm)
        vrange = None
        if self.settings.map_vmin == self.settings.map_vmin and \
           self.settings.map_vmax == self.settings.map_vmax:
            vrange = (self.settings.map_vmin, self.settings.map_vmax)
        # Colormap override
        if self.settings.map_cmap:
            from ..constants import MAP_CMAPS as _CM
            _CM[self.settings.map_name] = self.settings.map_cmap
        self.canvas_map.set_map(self.settings.map_name, vol, self.pd.mask,
                                self.settings.slice_idx, vrange,
                                keep_view=keep_view)
        if self._selected_vx is not None:
            self.canvas_map.draw_crosshair(self._selected_vx, self._selected_vy)

    def _render_raw(self, keep_view=False):
        if not self.pd.ensure_mask():
            return
        delta = self.settings.raw_delta or (self.profile.delta_ms_list() or [0])[0]
        vol_slice = self.pd.raw_volume_slice(
            delta, self.settings.raw_vol_idx,
            self.settings.axis, self.settings.slice_idx)
        if vol_slice is None:
            self.canvas_map.clear()
            self.status.setText(f"Raw DWI Δ={delta} not available.")
            return
        self.canvas_map.set_geometry(
            self.pd.mask.shape, self.pd.zooms,
            self.settings.axis, self.settings.margin_mm)
        self.canvas_map.set_raw(delta, self.settings.raw_vol_idx,
                                vol_slice, self.pd.mask,
                                self.settings.slice_idx, keep_view=keep_view)
        if self._selected_vx is not None:
            self.canvas_map.draw_crosshair(self._selected_vx, self._selected_vy)

    def _render_display(self, keep_view=False):
        if self.settings.display_mode == "map":
            self._render_map(keep_view=keep_view)
        else:
            self._render_raw(keep_view=keep_view)

    def _recompute_matches(self):
        if self._selected_vx is None or self._selected_vy is None:
            return
        self._matches, self._measured = find_top_matches(
            self.pd,
            self._selected_vx, self._selected_vy,
            self.settings.slice_idx, self.settings.axis,
            top_n=self.settings.top_n,
            sort_by=self.settings.sort_by,
            vi_min=self.settings.vi_min,
            vi_max=self.settings.vi_max,
            rho_max=None if self.settings.rho_max != self.settings.rho_max
                         else self.settings.rho_max,
            fit_s0=self.settings.fit_s0,
        )
        # Keep only ranks that still exist after re-computation
        ranks = {m.rank for m in (self._matches or [])}
        self.settings.selected_ranks &= ranks
        if self._matches and not self.settings.selected_ranks:
            self.settings.selected_ranks = {1}

    def _redraw_signal_and_table(self):
        # Table
        self.table.set_visible_columns(self.settings.visible_cols)
        self.table.set_sort_key(self.settings.sort_by)
        self.table.set_matches(self._matches or [],
                               selected_ranks=self.settings.selected_ranks)
        # Signal
        if not self._matches or self._measured is None:
            self.canvas_signal.clear_plot()
        else:
            n_fit = len(self.pd.fit_deltas) or 1
            n_b = int(self.pd.lib_bundle["meta"]["n_b"]) if self.pd.lib_bundle else N_SHELLS
            di = int(np.clip(self.settings.delta_idx, 0, n_fit - 1))
            delta_ms = self.pd.fit_deltas[di] if self.pd.fit_deltas else 0.0
            # Per-Δ S0 values at the selected voxel, needed for the
            # avg-S0 observed variant.
            s0_per_delta = None
            if (self.settings.show_obs_avg_s0 or self.settings.show_obs_fit_s0) \
                    and self._selected_vx is not None:
                s0_per_delta = self.pd.s0_per_delta_at(
                    self._selected_vx, self._selected_vy,
                    self.settings.slice_idx, self.settings.axis)
            self.canvas_signal.plot_matches(
                self._measured, self._matches,
                self.settings.selected_ranks,
                delta_idx=di, delta_ms=delta_ms,
                n_b=n_b, n_fit=n_fit,
                log_y=self.settings.log_y,
                s0_per_delta=s0_per_delta,
                show_obs_per_delta=self.settings.show_obs_per_delta,
                show_obs_avg_s0=self.settings.show_obs_avg_s0,
                show_obs_fit_s0=self.settings.show_obs_fit_s0,
            )
        self._refresh_info()

    def _refresh_info(self):
        if self._selected_vx is None:
            self.info_label.setText("Click a voxel to begin.")
            return
        vx, vy, sl = self._selected_vx, self._selected_vy, self.settings.slice_idx
        kv = self.pd.map_at("kio", vx, vy, sl, self.settings.axis)
        rv = self.pd.map_at("rho", vx, vy, sl, self.settings.axis)
        vv = self.pd.map_at("V", vx, vy, sl, self.settings.axis)
        sv = self.pd.map_at("residual", vx, vy, sl, self.settings.axis)

        def fmt(x, suf=""):
            return "—" if x is None else f"{x:.3f}{suf}"
        disk_line = (
            f"Disk    kio={fmt(kv)}  "
            f"ρ={('—' if rv is None else f'{rv/1e3:.0f}k')}  "
            f"V={fmt(vv)}  SSE(disk)={fmt(sv)}")

        if self._matches:
            m = self._matches[0]
            live_line = (
                f"Live #1 kio={m.kio:.1f}  ρ={m.rho/1e3:.0f}k  V={m.V:.2f}  "
                f"SSE(lin)={m.sse_lin:.5f}  SSE(log)={m.sse_log:.5f}")
            if m.s0_fit is not None:
                live_line += f"  S0={m.s0_fit:.3g}"
        else:
            live_line = "No live matches."

        self.info_label.setText(
            f"Voxel ({vx}, {vy})  —  slice {sl}  axis {self.settings.axis}\n"
            f"{disk_line}\n"
            f"{live_line}")

    # ================================================================
    #  Events
    # ================================================================
    def _on_voxel_clicked(self, vx: int, vy: int):
        self._selected_vx, self._selected_vy = vx, vy
        self.canvas_map.draw_crosshair(vx, vy)
        self._recompute_matches()
        self._redraw_signal_and_table()
        self.voxelChanged.emit(vx, vy, self.settings.slice_idx, self.settings.axis)

    def _on_hover(self, vx, vy):
        if vx is None:
            self.status.setText("")
            return
        self.status.setText(f"({vx}, {vy})  slice {self.settings.slice_idx}")

    def _on_signal_hover(self, x, y, text):
        if x is None:
            self.status.setText("")
            return
        if text:
            # collapse newlines so the status bar stays one-line
            flat = text.replace("\n", "  |  ")
            self.status.setText(f"b={x:.0f}  S/S0={y:.4f}   {flat}")
        else:
            self.status.setText(f"b={x:.0f}  S/S0={y:.4f}")

    def _on_selection_changed(self, ranks: set[int]):
        self.settings.selected_ranks = ranks or {1}
        self._redraw_signal_and_table()

    def _on_setting_changed(self, field: str):
        # Drive the right re-rendering path
        if field in {"display_mode", "map_name", "map_cmap", "map_vmin",
                     "map_vmax", "raw_delta", "raw_vol_idx"}:
            self._render_display(keep_view=False)
            if self.settings.display_mode == "raw":
                # Match re-computation doesn't depend on display; nothing else.
                pass
        elif field in {"axis", "slice_idx", "margin_mm"}:
            if field == "axis":
                n_sl = self.pd.mask.shape[self.settings.axis]
                self.settings.slice_idx = int(np.clip(self.settings.slice_idx, 0, n_sl - 1))
                self.settings_panel.update_slice_range(n_sl)
                self.settings_panel.sync_slice(self.settings.slice_idx)
            self._render_display(keep_view=False)
            self._recompute_matches()
            self._redraw_signal_and_table()
        elif field in {"top_n", "sort_by", "vi_min", "vi_max",
                        "rho_max", "fit_s0"}:
            self._recompute_matches()
            self._redraw_signal_and_table()
        elif field in {"delta_idx", "log_y",
                        "show_obs_per_delta", "show_obs_avg_s0",
                        "show_obs_fit_s0"}:
            self._redraw_signal_and_table()
        elif field == "visible_cols":
            self.table.set_visible_columns(self.settings.visible_cols)
        # Always bubble selection so compare views can follow
        if self._selected_vx is not None:
            self.voxelChanged.emit(
                self._selected_vx, self._selected_vy,
                self.settings.slice_idx, self.settings.axis)

    # ---- keyboard helpers ----
    def _cycle_delta(self, step: int):
        n = len(self.pd.fit_deltas) or 1
        self.settings.delta_idx = (self.settings.delta_idx + step) % n
        self.settings_panel.cb_delta.setCurrentIndex(self.settings.delta_idx)
        self._redraw_signal_and_table()

    def _cycle_match(self, step: int):
        if not self._matches:
            return
        cur = sorted(self.settings.selected_ranks)[-1] if self.settings.selected_ranks else 1
        new_rank = int(np.clip(cur + step, 1, len(self._matches)))
        self.settings.selected_ranks = {new_rank}
        self.table.set_selected_ranks({new_rank})
        self._redraw_signal_and_table()

    def _nudge_slice(self, step: int):
        if self.pd.mask is None:
            return
        n_sl = self.pd.mask.shape[self.settings.axis]
        self.settings.slice_idx = int(np.clip(self.settings.slice_idx + step, 0, n_sl - 1))
        self.settings_panel.sync_slice(self.settings.slice_idx)
        self._render_display(keep_view=True)
        self._recompute_matches()
        self._redraw_signal_and_table()

    def _cycle_map(self):
        avail = self.pd.available_maps()
        if not avail:
            return
        i = (avail.index(self.settings.map_name) + 1) % len(avail) \
            if self.settings.map_name in avail else 0
        self.settings.map_name = avail[i]
        idx = self.settings_panel.cb_map.findText(self.settings.map_name)
        if idx >= 0:
            self.settings_panel.cb_map.setCurrentIndex(idx)

    def _toggle_log_y(self):
        self.settings.log_y = not self.settings.log_y
        self.settings_panel.cb_log_y.setChecked(self.settings.log_y)

    # ---- syncing (compare mode) ----
    def sync_voxel(self, vx: int, vy: int, sl: int, axis: int):
        """Apply external voxel/slice change without re-emitting."""
        changed = False
        if axis != self.settings.axis:
            self.settings.axis = axis
            self.settings_panel.cb_axis.setCurrentIndex(axis)
            changed = True
        if sl != self.settings.slice_idx:
            self.settings.slice_idx = sl
            self.settings_panel.sync_slice(sl)
            changed = True
        if changed:
            self._render_display(keep_view=False)
        self._selected_vx, self._selected_vy = vx, vy
        self.canvas_map.draw_crosshair(vx, vy)
        self._recompute_matches()
        self._redraw_signal_and_table()

    # ---- buttons ----
    def _save_screenshot(self):
        out_dir = self.profile.output_dir or str(Path.cwd())
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        fname = f"viewer_{self.profile.name.replace(' ', '_')}_{ts}.png"
        path = str(Path(out_dir) / fname)
        self.canvas_map.figure.savefig(path, dpi=180, bbox_inches="tight",
                                        facecolor="white")
        QMessageBox.information(self, "Saved", f"Saved map to:\n{path}")

    def _reset_zoom(self):
        self._render_display(keep_view=False)

    def reload(self):
        """Drop cached maps/library for this profile and re-init."""
        from ..data import cache
        c = cache()
        for d, p in self.profile.inputs:
            c.drop(p)
        c.drop(self.profile.mask_path)
        for n in MAP_NAMES + ["s0_fit", "s0_fit_over_measured"]:
            c.drop(self.profile.map_path(n))
        c.drop(self.profile.library_path)
        self.pd = ProfileData(self.profile)
        self._selected_vx = self._selected_vy = None
        self._matches = []
        self._measured = None
        self._load_profile()
