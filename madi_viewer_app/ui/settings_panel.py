"""Per-viewer visualization settings panel.

All settings emit ``changed`` whenever the user modifies one of them —
the parent widget is expected to re-render on the signal. The panel
itself is stateless except for the current values; it does not persist
them on its own (the enclosing ``ViewerWidget`` owns the state).
"""
from __future__ import annotations

from dataclasses import dataclass, field

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QGroupBox, QFormLayout,
                              QComboBox, QCheckBox, QDoubleSpinBox,
                              QSpinBox, QPushButton, QHBoxLayout,
                              QLabel, QScrollArea)

from .match_table import ALL_COLUMNS, DEFAULT_VISIBLE
from ..constants import MAP_CMAPS, DEFAULT_TOP_N


@dataclass
class VizSettings:
    """The live state driven by SettingsPanel (one per viewer)."""

    # Display mode
    display_mode:  str = "map"            # "map" | "raw"
    map_name:      str = "kio"
    map_cmap:      str = ""               # "" = auto
    map_vmin:      float = float("nan")
    map_vmax:      float = float("nan")
    raw_delta:     float = 0.0            # first available will be chosen
    raw_vol_idx:   int = 0

    # Geometry
    axis:          int = 2                # 0=sag, 1=cor, 2=axial
    slice_idx:     int = 0
    margin_mm:     float = 1.5

    # Matcher
    top_n:         int = DEFAULT_TOP_N
    sort_by:       str = "lin"
    vi_min:        float = 0.5
    vi_max:        float = 0.95
    rho_max:       float = float("nan")
    fit_s0:        bool = False

    # Pre-processing (mirrors scripts/fit_data.py flags). When a profile
    # is opened these default to whatever the batch fit used, so live
    # matching sees the same signal the saved maps were fit against.
    rician_correct: bool = False
    noise_sigma:    float = float("nan")   # NaN → auto-estimate
    avg_s0:         bool = False

    # Signal plot
    delta_idx:     int = 0                # which Δ is visible
    log_y:         bool = False
    # Observed-signal display modes (can be combined)
    show_obs_per_delta: bool = True
    show_obs_avg_s0:    bool = False
    show_obs_fit_s0:    bool = False

    # Match table
    visible_cols:  set[str] = field(default_factory=lambda: set(DEFAULT_VISIBLE))

    # Selected match ranks (for signal overlay). Rank #1 is default.
    selected_ranks: set[int] = field(default_factory=lambda: {1})


class SettingsPanel(QWidget):

    changed = pyqtSignal(str)   # name of the field that changed

    def __init__(self, settings: VizSettings, parent=None):
        super().__init__(parent)
        self.settings = settings

        self._block_signals = False

        outer = QVBoxLayout(self)
        outer.setContentsMargins(2, 2, 2, 2)
        outer.setSpacing(4)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        inner = QWidget()
        layout = QVBoxLayout(inner)
        layout.setContentsMargins(4, 4, 4, 4)

        self._build_display_box(layout)
        self._build_geometry_box(layout)
        self._build_preproc_box(layout)
        self._build_matcher_box(layout)
        self._build_signal_box(layout)
        self._build_table_box(layout)
        layout.addStretch(1)

        scroll.setWidget(inner)
        outer.addWidget(scroll)

    # ---------------- group boxes ----------------
    def _build_display_box(self, parent_layout):
        g = QGroupBox("Display")
        f = QFormLayout(g)

        self.cb_mode = QComboBox()
        self.cb_mode.addItems(["Parameter maps", "Raw DWI volume"])
        self.cb_mode.setCurrentIndex(0 if self.settings.display_mode == "map" else 1)
        self.cb_mode.currentIndexChanged.connect(self._on_mode)
        f.addRow("Mode", self.cb_mode)

        self.cb_map = QComboBox()
        self.cb_map.addItems(["kio", "rho", "V", "residual",
                               "s0_fit", "s0_fit_over_measured"])
        self.cb_map.setCurrentText(self.settings.map_name)
        self.cb_map.currentTextChanged.connect(
            lambda v: self._set("map_name", v))
        f.addRow("Map", self.cb_map)

        self.cb_cmap = QComboBox()
        self.cb_cmap.addItem("auto", "")
        for c in ["viridis", "inferno", "plasma", "magma",
                  "cividis", "coolwarm", "gray", "jet", "turbo"]:
            self.cb_cmap.addItem(c, c)
        idx = self.cb_cmap.findData(self.settings.map_cmap)
        if idx >= 0:
            self.cb_cmap.setCurrentIndex(idx)
        self.cb_cmap.currentIndexChanged.connect(
            lambda _: self._set("map_cmap", self.cb_cmap.currentData() or ""))
        f.addRow("Colormap", self.cb_cmap)

        vmin = QDoubleSpinBox(); vmin.setDecimals(3); vmin.setRange(-1e9, 1e9)
        vmin.setSpecialValueText("auto"); vmin.setValue(vmin.minimum())
        if self.settings.map_vmin == self.settings.map_vmin:
            vmin.setValue(self.settings.map_vmin)
        vmin.editingFinished.connect(
            lambda: self._set("map_vmin",
                              float("nan") if vmin.value() == vmin.minimum()
                              else vmin.value()))
        f.addRow("vmin", vmin)

        vmax = QDoubleSpinBox(); vmax.setDecimals(3); vmax.setRange(-1e9, 1e9)
        vmax.setSpecialValueText("auto"); vmax.setValue(vmax.minimum())
        if self.settings.map_vmax == self.settings.map_vmax:
            vmax.setValue(self.settings.map_vmax)
        vmax.editingFinished.connect(
            lambda: self._set("map_vmax",
                              float("nan") if vmax.value() == vmax.minimum()
                              else vmax.value()))
        f.addRow("vmax", vmax)

        self._vmin_sb = vmin
        self._vmax_sb = vmax

        # Raw DWI controls
        self.cb_raw_delta = QComboBox()
        self.cb_raw_delta.currentTextChanged.connect(
            lambda v: self._set("raw_delta", float(v) if v else 0.0))
        f.addRow("Raw Δ (ms)", self.cb_raw_delta)

        self.sb_raw_vol = QSpinBox(); self.sb_raw_vol.setRange(0, 96)
        self.sb_raw_vol.valueChanged.connect(
            lambda v: self._set("raw_vol_idx", int(v)))
        f.addRow("Raw volume #", self.sb_raw_vol)

        parent_layout.addWidget(g)

    def _build_geometry_box(self, parent_layout):
        g = QGroupBox("Slice")
        f = QFormLayout(g)

        self.cb_axis = QComboBox()
        self.cb_axis.addItems(["0 — sagittal", "1 — coronal", "2 — axial"])
        self.cb_axis.setCurrentIndex(self.settings.axis)
        self.cb_axis.currentIndexChanged.connect(
            lambda i: self._set("axis", int(i)))
        f.addRow("Axis", self.cb_axis)

        self.sb_slice = QSpinBox(); self.sb_slice.setRange(0, 1000)
        self.sb_slice.setValue(int(self.settings.slice_idx))
        self.sb_slice.valueChanged.connect(
            lambda v: self._set("slice_idx", int(v)))
        f.addRow("Slice", self.sb_slice)

        margin = QDoubleSpinBox(); margin.setRange(0, 100); margin.setDecimals(1)
        margin.setValue(self.settings.margin_mm)
        margin.valueChanged.connect(lambda v: self._set("margin_mm", float(v)))
        f.addRow("Zoom margin (mm)", margin)

        parent_layout.addWidget(g)

    def _build_preproc_box(self, parent_layout):
        """Pre-processing toggles (Rician debias, σ, S0 averaging).

        These affect the signal VOLUME the live matcher consumes — i.e.
        flipping them triggers a rebuild of ``signal_vol`` upstream. They
        default to the values the opening profile was fit with.
        """
        g = QGroupBox("Pre-processing")
        f = QFormLayout(g)

        self.cb_rician = QCheckBox("Rician noise-bias correction")
        self.cb_rician.setChecked(self.settings.rician_correct)
        self.cb_rician.stateChanged.connect(
            lambda _: self._set("rician_correct", self.cb_rician.isChecked()))
        f.addRow("", self.cb_rician)

        self.sb_sigma = QDoubleSpinBox()
        self.sb_sigma.setRange(0, 1e6)
        self.sb_sigma.setDecimals(2)
        self.sb_sigma.setSpecialValueText("auto")
        self.sb_sigma.setValue(0)
        if self.settings.noise_sigma == self.settings.noise_sigma:
            self.sb_sigma.setValue(float(self.settings.noise_sigma))
        self.sb_sigma.editingFinished.connect(
            lambda: self._set(
                "noise_sigma",
                float("nan") if self.sb_sigma.value() == self.sb_sigma.minimum()
                else float(self.sb_sigma.value())))
        f.addRow("σ (noise)", self.sb_sigma)

        self.cb_avg_s0 = QCheckBox("Average S0 across Δ scans")
        self.cb_avg_s0.setChecked(self.settings.avg_s0)
        self.cb_avg_s0.stateChanged.connect(
            lambda _: self._set("avg_s0", self.cb_avg_s0.isChecked()))
        f.addRow("", self.cb_avg_s0)

        self.lbl_sigma_used = QLabel("")
        self.lbl_sigma_used.setStyleSheet("color:#6b7280;font-size:10px;")
        f.addRow("", self.lbl_sigma_used)

        parent_layout.addWidget(g)

    def set_sigma_display(self, sigma: float | None):
        """Called by the viewer after rebuild so the user sees the σ in use."""
        if sigma is None:
            self.lbl_sigma_used.setText("")
        else:
            self.lbl_sigma_used.setText(f"σ in use: {sigma:.2f}")

    def _build_matcher_box(self, parent_layout):
        g = QGroupBox("Matcher")
        f = QFormLayout(g)

        self.sb_top = QSpinBox(); self.sb_top.setRange(1, 50)
        self.sb_top.setValue(self.settings.top_n)
        self.sb_top.valueChanged.connect(
            lambda v: self._set("top_n", int(v)))
        f.addRow("Top N", self.sb_top)

        self.cb_sort = QComboBox()
        for k, lbl in [("lin", "SSE (linear)"),
                        ("log", "SSE (log)"),
                        ("kio", "k_io"),
                        ("rho", "ρ"),
                        ("V", "V"),
                        ("vi", "v_i")]:
            self.cb_sort.addItem(lbl, k)
        idx = self.cb_sort.findData(self.settings.sort_by)
        if idx >= 0:
            self.cb_sort.setCurrentIndex(idx)
        self.cb_sort.currentIndexChanged.connect(
            lambda _: self._set("sort_by", self.cb_sort.currentData()))
        f.addRow("Rank by", self.cb_sort)

        vi_min = QDoubleSpinBox(); vi_min.setRange(0, 1); vi_min.setDecimals(3)
        vi_min.setSingleStep(0.05); vi_min.setValue(self.settings.vi_min)
        vi_min.valueChanged.connect(lambda v: self._set("vi_min", float(v)))
        f.addRow("v_i min", vi_min)

        vi_max = QDoubleSpinBox(); vi_max.setRange(0, 1); vi_max.setDecimals(3)
        vi_max.setSingleStep(0.05); vi_max.setValue(self.settings.vi_max)
        vi_max.valueChanged.connect(lambda v: self._set("vi_max", float(v)))
        f.addRow("v_i max", vi_max)

        rho_max = QDoubleSpinBox(); rho_max.setRange(0, 1e10); rho_max.setDecimals(0)
        rho_max.setSpecialValueText("none"); rho_max.setValue(0)
        if self.settings.rho_max == self.settings.rho_max:
            rho_max.setValue(self.settings.rho_max)
        rho_max.valueChanged.connect(
            lambda v: self._set("rho_max", float("nan") if v == 0 else float(v)))
        f.addRow("ρ max", rho_max)

        self.cb_fit_s0 = QCheckBox("Fit S0 as free parameter")
        self.cb_fit_s0.setChecked(self.settings.fit_s0)
        self.cb_fit_s0.stateChanged.connect(
            lambda _: self._set("fit_s0", self.cb_fit_s0.isChecked()))
        f.addRow("", self.cb_fit_s0)

        parent_layout.addWidget(g)

    def _build_signal_box(self, parent_layout):
        g = QGroupBox("Signal plot")
        f = QFormLayout(g)

        self.cb_delta = QComboBox()
        self.cb_delta.currentIndexChanged.connect(
            lambda i: self._set("delta_idx", int(i)))
        f.addRow("Δ displayed", self.cb_delta)

        self.cb_log_y = QCheckBox("Log y-axis")
        self.cb_log_y.setChecked(self.settings.log_y)
        self.cb_log_y.stateChanged.connect(
            lambda _: self._set("log_y", self.cb_log_y.isChecked()))
        f.addRow("", self.cb_log_y)

        f.addRow(QLabel("<b>Observed display</b>"))
        self.cb_obs_per_delta = QCheckBox("S / S0_Δ   (per-Δ, default)")
        self.cb_obs_per_delta.setChecked(self.settings.show_obs_per_delta)
        self.cb_obs_per_delta.stateChanged.connect(
            lambda _: self._set("show_obs_per_delta",
                                 self.cb_obs_per_delta.isChecked()))
        f.addRow("", self.cb_obs_per_delta)

        self.cb_obs_avg_s0 = QCheckBox("S / S0_avg  (Δ-averaged S0)")
        self.cb_obs_avg_s0.setChecked(self.settings.show_obs_avg_s0)
        self.cb_obs_avg_s0.stateChanged.connect(
            lambda _: self._set("show_obs_avg_s0",
                                 self.cb_obs_avg_s0.isChecked()))
        f.addRow("", self.cb_obs_avg_s0)

        self.cb_obs_fit_s0 = QCheckBox("S / S0_fit  (best match's fitted S0)")
        self.cb_obs_fit_s0.setChecked(self.settings.show_obs_fit_s0)
        self.cb_obs_fit_s0.stateChanged.connect(
            lambda _: self._set("show_obs_fit_s0",
                                 self.cb_obs_fit_s0.isChecked()))
        f.addRow("", self.cb_obs_fit_s0)

        parent_layout.addWidget(g)

    def _build_table_box(self, parent_layout):
        g = QGroupBox("Match table columns")
        v = QVBoxLayout(g)
        self._col_boxes: dict[str, QCheckBox] = {}
        for key, label in ALL_COLUMNS:
            cb = QCheckBox(label)
            cb.setChecked(key in self.settings.visible_cols)
            cb.stateChanged.connect(
                lambda _=None, k=key: self._toggle_col(k))
            v.addWidget(cb)
            self._col_boxes[key] = cb
        parent_layout.addWidget(g)

    # ---------------- change plumbing ----------------
    def _set(self, field: str, value):
        if self._block_signals:
            return
        setattr(self.settings, field, value)
        self.changed.emit(field)

    def _on_mode(self, idx: int):
        self._set("display_mode", "map" if idx == 0 else "raw")

    def _toggle_col(self, key: str):
        s = set(self.settings.visible_cols)
        if key in s:
            s.discard(key)
        else:
            s.add(key)
        self._set("visible_cols", s)

    # ---------------- external updates ----------------
    def update_raw_deltas(self, deltas: list[float]):
        self._block_signals = True
        self.cb_raw_delta.clear()
        for d in deltas:
            self.cb_raw_delta.addItem(f"{d:g}")
        if deltas:
            # set current to existing raw_delta or first
            cur = self.settings.raw_delta if self.settings.raw_delta in deltas else deltas[0]
            self.settings.raw_delta = cur
            self.cb_raw_delta.setCurrentText(f"{cur:g}")
        self._block_signals = False

    def update_signal_deltas(self, deltas: list[float]):
        self._block_signals = True
        self.cb_delta.clear()
        for d in deltas:
            self.cb_delta.addItem(f"Δ = {d:g} ms")
        if self.settings.delta_idx >= len(deltas):
            self.settings.delta_idx = 0
        self.cb_delta.setCurrentIndex(self.settings.delta_idx)
        self._block_signals = False

    def update_slice_range(self, n_slices: int):
        self._block_signals = True
        self.sb_slice.setRange(0, max(0, n_slices - 1))
        self.sb_slice.setValue(int(min(self.settings.slice_idx, n_slices - 1)))
        self._block_signals = False

    def update_vol_range(self, n_vols: int):
        self._block_signals = True
        self.sb_raw_vol.setRange(0, max(0, n_vols - 1))
        self.sb_raw_vol.setValue(int(min(self.settings.raw_vol_idx, n_vols - 1)))
        self._block_signals = False

    def sync_slice(self, slice_idx: int):
        self._block_signals = True
        self.sb_slice.setValue(int(slice_idx))
        self._block_signals = False
