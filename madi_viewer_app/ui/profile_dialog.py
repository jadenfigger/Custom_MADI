"""Create / edit an OutputProfile.

Covers: name + description, library & mask pickers, any number of
``delta:path`` DWI entries, output directory, and every fitting option
that ``scripts/fit_data.py`` understands.

Input validation happens on ``accept()`` and a red status line explains
what's missing.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (QDialog, QDialogButtonBox, QFormLayout,
                              QGridLayout, QHBoxLayout, QLabel, QLineEdit,
                              QPushButton, QVBoxLayout, QWidget, QSpinBox,
                              QDoubleSpinBox, QCheckBox, QTabWidget,
                              QFileDialog, QTextEdit, QListWidget,
                              QListWidgetItem, QMessageBox, QComboBox)

from ..app_state import OutputProfile
from ..data import _cache
from ..constants import PROFILE_COLORS


class _PathPicker(QWidget):
    """Text line + Browse button."""
    def __init__(self, caption: str, file_mode: str = "file",
                 patterns: str = "All files (*)"):
        super().__init__()
        self.caption = caption
        self.file_mode = file_mode
        self.patterns = patterns
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.edit = QLineEdit()
        self.btn = QPushButton("Browse…")
        self.btn.clicked.connect(self._pick)
        layout.addWidget(self.edit, 1)
        layout.addWidget(self.btn)

    def _pick(self):
        if self.file_mode == "dir":
            p = QFileDialog.getExistingDirectory(self, self.caption, self.edit.text())
        else:
            p, _ = QFileDialog.getOpenFileName(
                self, self.caption, self.edit.text(), self.patterns)
        if p:
            self.edit.setText(p)

    def text(self) -> str:
        return self.edit.text().strip()

    def setText(self, t: str):
        self.edit.setText(t or "")


class _InputList(QWidget):
    """Editable list of (delta_ms, path) rows."""
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.list = QListWidget()
        self.list.setMinimumHeight(110)
        layout.addWidget(self.list, 1)

        row = QHBoxLayout()
        self.sb_delta = QDoubleSpinBox()
        self.sb_delta.setRange(0.01, 1000.0); self.sb_delta.setDecimals(2)
        self.sb_delta.setValue(15.0)
        self.sb_delta.setSuffix(" ms")
        self.edit_path = QLineEdit()
        self.edit_path.setPlaceholderText("path to NIfTI …")
        btn_browse = QPushButton("Browse…")
        btn_browse.clicked.connect(self._browse)
        btn_add = QPushButton("Add")
        btn_add.clicked.connect(self._add)
        btn_rm = QPushButton("Remove")
        btn_rm.clicked.connect(self._remove)
        row.addWidget(QLabel("Δ")); row.addWidget(self.sb_delta)
        row.addWidget(self.edit_path, 1)
        row.addWidget(btn_browse)
        row.addWidget(btn_add); row.addWidget(btn_rm)
        layout.addLayout(row)

    def _browse(self):
        p, _ = QFileDialog.getOpenFileName(
            self, "Pick DWI NIfTI", self.edit_path.text(),
            "NIfTI (*.nii *.nii.gz);;All files (*)")
        if p:
            self.edit_path.setText(p)

    def _add(self):
        d = float(self.sb_delta.value())
        p = self.edit_path.text().strip()
        if not p:
            return
        for i in range(self.list.count()):
            if self.list.item(i).data(Qt.UserRole)[0] == d:
                self.list.takeItem(i); break
        item = QListWidgetItem(f"Δ = {d:g} ms    {p}")
        item.setData(Qt.UserRole, (d, p))
        self.list.addItem(item)
        self.edit_path.clear()
        self._sort()

    def _remove(self):
        for it in self.list.selectedItems():
            self.list.takeItem(self.list.row(it))

    def _sort(self):
        items = []
        while self.list.count():
            items.append(self.list.takeItem(0))
        items.sort(key=lambda it: it.data(Qt.UserRole)[0])
        for it in items:
            self.list.addItem(it)

    def set_items(self, inputs: list):
        self.list.clear()
        for d, p in inputs:
            it = QListWidgetItem(f"Δ = {float(d):g} ms    {p}")
            it.setData(Qt.UserRole, (float(d), p))
            self.list.addItem(it)
        self._sort()

    def items(self) -> list:
        out = []
        for i in range(self.list.count()):
            d, p = self.list.item(i).data(Qt.UserRole)
            out.append([float(d), str(p)])
        return out


class ProfileDialog(QDialog):
    """Modal create/edit dialog. Returns an OutputProfile on accept."""

    def __init__(self, profile: Optional[OutputProfile] = None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Output profile" if profile else "New output profile")
        self.setMinimumWidth(700)
        self.setMinimumHeight(640)

        self.profile = profile or OutputProfile(color=PROFILE_COLORS[0])

        tabs = QTabWidget()
        tabs.addTab(self._build_basics(), "Basics")
        tabs.addTab(self._build_fitting(), "Fitting options")
        tabs.addTab(self._build_info(), "Library info")

        self.status = QLabel("")
        self.status.setStyleSheet("color:#dc2626;")

        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btns.accepted.connect(self._on_accept)
        btns.rejected.connect(self.reject)

        root = QVBoxLayout(self)
        root.addWidget(tabs, 1)
        root.addWidget(self.status)
        root.addWidget(btns)

        self._load_from_profile()

    # ---------------- panels ----------------
    def _build_basics(self) -> QWidget:
        w = QWidget(); f = QFormLayout(w)
        self.edit_name = QLineEdit()
        f.addRow("Name", self.edit_name)

        self.edit_desc = QTextEdit(); self.edit_desc.setMaximumHeight(70)
        f.addRow("Description", self.edit_desc)

        self.pick_lib = _PathPicker("Library (.npz)",
                                     patterns="NumPy npz (*.npz);;All files (*)")
        f.addRow("Library", self.pick_lib)

        self.pick_mask = _PathPicker("Brain mask NIfTI",
                                      patterns="NIfTI (*.nii *.nii.gz);;All files (*)")
        f.addRow("Mask", self.pick_mask)

        self.input_list = _InputList()
        f.addRow("DWI inputs (Δ:path)", self.input_list)

        self.pick_out = _PathPicker("Output directory", file_mode="dir")
        f.addRow("Output dir", self.pick_out)

        self.cb_color = QComboBox()
        for c in PROFILE_COLORS:
            self.cb_color.addItem(c, c)
        f.addRow("Plot color", self.cb_color)

        return w

    def _build_fitting(self) -> QWidget:
        w = QWidget(); f = QFormLayout(w)

        self.cb_rician = QCheckBox("Apply Rician bias correction")
        f.addRow("", self.cb_rician)

        self.sb_sigma = QDoubleSpinBox()
        self.sb_sigma.setRange(0, 1e6); self.sb_sigma.setDecimals(3)
        self.sb_sigma.setSpecialValueText("auto (background-estimated)")
        f.addRow("Noise σ", self.sb_sigma)

        self.cb_avg_s0 = QCheckBox(
            "Average b=0 volumes across Δ scans for normalization")
        f.addRow("", self.cb_avg_s0)

        self.cb_fit_s0 = QCheckBox(
            "Fit S0 as a free per-voxel parameter (analytic L2 projection)")
        f.addRow("", self.cb_fit_s0)

        self.cb_logspace = QCheckBox(
            "Rank library matches in log-space (matches the fitting pipeline "
            "when it was run with log-space)")
        f.addRow("", self.cb_logspace)

        self.sb_vi_min = QDoubleSpinBox(); self.sb_vi_min.setRange(0, 1)
        self.sb_vi_min.setDecimals(3); self.sb_vi_min.setSingleStep(0.05)
        f.addRow("v_i min", self.sb_vi_min)

        self.sb_vi_max = QDoubleSpinBox(); self.sb_vi_max.setRange(0, 1)
        self.sb_vi_max.setDecimals(3); self.sb_vi_max.setSingleStep(0.05)
        f.addRow("v_i max", self.sb_vi_max)

        self.sb_rho_max = QDoubleSpinBox()
        self.sb_rho_max.setRange(0, 1e10); self.sb_rho_max.setDecimals(0)
        self.sb_rho_max.setSpecialValueText("no upper bound")
        f.addRow("ρ max (cells/μL)", self.sb_rho_max)

        return w

    def _build_info(self) -> QWidget:
        w = QWidget(); v = QVBoxLayout(w)
        self.info_text = QTextEdit()
        self.info_text.setReadOnly(True)
        self.info_text.setPlaceholderText(
            "Pick a library to see its metadata here.")
        btn = QPushButton("Inspect library now")
        btn.clicked.connect(self._inspect_library)
        v.addWidget(btn)
        v.addWidget(self.info_text, 1)
        return w

    # ---------------- profile <-> form ----------------
    def _load_from_profile(self):
        p = self.profile
        self.edit_name.setText(p.name if p.name else "")
        self.edit_desc.setPlainText(p.description or "")
        self.pick_lib.setText(p.library_path)
        self.pick_mask.setText(p.mask_path)
        self.pick_out.setText(p.output_dir)
        self.input_list.set_items([[float(d), s] for d, s in p.inputs])
        idx = max(0, self.cb_color.findData(p.color))
        self.cb_color.setCurrentIndex(idx)
        self.cb_rician.setChecked(p.rician_correct)
        self.sb_sigma.setValue(0 if p.noise_sigma is None else float(p.noise_sigma))
        self.cb_avg_s0.setChecked(p.avg_s0)
        self.cb_fit_s0.setChecked(p.fit_s0)
        self.cb_logspace.setChecked(p.log_space)
        self.sb_vi_min.setValue(float(p.vi_min))
        self.sb_vi_max.setValue(float(p.vi_max))
        self.sb_rho_max.setValue(0 if p.rho_max is None else float(p.rho_max))

    def _populate_profile(self) -> OutputProfile:
        p = self.profile
        p.name = self.edit_name.text().strip() or "Untitled profile"
        p.description = self.edit_desc.toPlainText().strip()
        p.library_path = self.pick_lib.text()
        p.mask_path = self.pick_mask.text()
        p.output_dir = self.pick_out.text()
        p.inputs = self.input_list.items()
        p.color = self.cb_color.currentData() or PROFILE_COLORS[0]
        p.rician_correct = self.cb_rician.isChecked()
        p.noise_sigma = None if self.sb_sigma.value() == 0 else float(self.sb_sigma.value())
        p.avg_s0 = self.cb_avg_s0.isChecked()
        p.fit_s0 = self.cb_fit_s0.isChecked()
        p.log_space = self.cb_logspace.isChecked()
        p.vi_min = float(self.sb_vi_min.value())
        p.vi_max = float(self.sb_vi_max.value())
        rm = self.sb_rho_max.value()
        p.rho_max = None if rm == 0 else float(rm)
        return p

    # ---------------- actions ----------------
    def _inspect_library(self):
        lib_path = self.pick_lib.text()
        if not lib_path or not Path(lib_path).exists():
            self.info_text.setPlainText(
                "Library path is empty or does not exist.")
            return
        try:
            bundle = _cache.get_library(lib_path)
        except Exception as e:
            self.info_text.setPlainText(f"Failed to load library:\n{e}")
            return
        meta = bundle["meta"]
        kios = sorted(set(bundle["kios"].tolist()))
        rhos = sorted(set(bundle["rhos"].tolist()))
        Vs = sorted(set(bundle["Vs"].tolist()))
        self.info_text.setPlainText(
            f"Library: {lib_path}\n"
            f"Entries: {len(bundle['lib'])}\n"
            f"Δ values: {list(meta['deltas'])}\n"
            f"b-values per Δ: {meta['n_b']}\n"
            f"k_io ({len(kios)}): {kios}\n"
            f"ρ    ({len(rhos)}): {[f'{r/1e3:.0f}k' for r in rhos]}\n"
            f"V    ({len(Vs)}):   {[f'{v:.2f}' for v in Vs]}")

    def _on_accept(self):
        prof = self._populate_profile()
        miss = []
        if not prof.library_path:
            miss.append("library path is required")
        elif not Path(prof.library_path).exists():
            miss.append(f"library not found: {prof.library_path}")
        if not prof.mask_path:
            miss.append("mask path is required")
        elif not Path(prof.mask_path).exists():
            miss.append(f"mask not found: {prof.mask_path}")
        if not prof.inputs:
            miss.append("at least one DWI input is required")
        else:
            for d, p in prof.inputs:
                if not Path(p).exists():
                    miss.append(f"DWI not found (Δ={d}): {p}")
        if not prof.output_dir:
            miss.append("output directory is required")

        if prof.fit_s0 and prof.avg_s0:
            miss.append("cannot combine avg-s0 and fit-s0 (mutually exclusive)")

        if miss:
            self.status.setText(" · ".join(miss))
            return
        self.accept()

    def result_profile(self) -> OutputProfile:
        return self.profile
