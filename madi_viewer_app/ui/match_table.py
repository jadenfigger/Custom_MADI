"""Top-N library-match table with multi-selection and column toggling."""
from __future__ import annotations

from typing import Iterable, Optional

from PyQt5.QtCore import Qt, pyqtSignal, QItemSelection, QItemSelectionModel
from PyQt5.QtGui import QColor, QBrush
from PyQt5.QtWidgets import (QTableWidget, QTableWidgetItem,
                              QAbstractItemView, QHeaderView)

from ..constants import MATCH_COLORS


ALL_COLUMNS: list[tuple[str, str]] = [
    ("rank",    "#"),
    ("kio",     "k_io (s⁻¹)"),
    ("rho",     "ρ (k/μL)"),
    ("V",       "V (pL)"),
    ("vi",      "v_i"),
    ("sse_lin", "SSE (lin)"),
    ("sse_log", "SSE (log)"),
    ("s0_fit",  "S0 fit"),
]
DEFAULT_VISIBLE = {"rank", "kio", "rho", "V", "vi", "sse_lin", "sse_log"}


class MatchTable(QTableWidget):

    # set of ranks (1-based) currently selected for overlay.
    # Named ``ranksChanged`` rather than ``selectionChanged`` because
    # QAbstractItemView already has a protected ``selectionChanged``
    # virtual slot — overriding its name with a pyqtSignal causes Qt's
    # C++ side to raise ``TypeError: native Qt signal is not callable``.
    ranksChanged = pyqtSignal(set)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.setAlternatingRowColors(True)
        self.verticalHeader().setVisible(False)
        self.horizontalHeader().setStretchLastSection(True)
        self.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self._visible_cols: set[str] = set(DEFAULT_VISIBLE)
        self._sort_key: str = "sse_lin"
        self._matches: list = []
        self._suppress_signal = False
        self.itemSelectionChanged.connect(self._emit_selection)

    # ---------------- configuration ----------------
    def set_visible_columns(self, keys: Iterable[str]):
        self._visible_cols = set(keys)
        self._render()

    def visible_columns(self) -> set[str]:
        return set(self._visible_cols)

    def set_sort_key(self, key: str):
        self._sort_key = key
        self._render()

    # ---------------- data ----------------
    def set_matches(self, matches: list, selected_ranks: Optional[set[int]] = None):
        self._matches = list(matches or [])
        self._render()
        if selected_ranks is not None:
            self.set_selected_ranks(selected_ranks)

    def clear_matches(self):
        self._matches = []
        self._render()

    def selected_ranks(self) -> set[int]:
        out: set[int] = set()
        for idx in self.selectionModel().selectedRows():
            out.add(self._matches[idx.row()].rank)
        return out

    def set_selected_ranks(self, ranks: set[int]):
        self._suppress_signal = True
        try:
            sel_model = self.selectionModel()
            if sel_model is None:
                return
            sel = QItemSelection()
            n_cols = self.columnCount()
            if n_cols == 0:
                sel_model.clearSelection()
                return
            for row, m in enumerate(self._matches):
                if m.rank in ranks:
                    top_left = self.model().index(row, 0)
                    bottom_right = self.model().index(row, n_cols - 1)
                    sel.select(top_left, bottom_right)
            sel_model.select(
                sel,
                QItemSelectionModel.ClearAndSelect | QItemSelectionModel.Rows,
            )
        finally:
            self._suppress_signal = False

    # ---------------- internal ----------------
    def _render(self):
        cols = [c for c in ALL_COLUMNS if c[0] in self._visible_cols]
        self.setColumnCount(len(cols))
        self.setHorizontalHeaderLabels([c[1] for c in cols])

        for col, (key, _label) in enumerate(cols):
            if key == self._sort_key:
                header_item = self.horizontalHeaderItem(col)
                if header_item is not None:
                    header_item.setBackground(QBrush(QColor("#0f766e")))

        self.setRowCount(len(self._matches))
        for row, m in enumerate(self._matches):
            for col, (key, _label) in enumerate(cols):
                item = QTableWidgetItem(self._fmt(key, m))
                item.setTextAlignment(Qt.AlignCenter)
                color = QColor(MATCH_COLORS[(m.rank - 1) % len(MATCH_COLORS)])
                color.setAlpha(40)
                item.setBackground(QBrush(color))
                self.setItem(row, col, item)
        self.resizeColumnsToContents()

    def _fmt(self, key: str, m) -> str:
        if key == "rank":    return f"{m.rank}"
        if key == "kio":     return f"{m.kio:.1f}"
        if key == "rho":     return f"{m.rho / 1e3:.0f}"
        if key == "V":       return f"{m.V:.2f}"
        if key == "vi":      return f"{m.vi:.3f}"
        if key == "sse_lin": return f"{m.sse_lin:.5f}"
        if key == "sse_log": return f"{m.sse_log:.5f}"
        if key == "s0_fit":  return "—" if m.s0_fit is None else f"{m.s0_fit:.3g}"
        return ""

    def _emit_selection(self):
        if self._suppress_signal:
            return
        self.ranksChanged.emit(self.selected_ranks())
