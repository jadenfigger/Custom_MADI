"""Matplotlib + Qt stylesheets shared across widgets."""
from __future__ import annotations

import matplotlib as mpl

MPL_RC = {
    "font.family":       "sans-serif",
    "font.size":         9,
    "axes.titlesize":    10,
    "axes.titleweight":  "bold",
    "axes.labelsize":    9,
    "axes.linewidth":    0.8,
    "axes.edgecolor":    "#cbd5e1",
    "axes.facecolor":    "#ffffff",
    "axes.grid":         True,
    "grid.color":        "#e5e7eb",
    "grid.linewidth":    0.6,
    "grid.alpha":        0.9,
    "legend.fontsize":   8,
    "legend.framealpha": 0.9,
    "legend.edgecolor":  "#cbd5e1",
    "xtick.labelsize":   8,
    "ytick.labelsize":   8,
    "figure.facecolor":  "white",
}


def apply_mpl_style() -> None:
    mpl.rcParams.update(MPL_RC)


QT_STYLE = """
QMainWindow, QDialog, QWidget {
    background-color: #f8fafc;
    color: #111827;
}
QGroupBox {
    border: 1px solid #cbd5e1;
    border-radius: 4px;
    margin-top: 10px;
    padding-top: 8px;
    font-weight: bold;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 8px;
    padding: 0 4px;
    color: #374151;
}
QPushButton {
    background-color: #e2e8f0;
    border: 1px solid #94a3b8;
    padding: 5px 12px;
    border-radius: 3px;
}
QPushButton:hover { background-color: #cbd5e1; }
QPushButton:pressed { background-color: #94a3b8; }
QPushButton:disabled { color: #9ca3af; background-color: #f1f5f9; }
QPushButton#primary {
    background-color: #2563eb;
    color: white;
    border-color: #1e40af;
    font-weight: bold;
}
QPushButton#primary:hover { background-color: #1d4ed8; }
QPushButton#danger {
    background-color: #dc2626;
    color: white;
    border-color: #991b1b;
}
QPushButton#danger:hover { background-color: #b91c1c; }

QLineEdit, QDoubleSpinBox, QSpinBox, QComboBox, QPlainTextEdit {
    background: white;
    border: 1px solid #cbd5e1;
    padding: 3px;
    border-radius: 3px;
}
QTabBar::tab {
    background: #e2e8f0;
    padding: 6px 12px;
    border: 1px solid #cbd5e1;
    border-bottom: none;
    border-top-left-radius: 3px;
    border-top-right-radius: 3px;
}
QTabBar::tab:selected {
    background: white;
    font-weight: bold;
}
QTableWidget, QTableView {
    gridline-color: #e5e7eb;
    selection-background-color: #dbeafe;
    selection-color: #111827;
    background: white;
}
QHeaderView::section {
    background-color: #1f2937;
    color: white;
    padding: 4px;
    border: 1px solid #374151;
    font-weight: bold;
}
QListWidget {
    background: white;
    border: 1px solid #cbd5e1;
}
QListWidget::item:selected {
    background: #dbeafe;
    color: #111827;
}
QDockWidget::title {
    background: #1f2937;
    color: white;
    padding: 4px;
}
"""
