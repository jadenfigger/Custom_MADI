"""Main application window.

Left dock: profile browser (create / duplicate / edit / delete / open /
run fit / compare). Right: a QTabWidget hosting ViewerWidget instances
and CompareWidget instances. Menu bar + workspace persistence.
"""
from __future__ import annotations

import base64
from pathlib import Path
from typing import Optional

from PyQt5.QtCore import Qt, QByteArray
from PyQt5.QtGui import QKeySequence
from PyQt5.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                              QListWidget, QListWidgetItem, QPushButton,
                              QLabel, QTabWidget, QDockWidget, QMessageBox,
                              QInputDialog, QAction, QFileDialog,
                              QAbstractItemView)

from ..app_state import Workspace, OutputProfile, default_workspace_dir
from .profile_dialog import ProfileDialog
from .fitting_dialog import FittingDialog
from .viewer_widget import ViewerWidget
from .compare_widget import CompareWidget


class ProfileSidebar(QWidget):
    """Left dock: list of profiles + action buttons."""

    def __init__(self, window: "MainWindow"):
        super().__init__()
        self.window = window

        self.list = QListWidget()
        self.list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.list.itemDoubleClicked.connect(self._on_double_click)

        v = QVBoxLayout(self)
        v.setContentsMargins(4, 4, 4, 4)

        top = QHBoxLayout()
        btn_new = QPushButton("+ New")
        btn_new.setObjectName("primary")
        btn_new.clicked.connect(self._create)
        btn_refresh = QPushButton("↻")
        btn_refresh.setToolTip("Refresh (re-read workspace from disk)")
        btn_refresh.clicked.connect(self.refresh)
        top.addWidget(btn_new, 1)
        top.addWidget(btn_refresh)

        v.addLayout(top)
        v.addWidget(self.list, 1)

        btn_row_1 = QHBoxLayout()
        btn_open = QPushButton("Open")
        btn_open.clicked.connect(self._open_selected)
        btn_compare = QPushButton("Compare")
        btn_compare.clicked.connect(self._compare_selected)
        btn_row_1.addWidget(btn_open); btn_row_1.addWidget(btn_compare)
        v.addLayout(btn_row_1)

        btn_row_2 = QHBoxLayout()
        btn_edit = QPushButton("Edit")
        btn_edit.clicked.connect(self._edit_selected)
        btn_dup = QPushButton("Duplicate")
        btn_dup.clicked.connect(self._duplicate_selected)
        btn_row_2.addWidget(btn_edit); btn_row_2.addWidget(btn_dup)
        v.addLayout(btn_row_2)

        btn_row_3 = QHBoxLayout()
        btn_fit = QPushButton("Run fit")
        btn_fit.clicked.connect(self._fit_selected)
        btn_del = QPushButton("Delete")
        btn_del.setObjectName("danger")
        btn_del.clicked.connect(self._delete_selected)
        btn_row_3.addWidget(btn_fit); btn_row_3.addWidget(btn_del)
        v.addLayout(btn_row_3)

        self.refresh()

    # ---- refresh ----
    def refresh(self):
        self.list.clear()
        for prof in self.window.workspace.list_profiles():
            item = QListWidgetItem(self._format(prof))
            item.setData(Qt.UserRole, prof.id)
            item.setForeground(Qt.black)
            self.list.addItem(item)

    def _format(self, p: OutputProfile) -> str:
        mark = "●" if p.detect_fitted() else "○"
        return f"{mark}  {p.name}   [{', '.join(f'{d:g}' for d,_ in p.inputs)}]"

    # ---- actions ----
    def _selected_ids(self) -> list[str]:
        return [it.data(Qt.UserRole) for it in self.list.selectedItems()]

    def _create(self):
        dlg = ProfileDialog(parent=self.window)
        if dlg.exec_() == dlg.Accepted:
            prof = dlg.result_profile()
            self.window.workspace.save_profile(prof)
            self.refresh()
            self.window.open_profile(prof.id)

    def _open_selected(self):
        for pid in self._selected_ids():
            self.window.open_profile(pid)

    def _compare_selected(self):
        ids = self._selected_ids()
        if len(ids) < 2:
            QMessageBox.information(
                self.window, "Compare",
                "Select at least two profiles (Ctrl-click to multi-select).")
            return
        self.window.open_compare(ids)

    def _edit_selected(self):
        ids = self._selected_ids()
        if not ids:
            return
        prof = self.window.workspace.load_profile(ids[0])
        dlg = ProfileDialog(profile=prof, parent=self.window)
        if dlg.exec_() == dlg.Accepted:
            self.window.workspace.save_profile(dlg.result_profile())
            self.refresh()

    def _duplicate_selected(self):
        for pid in self._selected_ids():
            self.window.workspace.duplicate_profile(pid)
        self.refresh()

    def _fit_selected(self):
        for pid in self._selected_ids():
            prof = self.window.workspace.load_profile(pid)
            miss = prof.required_paths_missing()
            if miss:
                QMessageBox.warning(
                    self.window, "Can't fit",
                    "Missing required paths:\n  " + "\n  ".join(miss))
                continue
            dlg = FittingDialog(prof, parent=self.window)
            dlg.exec_()
            if dlg.succeeded():
                self.window.workspace.save_profile(prof)
                self.refresh()

    def _delete_selected(self):
        ids = self._selected_ids()
        if not ids:
            return
        reply = QMessageBox.question(
            self.window, "Delete profiles?",
            f"Delete {len(ids)} profile(s)? Output NIfTI files are not "
            f"touched; only the profile definitions will be removed.")
        if reply != QMessageBox.Yes:
            return
        for pid in ids:
            self.window.workspace.delete_profile(pid)
        self.refresh()

    def _on_double_click(self, item: QListWidgetItem):
        self.window.open_profile(item.data(Qt.UserRole))


class MainWindow(QMainWindow):

    def __init__(self, workspace: Workspace):
        super().__init__()
        self.workspace = workspace
        self.setWindowTitle("MADI Viewer")
        self.setMinimumSize(1400, 900)

        self._open_viewers: dict[str, ViewerWidget] = {}
        self._compare_views: dict[tuple, CompareWidget] = {}

        # Central tabs
        self.tabs = QTabWidget()
        self.tabs.setTabsClosable(True)
        self.tabs.tabCloseRequested.connect(self._close_tab)
        self.tabs.currentChanged.connect(self._on_tab_changed)
        self.setCentralWidget(self.tabs)

        # Sidebar dock
        self.sidebar = ProfileSidebar(self)
        dock = QDockWidget("Profiles", self)
        dock.setWidget(self.sidebar)
        dock.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)
        self.addDockWidget(Qt.LeftDockWidgetArea, dock)
        self._profiles_dock = dock

        # Menu
        self._build_menu()

        # Status bar
        self.statusBar().showMessage(
            f"Workspace: {workspace.ws_path}")

        # Restore window geometry
        self._restore_geometry()

        # Re-open tabs from last session
        for pid in list(workspace.open_profile_ids):
            self.open_profile(pid)
        if workspace.compare_profile_ids:
            try:
                self.open_compare(workspace.compare_profile_ids)
            except Exception:
                pass

    # ================================================================
    #  Menu
    # ================================================================
    def _build_menu(self):
        mb = self.menuBar()

        file_menu = mb.addMenu("&File")

        act_new = QAction("&New profile…", self)
        act_new.setShortcut(QKeySequence.New)
        act_new.triggered.connect(self.sidebar._create)
        file_menu.addAction(act_new)

        act_switch = QAction("Switch workspace…", self)
        act_switch.triggered.connect(self._switch_workspace)
        file_menu.addAction(act_switch)

        file_menu.addSeparator()
        act_quit = QAction("Quit", self)
        act_quit.setShortcut(QKeySequence.Quit)
        act_quit.triggered.connect(self.close)
        file_menu.addAction(act_quit)

        view_menu = mb.addMenu("&View")
        act_toggle_side = self._profiles_dock.toggleViewAction()
        act_toggle_side.setText("Toggle profile sidebar")
        view_menu.addAction(act_toggle_side)

        help_menu = mb.addMenu("&Help")
        act_about = QAction("About", self)
        act_about.triggered.connect(self._about)
        help_menu.addAction(act_about)
        act_help = QAction("Controls", self)
        act_help.triggered.connect(self._controls)
        help_menu.addAction(act_help)

    def _about(self):
        QMessageBox.information(
            self, "About MADI Viewer",
            "MADI Viewer\n\n"
            "Persistent diagnostic tool for MADI fitting outputs. Create\n"
            "output profiles (library + inputs + fitting options), run fits\n"
            "from within the app, and open multiple profiles side-by-side\n"
            "to compare voxels and parameter maps.\n\n"
            f"Workspace: {self.workspace.ws_path}")

    def _controls(self):
        QMessageBox.information(
            self, "Controls",
            "Keyboard shortcuts:\n"
            "  Click        — select voxel\n"
            "  ← / →        — cycle Δ in signal plot\n"
            "  ↑ / ↓        — cycle top-N matches\n"
            "  PgUp / PgDn  — change slice\n"
            "  M            — next parameter map\n"
            "  L            — toggle log y-axis on signal plot\n\n"
            "In the match table, Ctrl-click to overlay multiple matches on\n"
            "the signal plot at once.")

    # ================================================================
    #  Tabs
    # ================================================================
    def open_profile(self, pid: str):
        if pid in self._open_viewers:
            w = self._open_viewers[pid]
            self.tabs.setCurrentWidget(w)
            return w
        try:
            prof = self.workspace.load_profile(pid)
        except FileNotFoundError:
            self.statusBar().showMessage(f"Profile {pid} missing")
            return None
        w = ViewerWidget(prof, self)
        self._open_viewers[pid] = w
        self.tabs.addTab(w, prof.name)
        self.tabs.setCurrentWidget(w)
        return w

    def open_compare(self, pids: list[str]):
        key = tuple(sorted(pids))
        if key in self._compare_views:
            self.tabs.setCurrentWidget(self._compare_views[key])
            return self._compare_views[key]
        profs = [self.workspace.load_profile(p) for p in pids]
        w = CompareWidget(profs, self)
        self._compare_views[key] = w
        label = "Compare: " + " · ".join(p.name[:14] for p in profs)
        self.tabs.addTab(w, label)
        self.tabs.setCurrentWidget(w)
        return w

    def _close_tab(self, idx: int):
        w = self.tabs.widget(idx)
        self.tabs.removeTab(idx)
        for pid, v in list(self._open_viewers.items()):
            if v is w:
                self._open_viewers.pop(pid); break
        for k, v in list(self._compare_views.items()):
            if v is w:
                self._compare_views.pop(k); break
        w.deleteLater()

    def _on_tab_changed(self, idx: int):
        w = self.tabs.widget(idx)
        if isinstance(w, ViewerWidget):
            self.workspace.active_profile_id = w.profile.id

    # ================================================================
    #  Workspace
    # ================================================================
    def _switch_workspace(self):
        d = QFileDialog.getExistingDirectory(
            self, "Pick workspace directory",
            str(self.workspace.ws_path))
        if not d:
            return
        # Close tabs, save current, then reload
        self._save_state()
        while self.tabs.count():
            self._close_tab(0)
        self._open_viewers.clear()
        self._compare_views.clear()
        self.workspace = Workspace.load(Path(d))
        self.sidebar.window = self
        self.sidebar.refresh()
        self.statusBar().showMessage(f"Workspace: {self.workspace.ws_path}")

    def _save_state(self):
        self.workspace.open_profile_ids = [
            v.profile.id for v in self._open_viewers.values()]
        # Only one compare view is remembered (the most recent)
        if self._compare_views:
            last = list(self._compare_views.keys())[-1]
            self.workspace.compare_profile_ids = list(last)
        else:
            self.workspace.compare_profile_ids = []
        # Window geometry
        self.workspace.window_geometry = \
            base64.b64encode(bytes(self.saveGeometry())).decode()
        self.workspace.window_state = \
            base64.b64encode(bytes(self.saveState())).decode()
        self.workspace.save()

    def _restore_geometry(self):
        if self.workspace.window_geometry:
            try:
                self.restoreGeometry(QByteArray(
                    base64.b64decode(self.workspace.window_geometry)))
            except Exception:
                pass
        if self.workspace.window_state:
            try:
                self.restoreState(QByteArray(
                    base64.b64decode(self.workspace.window_state)))
            except Exception:
                pass

    def closeEvent(self, ev):
        self._save_state()
        return super().closeEvent(ev)
