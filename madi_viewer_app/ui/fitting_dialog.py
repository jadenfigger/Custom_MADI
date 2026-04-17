"""Run-fit dialog: streams stdout of scripts/fit_data.py."""
from __future__ import annotations

from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QPlainTextEdit,
                              QPushButton, QLabel)

from ..app_state import OutputProfile
from ..fitting import build_fit_command, run_fit_in_thread


class FittingDialog(QDialog):
    def __init__(self, profile: OutputProfile, parent=None):
        super().__init__(parent)
        self.profile = profile
        self.setWindowTitle(f"Fit — {profile.name}")
        self.setMinimumSize(820, 500)

        self.status = QLabel("Ready.")
        self.cmd_label = QLabel(
            " ".join(build_fit_command(profile)))
        self.cmd_label.setWordWrap(True)
        self.cmd_label.setStyleSheet("color:#475569; font-family:monospace;")

        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setFont(QFont("Consolas", 9))
        self.log.setStyleSheet("background:#0f172a; color:#e2e8f0;")

        self.btn_run = QPushButton("Run fit")
        self.btn_run.setObjectName("primary")
        self.btn_run.clicked.connect(self.start)

        self.btn_cancel = QPushButton("Cancel")
        self.btn_cancel.setEnabled(False)
        self.btn_cancel.clicked.connect(self.cancel)

        self.btn_close = QPushButton("Close")
        self.btn_close.clicked.connect(self.accept)

        hb = QHBoxLayout()
        hb.addWidget(self.status, 1)
        hb.addWidget(self.btn_run)
        hb.addWidget(self.btn_cancel)
        hb.addWidget(self.btn_close)

        v = QVBoxLayout(self)
        v.addWidget(QLabel("Command:"))
        v.addWidget(self.cmd_label)
        v.addWidget(self.log, 1)
        v.addLayout(hb)

        self._thread = None
        self._worker = None
        self._succeeded = False

    # ---- actions ----
    def start(self):
        self.log.clear()
        self.status.setText("Running…")
        self.btn_run.setEnabled(False)
        self.btn_cancel.setEnabled(True)
        self._thread, self._worker = run_fit_in_thread(
            self.profile,
            on_line=self._append,
            on_finished=self._on_finished,
            on_error=self._on_error,
        )
        self._thread.start()

    def cancel(self):
        if self._worker:
            self._worker.cancel()
            self.status.setText("Cancelling…")

    # ---- callbacks ----
    def _append(self, line: str):
        self.log.appendPlainText(line)
        self.log.verticalScrollBar().setValue(
            self.log.verticalScrollBar().maximum())

    def _on_error(self, msg: str):
        self._append(f"[ERROR] {msg}")

    def _on_finished(self, rc: int):
        self.btn_run.setEnabled(True)
        self.btn_cancel.setEnabled(False)
        if rc == 0:
            self._succeeded = True
            self.status.setText("Done. Fit completed successfully.")
            self._append("\n✓ Fit finished successfully.")
        else:
            self.status.setText(f"Failed (rc={rc}).")
            self._append(f"\n✗ Fit exited with code {rc}.")

    def succeeded(self) -> bool:
        return self._succeeded
