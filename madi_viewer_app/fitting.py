"""Fitting launcher — runs ``scripts/fit_data.py`` as a subprocess.

Runs in a QThread so the UI stays responsive, streams stdout/stderr line
by line via Qt signals, and updates the profile's ``fitted`` flag when
the child process exits cleanly.
"""
from __future__ import annotations

import os
import sys
import subprocess
from pathlib import Path

from PyQt5.QtCore import QObject, QThread, pyqtSignal

from .app_state import OutputProfile

_REPO_ROOT = Path(__file__).resolve().parent.parent
_FIT_SCRIPT = _REPO_ROOT / "scripts" / "fit_data.py"


def build_fit_command(profile: OutputProfile,
                      python_exe: str | None = None) -> list[str]:
    """Assemble the argv for scripts/fit_data.py from a profile."""
    py = python_exe or sys.executable
    cmd = [
        py, str(_FIT_SCRIPT),
        "--fit",
        "--library", profile.library_path,
        "--mask",    profile.mask_path,
        "--out",     profile.output_dir,
    ]
    cmd.append("--input")
    for d, p in sorted(profile.inputs, key=lambda x: float(x[0])):
        cmd.append(f"{float(d):g}:{p}")
    if profile.rician_correct:
        cmd.append("--rician-correct")
    if profile.noise_sigma is not None:
        cmd += ["--noise-sigma", str(float(profile.noise_sigma))]
    if profile.avg_s0:
        cmd.append("--avg-s0")
    if profile.fit_s0:
        cmd.append("--fit-s0")
    if profile.log_space:
        cmd.append("--log_space")
    cmd += ["--vi-min", f"{profile.vi_min:g}",
            "--vi-max", f"{profile.vi_max:g}"]
    if profile.rho_max is not None:
        cmd += ["--rho-max", f"{float(profile.rho_max):g}"]
    return cmd


class FitWorker(QObject):
    """Runs fit_data.py, emits signals while the job runs."""

    line = pyqtSignal(str)
    finished = pyqtSignal(int)   # returncode
    error = pyqtSignal(str)

    def __init__(self, profile: OutputProfile, python_exe: str | None = None):
        super().__init__()
        self.profile = profile
        self.python_exe = python_exe
        self._proc: subprocess.Popen | None = None
        self._cancelled = False

    def start(self):
        try:
            cmd = build_fit_command(self.profile, self.python_exe)
            Path(self.profile.output_dir).mkdir(parents=True, exist_ok=True)
            self.line.emit("$ " + " ".join(repr(c) if " " in c else c for c in cmd))
            self._proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                cwd=str(_REPO_ROOT),
                env={**os.environ, "PYTHONUNBUFFERED": "1"},
            )
            for line in iter(self._proc.stdout.readline, ""):
                if self._cancelled:
                    self._proc.kill()
                    break
                self.line.emit(line.rstrip())
            rc = self._proc.wait()
            self.finished.emit(rc)
        except Exception as e:
            self.error.emit(str(e))
            self.finished.emit(-1)

    def cancel(self):
        self._cancelled = True
        if self._proc and self._proc.poll() is None:
            try:
                self._proc.terminate()
            except Exception:
                pass


def run_fit_in_thread(profile: OutputProfile,
                      on_line, on_finished, on_error=None,
                      python_exe: str | None = None) -> tuple[QThread, FitWorker]:
    """Convenience: wire FitWorker onto a new QThread. Caller keeps both refs."""
    thread = QThread()
    worker = FitWorker(profile, python_exe=python_exe)
    worker.moveToThread(thread)
    thread.started.connect(worker.start)
    worker.line.connect(on_line)
    worker.finished.connect(on_finished)
    if on_error is not None:
        worker.error.connect(on_error)
    worker.finished.connect(thread.quit)
    thread.finished.connect(thread.deleteLater)
    return thread, worker
