"""Application entry point."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _need_pyqt_message():
    return (
        "PyQt5 is required for the MADI Viewer app.\n"
        "Install with:\n"
        "    pip install PyQt5\n"
    )


def main() -> int:
    ap = argparse.ArgumentParser(
        description="MADI Viewer — persistent diagnostic app for MADI fittings.")
    ap.add_argument("--workspace", default=None,
                    help="Workspace directory (default: ~/.madi_viewer)")
    args = ap.parse_args()

    try:
        from PyQt5.QtWidgets import QApplication
    except ImportError:
        sys.stderr.write(_need_pyqt_message())
        return 2

    # Make sure the repo root is importable
    repo_root = Path(__file__).resolve().parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    from .ui.style import apply_mpl_style, QT_STYLE
    from .ui.main_window import MainWindow
    from .app_state import Workspace

    apply_mpl_style()

    app = QApplication(sys.argv)
    app.setApplicationName("MADI Viewer")
    app.setOrganizationName("madi")
    app.setStyleSheet(QT_STYLE)

    ws = Workspace.load(Path(args.workspace) if args.workspace else None)
    w = MainWindow(ws)
    w.show()
    return app.exec_()


if __name__ == "__main__":
    sys.exit(main())
