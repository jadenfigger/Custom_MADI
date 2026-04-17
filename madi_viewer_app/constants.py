"""Constants for the MADI viewer app.

Mirrors the acquisition protocol used by scripts/madi_viewer.py so this
module is a one-stop reference rather than relying on the scripts folder.
"""
from __future__ import annotations

import numpy as np

# ------------- Acquisition protocol -------------
SHELLS = [
    (1000, slice(1, 25)),
    (2500, slice(25, 49)),
    (4000, slice(49, 73)),
    (6000, slice(73, 97)),
]
BVALS_DISPLAY = np.array([1000, 2500, 4000, 6000])
N_SHELLS = len(SHELLS)

# ------------- Map metadata -------------
MAP_NAMES = ["kio", "rho", "V", "residual"]

MAP_CMAPS = {
    "kio": "inferno",
    "rho": "viridis",
    "V": "plasma",
    "residual": "magma",
    "s0_fit": "cividis",
    "s0_fit_over_measured": "coolwarm",
}

MAP_LABELS = {
    "kio": r"$k_{io}$  (s$^{-1}$)",
    "rho": r"$\rho$  (cells/μL)",
    "V": r"$V$  (pL)",
    "residual": "SSE",
    "s0_fit": "fitted S0",
    "s0_fit_over_measured": "fit S0 / meas S0",
}

# ------------- Plot palette -------------
MATCH_COLORS = [
    "#2563eb", "#dc2626", "#059669", "#d97706", "#7c3aed",
    "#0891b2", "#db2777", "#65a30d", "#ea580c", "#4f46e5",
    "#0ea5e9", "#f59e0b", "#10b981", "#e11d48", "#8b5cf6",
]
OBS_COLOR = "#111827"

# One deterministic color per profile (cycled when > N profiles open)
PROFILE_COLORS = [
    "#1d4ed8", "#be123c", "#047857", "#b45309", "#6d28d9",
    "#0e7490", "#a21caf", "#65a30d", "#c2410c", "#4338ca",
]

# ------------- Defaults -------------
DEFAULT_TOP_N = 10
DEFAULT_MARGIN_MM = 1.5
