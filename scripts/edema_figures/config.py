"""
config.py — Paths and tunables for the paper-figure replication scripts
(scripts/edema_figures/fig{1,3,4,6}_*.py).

All subject lists / method choices called out in the design interview live
here as plain module-level constants (not buried in the figure scripts) so
they can be changed without touching plotting logic.
"""

import os
from dataclasses import dataclass, field
from typing import Dict, List

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
# Overridable via env vars so the same tracked config.py works unmodified on
# both the local machine and Sol (or any other host) -- e.g. on Sol:
#   export MADI_EDEMA_DATA_ROOT=/scratch/jfigger/edema
DATA_ROOT = os.environ.get(
    "MADI_EDEMA_DATA_ROOT",
    "/mnt/c/miscellaneous/coding_projects/python/mri_processing/data_storage/data/edema")
ORIGINAL_ROOT = os.environ.get(
    "MADI_EDEMA_ORIGINAL_ROOT",
    "/mnt/c/miscellaneous/coding_projects/python/mri_processing/data_storage/data/edema_original")

MADI_ROOT = os.path.join(DATA_ROOT, "derivatives", "madi")
ROIS_ROOT = os.path.join(DATA_ROOT, "derivatives", "rois")
PREPROC_ROOT = os.path.join(DATA_ROOT, "derivatives", "preproc")
FLAIR_ROOT = os.path.join(DATA_ROOT, "derivatives", "flair")
TUMORSYNTH_ROOT = os.path.join(DATA_ROOT, "derivatives", "tumorsynth")

# SRI-24 template TumorSynth output is registered to (see
# docs/tumorsynth_install.md §1c) -- needed to bring its labels back into each
# subject's native space. Not used by run_all.py's own pipeline (roi_space /
# flair_space / fig1,3,4,6) -- only by tumorsynth_roi_space.py.
SRI24_TEMPLATE = os.environ.get("MADI_SRI24_TEMPLATE", "/home/jaden/sri24/T1.nii.gz")
FIGURES_OUT = os.path.join(MADI_ROOT, "figures")
PAPER_FIGURES_DIR = os.path.join(ORIGINAL_ROOT, "figures")

# ---------------------------------------------------------------------------
# Fitting methods
# ---------------------------------------------------------------------------
# Directory suffix under sub-XXX/dwi/method-<...>/ matches these names exactly.
ALL_METHODS = ["MAP", "MAP-fits0", "BAYES", "BAYES-fits0"]

# MAP-family methods write kio_map.nii.gz etc.; BAYES-family write kio_mean.nii.gz.
def map_file_stem(method: str) -> str:
    return "mean" if method.startswith("BAYES") else "map"

PARAMS = ["kio", "rho", "V"]

PARAM_LABELS = {
    "kio": r"$k_{io}$  (s$^{-1}$)",
    "rho": r"$\rho$  (cells/μL)",
    "V": r"$V$  (pL)",
    "flair": "FLAIR",
}

# ---------------------------------------------------------------------------
# ROI masks
# ---------------------------------------------------------------------------
# Subjects with a usable T1-space contra mask in derivatives/rois.
CONTRA_MASK_SUBJECTS = ["001", "002", "003", "011", "187"]
# Subjects with a usable T1-space edema mask (single axial slice each).
EDEMA_MASK_SUBJECTS = ["001", "003", "187"]
# Subjects with a raw FLAIR anatomical available (see flair_space.py).
FLAIR_SUBJECTS = ["001", "002", "003", "011", "187"]

# Brain mask filename stems to try, in priority order (matches
# scripts/build_cohort_manifest.py's DEFAULT_MASK_PRIORITY).
BRAIN_MASK_PRIORITY = ["desc-nodif-brain-clean_mask", "desc-nodif-brain_mask"]

# ---------------------------------------------------------------------------
# Figure 1 — multi-subject kio/rho/V slice montage
# ---------------------------------------------------------------------------
FIG1_SUBJECTS = ["001", "002", "003", "011", "187"]
FIG1_METHOD = "BAYES-fits0"
# Leftmost reference column, ahead of the kio/rho/V panels. Only rendered
# for subjects in FLAIR_SUBJECTS (currently all of FIG1_SUBJECTS).
FIG1_SHOW_FLAIR = True
# Raw FLAIR signal isn't on a comparable scale across subjects/scanners (unlike
# kio/rho/V, which are physical units) -- a global pooled window washes out or
# blows out whichever subjects sit far from the pooled percentiles. When True,
# each subject's FLAIR panel is windowed to its own brain-voxel percentiles,
# then displayed on a shared normalized 0-1 scale (so one colorbar still
# applies to every row) instead of one shared raw-value window.
FIG1_FLAIR_PER_SUBJECT_WINDOW = True

# Manual (vmin, vmax) overrides, keyed by param name ("kio", "rho", "V",
# "flair"), replacing the auto-computed percentile window for that column.
# Leave a param out to keep the auto window. For "flair" when
# FIG1_FLAIR_PER_SUBJECT_WINDOW is True, this instead fixes the *raw* window
# used to normalize every subject to 0-1 (same window for all subjects,
# rather than each subject's own percentiles).
FIG1_CONTRAST_OVERRIDE: Dict[str, tuple] = {}
FIG4_CONTRAST_OVERRIDE: Dict[str, tuple] = {}
# Subjects without an edema mask fall back to the geometric mid-slice
# (n_z // 2) unless overridden here (subject -> explicit z index).
FIG1_SLICE_OVERRIDE: Dict[str, int] = {"002": 36, "011": 38}

FIG1_CONTRAST_OVERRIDE = {"kio": (0, 60), "rho": (0, 1.2e6)}

# ---------------------------------------------------------------------------
# Figure 3 — box plots, edema vs. contra
# ---------------------------------------------------------------------------
FIG3_SUBJECTS = ["001", "003", "187"]
FIG3_METHOD = "BAYES"
FIG3_MARKERS = {"001": "o", "003": "s", "187": "^"}
# Distinct per-subject hues (reused wherever a figure needs subject color
# coding, e.g. Fig 3's voxel cloud).
SUBJECT_COLORS = {"001": "#1d4ed8", "003": "#be123c", "187": "#047857"}

# ---------------------------------------------------------------------------
# Figure 4 — fitting-method comparison grid
# ---------------------------------------------------------------------------
FIG4_SUBJECT = "187"
FIG4_METHODS = ALL_METHODS
FIG4_CONTRAST_OVERRIDE = {"kio": (0, 60), "rho": (0, 1.2e6), "V": (0, 4.0)}


# ---------------------------------------------------------------------------
# Figure 6 — poster row + method x parameter bar grid
# ---------------------------------------------------------------------------
FIG6_POSTER_SUBJECTS = ["187", "001", "003"]  # 187 leftmost, per design
FIG6_BAR_SUBJECTS = ["001", "003", "187"]
FIG6_METHODS = ALL_METHODS
FIG6_POSTER_METHOD = "BAYES"
FIG6_POSTER_PARAM = "kio"

# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------
WINDOW_PERCENTILES = (1.0, 99.0)  # global per-parameter grayscale window
FIGURE_DPI = 300


@dataclass
class FigureConfig:
    """Bundle of the above, in case a script wants to pass overrides around."""
    subjects: List[str] = field(default_factory=lambda: list(FIG1_SUBJECTS))
    method: str = FIG1_METHOD
    params: List[str] = field(default_factory=lambda: list(PARAMS))
    window_percentiles: tuple = WINDOW_PERCENTILES
    dpi: int = FIGURE_DPI
