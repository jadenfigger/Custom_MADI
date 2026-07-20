#!/usr/bin/env python3
"""
fig_tissue_method_bars.py — grey matter vs. white matter, averaged across all
subjects, one bar chart per parameter (kio, rho, V), inner-grouped by fitting
method.

This is the tissue-focused counterpart to fig_roi_method_bars.py: same
collect()/make_figure() engine (already parameterized by `rois`), just called
with a fixed two-ROI list instead of the auto-discovered full set, so the two
don't crowd each other's x-axis.

Usage
-----
    conda activate mri
    PYTHONPATH=. python -m scripts.edema_figures.fig_tissue_method_bars
"""

import os

from . import config
from .fig_roi_method_bars import collect, make_figure

TISSUE_ROIS = ["grey-matter", "white-matter"]


def main():
    methods = config.ALL_METHODS
    print(f"ROIs: {TISSUE_ROIS}   methods: {methods}")
    for param in config.PARAMS:
        stats = collect(param, TISSUE_ROIS, methods)
        make_figure(param, TISSUE_ROIS, methods, stats,
                    out_stem="fig_tissue_method_bars",
                    title="{param} — grey vs. white matter by fitting method")


if __name__ == "__main__":
    main()
