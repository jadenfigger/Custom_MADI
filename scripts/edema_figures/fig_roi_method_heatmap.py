#!/usr/bin/env python3
"""
fig_roi_method_heatmap.py — bird's-eye view of every ROI x every fitting
method at once (mean over subjects), one heatmap per parameter.

The bar charts (fig_roi_method_bars.py, fig_tissue_*_bars.py) are better for
reading exact values and error bars off a handful of ROIs; this is the
complementary "scan everything at a glance" view once the ROI list grows
past what a bar chart can hold readably (edema/contra/tumor/tumor-core/
tumor-net/tumor-edema/grey-matter/white-matter and counting). Reuses
fig_roi_method_bars.collect() directly, so it shares the exact same numbers
as the bar charts -- just a different rendering.

Color: matplotlib's YlOrRd (yellow->orange->red), the conventional heatmap
spectrum -- this is a magnitude encoding, not a categorical one, so it does
not reuse METHOD_COLORS/TISSUE_COLORS. Cells are direct-labeled with mean
(the value the hue encodes) and std below it (dataviz relief rule: text, not
color alone, carries the reading for the pale end of the ramp).

Usage
-----
    conda activate mri
    PYTHONPATH=. python -m scripts.edema_figures.fig_roi_method_heatmap
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from . import config
from .fig_roi_method_bars import collect, discover_rois

# Conventional yellow->orange->red heatmap spectrum.
HEAT_RAMP = plt.get_cmap("YlOrRd")


def make_figure(param, rois, methods, stats):
    n_roi, n_method = len(rois), len(methods)
    grid = np.full((n_roi, n_method), np.nan)
    std_grid = np.full((n_roi, n_method), np.nan)
    for i, roi in enumerate(rois):
        for j, method in enumerate(methods):
            grid[i, j] = stats[roi][method][0]
            std_grid[i, j] = stats[roi][method][1]

    fig, ax = plt.subplots(figsize=(1.3 * n_method + 2.0, 0.65 * n_roi + 1.6), constrained_layout=True)
    im = ax.imshow(grid, cmap=HEAT_RAMP, aspect="auto")

    ax.set_xticks(range(n_method))
    ax.set_xticklabels(methods, fontsize=9, rotation=20, ha="right")
    ax.set_yticks(range(n_roi))
    ax.set_yticklabels([r.replace("-", " ").title() for r in rois], fontsize=9)

    # direct labels: dark text on pale cells, light text on saturated cells
    # (relief rule -- never rely on the ramp alone to carry the value).
    vmin, vmax = np.nanmin(grid), np.nanmax(grid)
    mid = (vmin + vmax) / 2 if np.isfinite(vmin) and np.isfinite(vmax) else 0
    for i in range(n_roi):
        for j in range(n_method):
            v = grid[i, j]
            if not np.isfinite(v):
                ax.text(j, i, "—", ha="center", va="center", fontsize=8, color="0.5")
                continue
            color = "white" if v > mid else "#0b0b0b"
            s = std_grid[i, j]
            label = f"{v:.3g}\n±{s:.2g}" if np.isfinite(s) else f"{v:.3g}"
            ax.text(j, i, label, ha="center", va="center", fontsize=8, color=color, linespacing=1.6)

    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks(np.arange(-0.5, n_method, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n_roi, 1), minor=True)
    ax.grid(which="minor", color="#fcfcfb", linewidth=2)
    ax.tick_params(which="minor", length=0)
    ax.tick_params(which="major", length=0)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cbar.set_label(config.PARAM_LABELS[param], fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    ax.set_title(f"{config.PARAM_LABELS[param]} — mean by ROI and fitting method", fontsize=12)

    os.makedirs(config.FIGURES_OUT, exist_ok=True)
    out = os.path.join(config.FIGURES_OUT, f"fig_roi_method_heatmap_{param}.png")
    fig.savefig(out, dpi=config.FIGURE_DPI)
    plt.close(fig)
    print(f"saved {out}")


def main():
    methods = config.ALL_METHODS
    rois = discover_rois()
    if not rois:
        raise SystemExit("no DWI-space ROI masks found under derivatives/rois")
    print(f"ROIs: {rois}   methods: {methods}")
    for param in config.PARAMS:
        stats = collect(param, rois, methods)
        make_figure(param, rois, methods, stats)


if __name__ == "__main__":
    main()
