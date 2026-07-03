#!/usr/bin/env python3
"""
fig6_bars.py — replicate paper Figure 6's layout (representative map(s) with
ROI outline, plus edema-vs-contralateral bar charts with error bars) using
MADI kio/rho/V across all 4 fitting methods.

Top row: kio (BAYES) map at each subject's edema slice, subjects 187/001/003
(187 leftmost), edema mask outlined -- the paper's panel (a)/(b) role.
Below: a 4 (method) x 3 (parameter) grid of bar charts. Each panel: x-axis =
subject (001, 003, 187), grouped bars = edema vs. contra, bar height = ROI
mean, error bar = std across that ROI's voxels (spatial variability within
the mask -- used uniformly across all 4 methods since MAP/MAP-fits0 have no
per-voxel uncertainty map to draw on, only BAYES/BAYES-fits0 do).

Usage
-----
    python -m scripts.edema_figures.fig6_bars
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
from matplotlib.gridspec import GridSpec
import numpy as np

from . import config, loaders, slicing, windowing

REGIONS = [("edema", "edema"), ("contra", "contra")]
BAR_COLOR = {"edema": "0.35", "contra": "0.75"}


def build_poster_row(fig, gs, row_idx, n_cols):
    subjects = config.FIG6_POSTER_SUBJECTS
    method = config.FIG6_POSTER_METHOD
    param = config.FIG6_POSTER_PARAM

    kio_maps, brain_masks, edema_masks, slices = {}, {}, {}, {}
    for subject in subjects:
        kio_maps[subject] = loaders.load_param_map(subject, method, param)
        brain_masks[subject] = loaders.load_brain_mask(subject, kio_maps[subject].shape)
        edema_masks[subject] = loaders.load_dwi_mask(subject, "edema")
        slices[subject] = loaders.mask_slice_index(edema_masks[subject])

    windows = windowing.compute_windows(
        {param: [kio_maps[s][brain_masks[s]] for s in subjects]}
    )
    vmin, vmax = windows[param]

    im = None
    for c, subject in enumerate(subjects):
        ax = fig.add_subplot(gs[row_idx, c])
        z = slices[subject]
        data2d = slicing.axial(kio_maps[subject], z)
        brain2d = slicing.axial(brain_masks[subject], z)
        edema2d = slicing.axial(edema_masks[subject], z)
        im = windowing.render_panel(ax, data2d, brain2d, vmin, vmax)
        ax.contour(edema2d, colors="red", linewidths=1.2)
        ax.set_title(f"sub-{subject} (z={z})", fontsize=11)

    cax = fig.add_subplot(gs[row_idx, n_cols])
    cbar = fig.colorbar(im, cax=cax)
    cbar.locator = matplotlib.ticker.MaxNLocator(nbins=4)
    cbar.update_ticks()
    cax.tick_params(labelsize=9)
    cax.set_ylabel(config.PARAM_LABELS[param], fontsize=9)


def build_bar_panel(ax, param, method):
    subjects = config.FIG6_BAR_SUBJECTS
    x = np.arange(len(subjects))
    width = 0.32

    for i, (region, label) in enumerate(REGIONS):
        means, stds = [], []
        for subject in subjects:
            data = loaders.load_param_map(subject, method, param)
            mask = loaders.load_dwi_mask(subject, region)
            mean, std = loaders.roi_mean_std(data, mask)
            means.append(mean)
            stds.append(std)
        offset = (i - 0.5) * width
        ax.bar(x + offset, means, width, yerr=stds, capsize=2,
               error_kw=dict(elinewidth=0.8, capthick=0.8),
               color=BAR_COLOR[region], edgecolor="none", label=label)

    ax.set_xticks(x)
    ax.set_xticklabels([f"sub-{s}" for s in subjects], fontsize=8)
    for tick, subject in zip(ax.get_xticklabels(), subjects):
        tick.set_color(config.SUBJECT_COLORS[subject])
        tick.set_fontweight("bold")
    ax.set_ylabel(config.PARAM_LABELS[param], fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("0.6")
    ax.spines["bottom"].set_color("0.6")
    ax.tick_params(colors="0.3", labelsize=8)
    ax.yaxis.grid(True, color="0.88", linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)


def main():
    methods = config.FIG6_METHODS
    params = config.PARAMS
    n_cols = len(params)

    n_rows = 1 + len(methods)
    width_ratios = [1.0] * n_cols + [0.07]
    fig = plt.figure(figsize=(3.8 * n_cols, 3.2 * n_rows), constrained_layout=True)
    gs = GridSpec(n_rows, n_cols + 1, figure=fig, width_ratios=width_ratios)

    build_poster_row(fig, gs, 0, n_cols)

    handles = None
    for r, method in enumerate(methods, start=1):
        for c, param in enumerate(params):
            ax = fig.add_subplot(gs[r, c])
            build_bar_panel(ax, param, method)
            if c == 0:
                ax.text(-0.35, 0.5, method, fontsize=11, ha="right", va="center",
                        transform=ax.transAxes, rotation=90)
            if handles is None:
                handles, _ = ax.get_legend_handles_labels()

    fig.legend(handles, [label for _, label in REGIONS], loc="outside lower center",
               ncol=2, frameon=False)

    fig.suptitle("Figure 6 replication — edema vs. contralateral, all methods/parameters",
                 fontsize=13)

    os.makedirs(config.FIGURES_OUT, exist_ok=True)
    out_path = os.path.join(config.FIGURES_OUT, "fig6_method_parameter_grid.png")
    fig.savefig(out_path, dpi=config.FIGURE_DPI)
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
