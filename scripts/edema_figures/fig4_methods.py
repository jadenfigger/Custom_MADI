#!/usr/bin/env python3
"""
fig4_methods.py — kio/rho/V x fitting-method comparison grid for a single
subject (sub-187 by default), at that subject's edema-mask slice.

Rows = parameters (kio, rho, V), columns = fitting methods (MAP, MAP-fits0,
BAYES, BAYES-fits0). No ROI overlay -- clean maps, grayscale, one colorbar
per row windowed across that row's own 4 method panels only (this figure's
whole point is comparing methods, so the window is local to this subject/
figure, not shared with Figs 1/3/6).

Usage
-----
    python -m scripts.edema_figures.fig4_methods
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
from matplotlib.gridspec import GridSpec
import numpy as np

from . import config, loaders, slicing, windowing


def main():
    subject = config.FIG4_SUBJECT
    methods = config.FIG4_METHODS
    params = config.PARAMS

    maps = {(param, method): loaders.load_param_map(subject, method, param)
            for param in params for method in methods}
    shape = maps[(params[0], methods[0])].shape
    brain_mask = loaders.load_brain_mask(subject, shape)

    if subject in config.EDEMA_MASK_SUBJECTS:
        edema_mask = loaders.load_dwi_mask(subject, "edema")
        z = loaders.mask_slice_index(edema_mask)
    else:
        z = shape[2] // 2

    pooled = {
        param: [maps[(param, m)][brain_mask] for m in methods]
        for param in params
    }
    windows = windowing.compute_windows(pooled)
    windows.update(config.FIG4_CONTRAST_OVERRIDE)

    n_rows, n_cols = len(params), len(methods)
    width_ratios = [1.0] * n_cols + [0.07]
    fig = plt.figure(figsize=(2.5 * n_cols, 2.5 * n_rows))
    gs = GridSpec(n_rows, n_cols + 1, figure=fig, width_ratios=width_ratios,
                   wspace=0.04, hspace=0.14)

    brain2d = slicing.axial(brain_mask, z)
    crop_bbox = slicing.square_crop_bbox(brain2d)

    for r, param in enumerate(params):
        vmin, vmax = windows[param]
        row_image = None
        for c, method in enumerate(methods):
            ax = fig.add_subplot(gs[r, c])
            data2d = slicing.axial(maps[(param, method)], z)
            row_image = windowing.render_panel(ax, data2d, brain2d, vmin, vmax, crop_bbox=crop_bbox)
            if r == 0:
                ax.set_title(method, fontsize=12)
            if c == 0:
                ax.text(-0.15, 0.5, config.PARAM_LABELS[param], fontsize=12,
                        ha="right", va="center", transform=ax.transAxes)
        cax = fig.add_subplot(gs[r, n_cols])
        cbar = fig.colorbar(row_image, cax=cax)
        cbar.locator = matplotlib.ticker.MaxNLocator(nbins=4)
        cbar.update_ticks()
        cax.tick_params(labelsize=9)

    fig.suptitle(f"Figure 4 replication — sub-{subject} method comparison (z={z})", fontsize=14)
    windowing.style_dark_figure(fig)

    os.makedirs(config.FIGURES_OUT, exist_ok=True)
    out_path = os.path.join(config.FIGURES_OUT, "fig4_method_comparison.png")
    fig.savefig(out_path, dpi=config.FIGURE_DPI, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
