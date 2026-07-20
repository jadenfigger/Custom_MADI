#!/usr/bin/env python3
"""
fig_tissue_subject_bars.py — grey matter vs. white matter, one bar per
subject, for a single fixed fitting method (BAYES-fits0 by default). One
figure per parameter (kio, rho, V).

Where fig_tissue_method_bars.py averages over subjects to compare methods,
this one holds the method fixed and shows subject-to-subject variability
directly -- useful for spotting whether GM/WM separation is a cohort-wide
effect or driven by a few subjects.

Bar height = that subject's ROI mean; error bar = std across voxels within
the ROI (within-subject spread, not between-subject).

Usage
-----
    conda activate mri
    PYTHONPATH=. python -m scripts.edema_figures.fig_tissue_subject_bars
    PYTHONPATH=. python -m scripts.edema_figures.fig_tissue_subject_bars --method BAYES
"""

import argparse
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from . import config, loaders
from .fig_tissue_method_bars import TISSUE_ROIS
from .fig_roi_method_bars import subjects_with_roi

# dataviz reference categorical palette, slots 5/6 -- distinct from
# METHOD_COLORS (slots 1-4) since this chart's color dimension is ROI, not
# fitting method.
TISSUE_COLORS = {
    "grey-matter":  "#4a3aa7",  # slot 5 violet
    "white-matter": "#e34948",  # slot 6 red
}


def collect(param, subjects, rois, method):
    """stats[subject][roi] -> (mean, std_within_roi, n_voxels)."""
    stats = defaultdict(dict)
    for sub in subjects:
        for roi in rois:
            try:
                data = loaders.load_param_map(sub, method, param)
                mask = loaders.load_dwi_mask(sub, roi)
            except FileNotFoundError:
                stats[sub][roi] = (np.nan, np.nan, 0)
                continue
            mean, std = loaders.roi_mean_std(data, mask)
            n = int(mask.sum())
            stats[sub][roi] = (mean, std, n)
    return stats


def make_figure(param, subjects, rois, stats, method):
    n_sub, n_roi = len(subjects), len(rois)
    x = np.arange(n_sub)
    width = 0.8 / n_roi

    fig, ax = plt.subplots(figsize=(1.1 * n_sub + 2.2, 4.2), constrained_layout=True)

    for j, roi in enumerate(rois):
        heights = [stats[sub][roi][0] for sub in subjects]
        errs = [stats[sub][roi][1] for sub in subjects]
        offset = (j - (n_roi - 1) / 2) * width
        ax.bar(
            x + offset, heights, width, yerr=errs, capsize=2,
            error_kw=dict(elinewidth=0.8, capthick=0.8),
            color=TISSUE_COLORS.get(roi, "0.5"),
            edgecolor="0.25", linewidth=0.5, label=roi.replace("-", " ").title(), zorder=3,
        )

    ax.set_xticks(x)
    ax.set_xticklabels([f"sub-{s}" for s in subjects], fontsize=10)
    ax.set_ylabel(config.PARAM_LABELS[param], fontsize=11)
    ax.set_title(f"{config.PARAM_LABELS[param]} — grey vs. white matter by subject ({method})",
                 fontsize=12)
    ax.margins(x=0.02)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("0.6")
    ax.spines["bottom"].set_color("0.6")
    ax.tick_params(colors="0.3", labelsize=9)
    ax.yaxis.grid(True, color="0.88", linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)

    ax.legend(title="Tissue", frameon=False, fontsize=9, title_fontsize=9, loc="upper right")

    os.makedirs(config.FIGURES_OUT, exist_ok=True)
    out = os.path.join(config.FIGURES_OUT, f"fig_tissue_subject_bars_{param}.png")
    fig.savefig(out, dpi=config.FIGURE_DPI)
    plt.close(fig)
    print(f"saved {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--method", default="BAYES-fits0", choices=config.ALL_METHODS)
    args = ap.parse_args()

    subjects = sorted(set(subjects_with_roi("grey-matter")) & set(subjects_with_roi("white-matter")))
    if not subjects:
        raise SystemExit("no subjects have both grey-matter and white-matter DWI-space masks")
    print(f"subjects: {subjects}   method: {args.method}")

    for param in config.PARAMS:
        stats = collect(param, subjects, TISSUE_ROIS, args.method)
        make_figure(param, subjects, TISSUE_ROIS, stats, args.method)


if __name__ == "__main__":
    main()
