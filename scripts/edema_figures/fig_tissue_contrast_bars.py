#!/usr/bin/env python3
"""
fig_tissue_contrast_bars.py — grey-minus-white-matter contrast (kio_GM -
kio_WM, etc.), one bar per subject, inner-grouped by fitting method. One
figure per parameter.

This is the "does every method agree on tissue separation" chart: fig_tissue_
method_bars.py and fig_tissue_subject_bars.py both show GM > WM holds up, but
not whether the *size* of that gap is method-dependent. A method whose bars
here sit systematically higher or lower than the others is more (or less)
sensitive to the GM/WM contrast for a given subject -- useful for judging
which fitting method is more informative for tissue-specific comparisons.

Error bars combine each ROI's within-subject std in quadrature
(sqrt(std_gm^2 + std_wm^2)) as a simple, understated uncertainty estimate on
the difference -- not a rigorous propagated SEM, just enough to flag which
contrasts are noise-dominated.

Usage
-----
    conda activate mri
    PYTHONPATH=. python -m scripts.edema_figures.fig_tissue_contrast_bars
"""

import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from . import config, loaders
from .fig_roi_method_bars import METHOD_COLORS, subjects_with_roi


def collect(param, subjects, methods):
    """stats[subject][method] -> (gm_minus_wm, combined_std)."""
    stats = defaultdict(dict)
    for sub in subjects:
        for method in methods:
            try:
                data = loaders.load_param_map(sub, method, param)
                gm_mask = loaders.load_dwi_mask(sub, "grey-matter")
                wm_mask = loaders.load_dwi_mask(sub, "white-matter")
            except FileNotFoundError:
                stats[sub][method] = (np.nan, np.nan)
                continue
            gm_mean, gm_std = loaders.roi_mean_std(data, gm_mask)
            wm_mean, wm_std = loaders.roi_mean_std(data, wm_mask)
            stats[sub][method] = (gm_mean - wm_mean, float(np.hypot(gm_std, wm_std)))
    return stats


def make_figure(param, subjects, methods, stats):
    n_sub, n_method = len(subjects), len(methods)
    x = np.arange(n_sub)
    width = 0.8 / n_method

    fig, ax = plt.subplots(figsize=(1.3 * n_sub + 2.2, 4.2), constrained_layout=True)

    for j, method in enumerate(methods):
        heights = [stats[sub][method][0] for sub in subjects]
        errs = [stats[sub][method][1] for sub in subjects]
        offset = (j - (n_method - 1) / 2) * width
        ax.bar(
            x + offset, heights, width, yerr=errs, capsize=2,
            error_kw=dict(elinewidth=0.8, capthick=0.8),
            color=METHOD_COLORS.get(method, "0.5"),
            edgecolor="0.25", linewidth=0.5, label=method, zorder=3,
        )

    ax.axhline(0, color="0.3", linewidth=0.9, zorder=2)
    ax.set_xticks(x)
    ax.set_xticklabels([f"sub-{s}" for s in subjects], fontsize=10)
    ax.set_ylabel(f"{config.PARAM_LABELS[param]} (grey − white)", fontsize=11)
    ax.set_title(f"{config.PARAM_LABELS[param]} — grey/white matter contrast by subject and method",
                 fontsize=12)
    ax.margins(x=0.02)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("0.6")
    ax.spines["bottom"].set_color("0.6")
    ax.tick_params(colors="0.3", labelsize=9)
    ax.yaxis.grid(True, color="0.88", linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)

    ax.legend(title="Fitting method", frameon=False, fontsize=9,
              title_fontsize=9, ncol=2, loc="upper right")

    os.makedirs(config.FIGURES_OUT, exist_ok=True)
    out = os.path.join(config.FIGURES_OUT, f"fig_tissue_contrast_bars_{param}.png")
    fig.savefig(out, dpi=config.FIGURE_DPI)
    plt.close(fig)
    print(f"saved {out}")


def main():
    methods = config.ALL_METHODS
    subjects = sorted(set(subjects_with_roi("grey-matter")) & set(subjects_with_roi("white-matter")))
    if not subjects:
        raise SystemExit("no subjects have both grey-matter and white-matter DWI-space masks")
    print(f"subjects: {subjects}   methods: {methods}")

    for param in config.PARAMS:
        stats = collect(param, subjects, methods)
        make_figure(param, subjects, methods, stats)


if __name__ == "__main__":
    main()
