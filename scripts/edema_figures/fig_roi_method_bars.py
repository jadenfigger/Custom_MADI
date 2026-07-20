#!/usr/bin/env python3
"""
fig_roi_method_bars.py — grouped bar charts of MADI parameters, one figure per
parameter (kio, rho, V).

Layout of each figure
---------------------
    x-axis outer group = ROI            (edema, contra, + any TumorSynth ROIs)
    inner group        = fitting method (MAP, MAP-fits0, BAYES, BAYES-fits0)
    bar height         = mean over subjects of that ROI's per-subject mean
    error bar          = std across subjects (between-subject variability)

ROIs are auto-discovered by globbing every
`derivatives/rois/sub-XXX/*-dwi_mask.nii.gz`, so once TumorSynth ROIs are
resampled into DWI space (see docs/tumorsynth_install.md §6) they appear here
with no code change.

Method colors use the dataviz reference categorical palette (validated
colorblind-safe, fixed order).

Usage
-----
    conda activate mri
    PYTHONPATH=. python -m scripts.edema_figures.fig_roi_method_bars
"""

import glob
import os
import re
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from . import config, loaders

# dataviz reference categorical palette, fixed slot order (CVD-validated).
METHOD_COLORS = {
    "MAP":         "#2a78d6",  # slot 1 blue
    "MAP-fits0":   "#1baf7a",  # slot 2 aqua
    "BAYES":       "#eda100",  # slot 3 yellow
    "BAYES-fits0": "#008300",  # slot 4 green
}

# ROIs we never want as a bar group (brain/whole-head masks, not lesion ROIs).
ROI_DENYLIST = {"brain", "nodif"}


def discover_rois():
    """Return ROI descriptors that at least one subject has a DWI-space mask for,
    ordered with the familiar edema/contra first, then the rest alphabetically."""
    found = set()
    pattern = os.path.join(config.ROIS_ROOT, "sub-*", "*-dwi_mask.nii.gz")
    rx = re.compile(r"desc-(.+?)-dwi_mask\.nii\.gz$")
    for path in glob.glob(pattern):
        m = rx.search(os.path.basename(path))
        if m and m.group(1) not in ROI_DENYLIST:
            found.add(m.group(1))
    preferred = [r for r in ("edema", "contra") if r in found]
    rest = sorted(found - set(preferred))
    return preferred + rest


def subjects_with_roi(roi):
    """All subject IDs that have a DWI-space mask for this ROI."""
    subs = []
    for path in sorted(glob.glob(os.path.join(config.ROIS_ROOT, "sub-*"))):
        sub = os.path.basename(path).replace("sub-", "")
        if os.path.exists(loaders.dwi_mask_path(sub, roi)):
            subs.append(sub)
    return subs


def collect(param, rois, methods):
    """means[roi][method] -> (mean_over_subjects, std_over_subjects, n)."""
    stats = defaultdict(dict)
    for roi in rois:
        subs = subjects_with_roi(roi)
        for method in methods:
            per_subject = []
            for sub in subs:
                try:
                    data = loaders.load_param_map(sub, method, param)
                    mask = loaders.load_dwi_mask(sub, roi)
                except FileNotFoundError:
                    continue
                mean, _ = loaders.roi_mean_std(data, mask)
                if np.isfinite(mean):
                    per_subject.append(mean)
            arr = np.asarray(per_subject, float)
            if arr.size:
                stats[roi][method] = (arr.mean(), arr.std(), arr.size)
            else:
                stats[roi][method] = (np.nan, np.nan, 0)
    return stats


def make_figure(param, rois, methods, stats, out_stem="fig_roi_method_bars", title="{param} by ROI and fitting method"):
    n_roi, n_method = len(rois), len(methods)
    x = np.arange(n_roi)
    width = 0.8 / n_method

    fig, ax = plt.subplots(figsize=(1.9 * n_roi + 2.2, 4.2), constrained_layout=True)

    for j, method in enumerate(methods):
        heights = [stats[roi][method][0] for roi in rois]
        errs = [stats[roi][method][1] for roi in rois]
        offset = (j - (n_method - 1) / 2) * width
        bars = ax.bar(
            x + offset, heights, width, yerr=errs, capsize=2,
            error_kw=dict(elinewidth=0.8, capthick=0.8),
            color=METHOD_COLORS.get(method, "0.5"),
            edgecolor="0.25", linewidth=0.5, label=method, zorder=3,
        )
        # annotate subject count under sparse groups (relief for pale hues)
        for roi_i, roi in enumerate(rois):
            n = stats[roi][method][2]
            if 0 < n < 3:
                ax.annotate(f"n={n}", (x[roi_i] + offset, 0),
                            textcoords="offset points", xytext=(0, 2),
                            ha="center", va="bottom", fontsize=6, color="0.3",
                            rotation=90)

    ax.set_xticks(x)
    ax.set_xticklabels([r.replace("-", " ").title() for r in rois], fontsize=10)
    ax.set_ylabel(config.PARAM_LABELS[param], fontsize=11)
    ax.set_title(title.format(param=config.PARAM_LABELS[param]), fontsize=12)
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
    out = os.path.join(config.FIGURES_OUT, f"{out_stem}_{param}.png")
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
