#!/usr/bin/env python3
"""
fig_edema_contra_scatter.py — "figure of my choice": how well each MADI fitting
method separates edema from contralateral (healthy-appearing) tissue.

One panel per parameter (kio, rho, V). Within a panel each point is one subject:
    x = contralateral ROI mean
    y = edema ROI mean
    color = fitting method   (dataviz categorical palette)
    marker = subject         (shared with the other paper figures)
A y = x reference line is drawn: points above the line = edema is higher than
healthy tissue for that subject/method, below = lower. Distance from the line is
the effect the maps are meant to show, so a method whose cloud sits farther from
y = x is separating the two tissues more strongly.

Only subjects that have BOTH an edema and a contra DWI-space mask are used.

Usage
-----
    conda activate mri
    PYTHONPATH=. python -m scripts.edema_figures.fig_edema_contra_scatter
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from . import config, loaders
from .fig_roi_method_bars import METHOD_COLORS, subjects_with_roi

# distinct marker per subject; extends config.FIG3_MARKERS to the full cohort.
SUBJECT_MARKERS = {"001": "o", "003": "s", "187": "^", "002": "D",
                   "011": "v", "132": "P", "150": "X", "175": "*",
                   "196": "<", "260": ">"}


def paired_subjects():
    """Subjects with both an edema and a contra DWI-space mask."""
    edema = set(subjects_with_roi("edema"))
    contra = set(subjects_with_roi("contra"))
    return sorted(edema & contra)


def panel(ax, param, subjects, methods):
    xs, ys = [], []
    for method in methods:
        for sub in subjects:
            try:
                data = loaders.load_param_map(sub, method, param)
            except FileNotFoundError:
                continue
            cx, _ = loaders.roi_mean_std(data, loaders.load_dwi_mask(sub, "contra"))
            ey, _ = loaders.roi_mean_std(data, loaders.load_dwi_mask(sub, "edema"))
            if not (np.isfinite(cx) and np.isfinite(ey)):
                continue
            xs.append(cx); ys.append(ey)
            ax.scatter(cx, ey, s=55, color=METHOD_COLORS.get(method, "0.5"),
                       marker=SUBJECT_MARKERS.get(sub, "o"),
                       edgecolor="0.2", linewidth=0.5, zorder=3)

    # y = x reference across the pooled data range
    if xs:
        lo = min(min(xs), min(ys))
        hi = max(max(xs), max(ys))
        pad = 0.05 * (hi - lo or 1.0)
        line = [lo - pad, hi + pad]
        ax.plot(line, line, color="0.55", linestyle="--", linewidth=1.0,
                zorder=1, label="edema = contra")
        ax.set_xlim(line); ax.set_ylim(line)

    ax.set_title(config.PARAM_LABELS[param], fontsize=11)
    ax.set_xlabel("contralateral mean", fontsize=9)
    ax.set_ylabel("edema mean", fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("0.6")
    ax.spines["bottom"].set_color("0.6")
    ax.tick_params(colors="0.3", labelsize=8)
    ax.grid(True, color="0.9", linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)


def main():
    methods = config.ALL_METHODS
    subjects = paired_subjects()
    if not subjects:
        raise SystemExit("no subjects have both edema and contra masks")
    print(f"paired subjects: {subjects}")

    params = config.PARAMS
    fig, axes = plt.subplots(1, len(params),
                             figsize=(4.4 * len(params), 4.4),
                             constrained_layout=True)
    for ax, param in zip(np.atleast_1d(axes), params):
        panel(ax, param, subjects, methods)

    # two legends: color = method, marker = subject
    method_handles = [plt.Line2D([0], [0], marker="o", linestyle="",
                                 markerfacecolor=METHOD_COLORS[m],
                                 markeredgecolor="0.2", markersize=9, label=m)
                      for m in methods]
    subj_handles = [plt.Line2D([0], [0], marker=SUBJECT_MARKERS.get(s, "o"),
                               linestyle="", markerfacecolor="0.6",
                               markeredgecolor="0.2", markersize=9, label=f"sub-{s}")
                    for s in subjects]
    leg1 = fig.legend(handles=method_handles, title="Fitting method",
                      loc="outside lower left", ncol=len(methods), frameon=False,
                      fontsize=9, title_fontsize=9)
    fig.add_artist(leg1)
    fig.legend(handles=subj_handles, title="Subject",
               loc="outside lower right", ncol=min(len(subjects), 5),
               frameon=False, fontsize=9, title_fontsize=9)

    fig.suptitle("Edema vs. contralateral separation by fitting method",
                 fontsize=13)

    os.makedirs(config.FIGURES_OUT, exist_ok=True)
    out = os.path.join(config.FIGURES_OUT, "fig_edema_contra_scatter.png")
    fig.savefig(out, dpi=config.FIGURE_DPI)
    plt.close(fig)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
