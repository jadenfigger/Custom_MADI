#!/usr/bin/env python3
"""
fig3_boxplots.py — replicate paper Figure 3's edema-vs-non-edema comparison
using MADI kio/rho/V (BAYES method), but as per-voxel distributions rather
than the paper's per-subject box plot.

Only subjects with both an edema and contra mask (001, 003, 187) can
contribute. Each region (edema/contra) is shown as a light violin (the
pooled voxel distribution across all 3 subjects) with every individual
voxel plotted as a small jittered, semi-transparent point colored by
subject -- this shows the real within-ROI spread (hundreds of voxels) that
collapsing each subject to one mean would hide. kio/rho/V don't share a
common unit/scale (unlike the paper's ~0-1 normalized metrics), so each
parameter gets its own subplot/y-axis instead of one shared axis.

Usage
-----
    python -m scripts.edema_figures.fig3_boxplots
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from . import config, loaders

REGIONS = [("edema", "edematous"), ("contra", "non-edematous")]
REGION_X = {"edema": 0.0, "contra": 1.0}
JITTER_WIDTH = 0.32
RNG_SEED = 0


def voxel_values():
    """values[param][region][subject] = 1-D array of raw voxel values."""
    values = {p: {r: {} for r, _ in REGIONS} for p in config.PARAMS}
    for subject in config.FIG3_SUBJECTS:
        for param in config.PARAMS:
            data = loaders.load_param_map(subject, config.FIG3_METHOD, param)
            for region, _ in REGIONS:
                mask = loaders.load_dwi_mask(subject, region)
                v = data[mask]
                values[param][region][subject] = v[np.isfinite(v)]
    return values


def main():
    values = voxel_values()
    subjects = config.FIG3_SUBJECTS
    rng = np.random.default_rng(RNG_SEED)

    fig, axes = plt.subplots(1, len(config.PARAMS), figsize=(3.3 * len(config.PARAMS), 4.3))

    for ax, param in zip(axes, config.PARAMS):
        for region, _ in REGIONS:
            x0 = REGION_X[region]
            pooled = np.concatenate([values[param][region][s] for s in subjects])
            parts = ax.violinplot([pooled], positions=[x0], widths=0.85,
                                   showextrema=False, showmedians=False)
            for body in parts["bodies"]:
                body.set_facecolor("0.85")
                body.set_edgecolor("none")
                body.set_alpha(0.9)
                body.set_zorder(1)
            median = np.median(pooled)
            ax.hlines(median, x0 - 0.34, x0 + 0.34, color="0.3", linewidth=1.6, zorder=2)

            for subject in subjects:
                v = values[param][region][subject]
                jitter = rng.uniform(-JITTER_WIDTH, JITTER_WIDTH, size=v.size)
                ax.scatter(
                    x0 + jitter, v, s=7, color=config.SUBJECT_COLORS[subject],
                    alpha=0.35, linewidths=0, zorder=3,
                )

        ax.set_xlim(-0.65, 1.65)
        ax.set_xticks([REGION_X[r] for r, _ in REGIONS])
        ax.set_xticklabels([label for _, label in REGIONS])
        ax.set_ylabel(config.PARAM_LABELS[param])
        ax.set_title(param)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    handles = [
        plt.Line2D([], [], marker="o", color=config.SUBJECT_COLORS[s],
                   linestyle="None", markersize=8, label=f"sub-{s}")
        for s in subjects
    ]
    fig.legend(handles=handles, loc="lower center", ncol=len(subjects),
               bbox_to_anchor=(0.5, -0.05), frameon=False)

    fig.suptitle(f"Figure 3 replication — edema vs. non-edema, per-voxel ({config.FIG3_METHOD})",
                 fontsize=13)
    fig.tight_layout(rect=[0, 0.03, 1, 1])

    os.makedirs(config.FIGURES_OUT, exist_ok=True)
    out_path = os.path.join(config.FIGURES_OUT, "fig3_boxplots.png")
    fig.savefig(out_path, dpi=config.FIGURE_DPI, bbox_inches="tight")
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
