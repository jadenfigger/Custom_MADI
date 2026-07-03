#!/usr/bin/env python3
"""
fig1_slices.py — replicate paper Figure 1's layout (one row per participant,
one column per map) using MADI kio/rho/V instead of the paper's DTI/NODDI/
SMI scalars.

Subjects: FIG1_SUBJECTS (config.py), method: FIG1_METHOD (BAYES).
Slice per subject: the edema mask's dominant z-slice for subjects that have
one (001/003/187), else the geometric mid-slice (n_z // 2), overridable via
FIG1_SLICE_OVERRIDE in config.py.

Usage
-----
    python -m scripts.edema_figures.fig1_slices
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
from matplotlib.gridspec import GridSpec
import numpy as np

from . import config, loaders, slicing, windowing


def pick_slice(subject: str) -> int:
    if subject in config.FIG1_SLICE_OVERRIDE:
        return config.FIG1_SLICE_OVERRIDE[subject]
    if subject in config.EDEMA_MASK_SUBJECTS:
        mask = loaders.load_dwi_mask(subject, "edema")
        return loaders.mask_slice_index(mask)
    kio = loaders.load_param_map(subject, config.FIG1_METHOD, "kio")
    return kio.shape[2] // 2


def main():
    subjects = config.FIG1_SUBJECTS
    method = config.FIG1_METHOD
    show_flair = config.FIG1_SHOW_FLAIR and all(s in config.FLAIR_SUBJECTS for s in subjects)
    flair_per_subject = show_flair and config.FIG1_FLAIR_PER_SUBJECT_WINDOW
    params = (["flair"] if show_flair else []) + config.PARAMS

    # Load everything up front: maps, brain masks, chosen slice per subject.
    maps = {}       # (subject, param) -> 3D array
    brain_masks = {}  # subject -> 3D bool
    slices = {}      # subject -> z index
    for subject in subjects:
        for param in config.PARAMS:
            maps[(subject, param)] = loaders.load_param_map(subject, method, param)
        if show_flair:
            maps[(subject, "flair")] = loaders.load_flair_map(subject)
        shape = maps[(subject, config.PARAMS[0])].shape
        brain_masks[subject] = loaders.load_brain_mask(subject, shape)
        slices[subject] = pick_slice(subject)

    # Global per-parameter windows, pooled across all subjects' brain voxels.
    # FLAIR is excluded here when windowed per-subject (see below) since raw
    # FLAIR signal isn't on a comparable scale across subjects.
    pooled_params = [p for p in params if p != "flair" or not flair_per_subject]
    pooled = {
        param: [maps[(s, param)][brain_masks[s]] for s in subjects]
        for param in pooled_params
    }
    windows = windowing.compute_windows(pooled)

    contrast_override = config.FIG1_CONTRAST_OVERRIDE
    for param, override in contrast_override.items():
        if param != "flair" or not flair_per_subject:
            windows[param] = override

    # Each subject's own brain-voxel percentile window for FLAIR, used to
    # normalize that subject's panel to 0-1 before display (instead of a
    # shared raw-value window) when FIG1_FLAIR_PER_SUBJECT_WINDOW is set.
    # A "flair" contrast override fixes this window to the same value for
    # every subject rather than each computing its own.
    flair_windows = {}
    if flair_per_subject:
        flair_override = contrast_override.get("flair")
        for s in subjects:
            if flair_override is not None:
                flair_windows[s] = flair_override
            else:
                flair_windows[s] = windowing.compute_windows(
                    {"flair": [maps[(s, "flair")][brain_masks[s]]]}
                )["flair"]

    n_rows, n_cols = len(subjects), len(params)

    # Explicit GridSpec: [image, image, image, cbar, image, image, image,
    # cbar, ...] would be wasteful; instead one narrow cbar column per
    # parameter, immediately right of that parameter's image column.
    width_ratios = []
    col_of_param = {}
    cbar_col_of_param = {}
    for param in params:
        col_of_param[param] = len(width_ratios)
        width_ratios.append(1.0)
        cbar_col_of_param[param] = len(width_ratios)
        width_ratios.append(0.07)
    fig = plt.figure(figsize=(3.0 * n_cols, 2.4 * n_rows))
    gs = GridSpec(n_rows, len(width_ratios), figure=fig, width_ratios=width_ratios,
                   wspace=0.06, hspace=0.03)

    # Zoom each subject's row in on just their brain (same crop reused
    # across that row's 3 parameter columns, since it's the same slice).
    crop_bboxes = {
        subject: slicing.square_crop_bbox(slicing.axial(brain_masks[subject], slices[subject]))
        for subject in subjects
    }

    col_images = {param: None for param in params}
    for r, subject in enumerate(subjects):
        z = slices[subject]
        bbox = crop_bboxes[subject]
        for param in params:
            c = col_of_param[param]
            ax = fig.add_subplot(gs[r, c])
            data2d = slicing.axial(maps[(subject, param)], z)
            brain2d = slicing.axial(brain_masks[subject], z)
            if param == "flair" and flair_per_subject:
                vmin, vmax = flair_windows[subject]
                data2d = (data2d - vmin) / (vmax - vmin) if vmax > vmin else data2d - vmin
                vmin, vmax = 0.0, 1.0
            else:
                vmin, vmax = windows[param]
            im = windowing.render_panel(ax, data2d, brain2d, vmin, vmax, crop_bbox=bbox)
            col_images[param] = im
            if r == 0:
                ax.set_title(config.PARAM_LABELS[param], fontsize=12)
            if param == params[0]:
                ax.text(-0.12, 0.5, f"sub-{subject}\n(z={z})", fontsize=10,
                        ha="right", va="center", transform=ax.transAxes)

    for param in params:
        c = cbar_col_of_param[param]
        cax = fig.add_subplot(gs[:, c])
        cbar = fig.colorbar(col_images[param], cax=cax)
        cbar.locator = matplotlib.ticker.MaxNLocator(nbins=5)
        cbar.update_ticks()
        cax.tick_params(labelsize=9)

    title_cols = "FLAIR/kio/rho/V" if show_flair else "kio/rho/V"
    fig.suptitle(f"Figure 1 replication — {title_cols} slices ({method})", fontsize=14)
    windowing.style_dark_figure(fig)

    os.makedirs(config.FIGURES_OUT, exist_ok=True)
    out_path = os.path.join(config.FIGURES_OUT, "fig1_subject_slices.png")
    fig.savefig(out_path, dpi=config.FIGURE_DPI, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
