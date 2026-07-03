"""
windowing.py — global per-parameter grayscale brightness windows, shared
across every panel of a figure so panels stay visually comparable.
"""

from typing import Dict, Iterable, Tuple

import numpy as np

from . import config


def compute_windows(
    param_volumes: Dict[str, Iterable[np.ndarray]],
    percentiles: Tuple[float, float] = config.WINDOW_PERCENTILES,
) -> Dict[str, Tuple[float, float]]:
    """For each parameter, pool all brain-masked voxel values passed in and
    compute one (vmin, vmax) percentile window shared by every panel showing
    that parameter in the figure.

    Parameters
    ----------
    param_volumes : dict of param -> iterable of 1-D arrays (already brain-
        masked, i.e. background voxels excluded) contributing to that
        parameter's global window.
    """
    lo, hi = percentiles
    windows = {}
    for param, arrays in param_volumes.items():
        pooled = np.concatenate([np.asarray(a).ravel() for a in arrays])
        pooled = pooled[np.isfinite(pooled)]
        if pooled.size == 0:
            windows[param] = (0.0, 1.0)
            continue
        windows[param] = (float(np.percentile(pooled, lo)), float(np.percentile(pooled, hi)))
    return windows


def render_panel(ax, data2d: np.ndarray, brain2d: np.ndarray, vmin: float, vmax: float,
                  crop_bbox: tuple = None):
    """Imshow a single grayscale slice, background zeroed outside brain2d.

    data2d, brain2d are 2-D arrays already sliced+oriented for display
    (see slicing.py). If crop_bbox is given (see slicing.square_crop_bbox),
    the axes view is zoomed to it instead of showing the full field of
    view. Returns the AxesImage for colorbar attachment.
    """
    display = np.where(brain2d, data2d, np.nan)
    im = ax.imshow(display, cmap="gray", vmin=vmin, vmax=vmax, origin="lower")
    ax.set_facecolor("black")
    ax.set_xticks([])
    ax.set_yticks([])
    if crop_bbox is not None:
        x0, x1, y0, y1 = crop_bbox
        ax.set_xlim(x0, x1)
        ax.set_ylim(y0, y1)
    return im


def style_dark_figure(fig, text_color: str = "0.92"):
    """Black figure background, light text, for brain-slice montages
    (Figs 1 and 4) -- matches conventional neuroimaging display and avoids
    a bright white canvas fighting the black-background panels."""
    fig.patch.set_facecolor("black")
    for ax in fig.axes:
        ax.title.set_color(text_color)
        ax.xaxis.label.set_color(text_color)
        ax.yaxis.label.set_color(text_color)
        ax.tick_params(colors=text_color, labelcolor=text_color)
        for text in ax.texts:
            text.set_color(text_color)
        if ax.images:
            # brain-slice panel: borderless, blends into the black canvas
            for spine in ax.spines.values():
                spine.set_visible(False)
        else:
            # colorbar axis: keep a subtle light outline
            for spine in ax.spines.values():
                spine.set_color(text_color)
                spine.set_linewidth(0.6)
        offset = ax.yaxis.get_offset_text()
        offset.set_color(text_color)
    if fig._suptitle is not None:
        fig._suptitle.set_color(text_color)
