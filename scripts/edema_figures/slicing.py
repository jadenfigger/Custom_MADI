"""
slicing.py — consistent axial-slice extraction/orientation for display.

NIfTI volumes here are (x, y, z) with z the axial (through-slice) axis.
`axial(volume, z)` returns a 2-D array transposed for `imshow(..., origin=
"lower")` to show it the conventional way (rows = posterior->anterior).
Left/right handedness has not been independently verified against the
paper's convention -- check one output against edema_original/*/models/
ROIs/T1_brain.nii.gz for the same subject/lesion before trusting L/R.
"""

import numpy as np


def axial(volume: np.ndarray, z: int) -> np.ndarray:
    return volume[:, :, z].T


def square_crop_bbox(mask2d: np.ndarray, margin_frac: float = 0.12) -> tuple:
    """Tight square (xmin, xmax, ymin, ymax) around a 2-D boolean mask's
    content, padded by margin_frac of the larger side. Used to zoom
    imshow panels in on the brain instead of showing the whole (mostly
    empty) field of view. Coordinates are in the same index space as the
    2-D array (i.e. directly usable with ax.set_xlim/set_ylim after
    plotting with imshow's default extent)."""
    ys, xs = np.nonzero(mask2d)
    if xs.size == 0:
        h, w = mask2d.shape
        return (0, w, 0, h)
    x0, x1 = xs.min(), xs.max()
    y0, y1 = ys.min(), ys.max()
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    side = max(x1 - x0, y1 - y0) * (1.0 + margin_frac)
    half = side / 2.0
    return (cx - half, cx + half, cy - half, cy + half)
