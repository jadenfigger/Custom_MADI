"""
loaders.py — load MADI parameter maps, brain masks, and ROI masks for the
paper-figure replication scripts.
"""

import os
from typing import Optional, Tuple

import nibabel as nib
import numpy as np

from . import config


def _load_nii(path: str) -> Tuple[np.ndarray, np.ndarray]:
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    img = nib.load(path)
    return np.asarray(img.dataobj), img.affine


def param_map_path(subject: str, method: str, param: str) -> str:
    stem = config.map_file_stem(method)
    return os.path.join(
        config.MADI_ROOT, f"sub-{subject}", "dwi", f"method-{method}",
        f"{param}_{stem}.nii.gz",
    )


def load_param_map(subject: str, method: str, param: str) -> np.ndarray:
    """Load one of kio/rho/V for (subject, method) in native DWI space."""
    path = param_map_path(subject, method, param)
    data, _ = _load_nii(path)
    return data


def brain_mask_path(subject: str) -> Optional[str]:
    dwi_dir = os.path.join(config.PREPROC_ROOT, f"sub-{subject}", "dwi")
    for stem in config.BRAIN_MASK_PRIORITY:
        for ext in (".nii.gz", ".nii"):
            p = os.path.join(dwi_dir, f"sub-{subject}_{stem}{ext}")
            if os.path.exists(p):
                return p
    return None


def load_brain_mask(subject: str, shape: tuple) -> np.ndarray:
    """Boolean brain mask in native DWI space, or all-True if none is found."""
    path = brain_mask_path(subject)
    if path is None:
        return np.ones(shape, dtype=bool)
    data, _ = _load_nii(path)
    return data.astype(bool)


def load_flair_map(subject: str) -> np.ndarray:
    """Load sub-XXX's FLAIR, already resampled into native DWI space by
    flair_space.py."""
    from . import flair_space
    path = flair_space.flair_dwi_path(subject)
    data, _ = _load_nii(path)
    return data


def dwi_mask_path(subject: str, desc: str) -> str:
    """Path to the DWI-space-resampled ROI mask (produced by roi_space.py)."""
    return os.path.join(
        config.ROIS_ROOT, f"sub-{subject}", f"sub-{subject}_desc-{desc}-dwi_mask.nii.gz",
    )


def load_dwi_mask(subject: str, desc: str) -> np.ndarray:
    """Boolean ROI mask (contra/edema) already resampled into DWI space."""
    path = dwi_mask_path(subject, desc)
    data, _ = _load_nii(path)
    return data.astype(bool)


def mask_slice_index(mask: np.ndarray) -> int:
    """Return the z-index with the most nonzero voxels in a (mostly)
    single-slice mask."""
    counts = mask.sum(axis=(0, 1))
    if not counts.any():
        raise ValueError("mask has no nonzero voxels")
    return int(np.argmax(counts))


def roi_mean_std(values: np.ndarray, mask: np.ndarray) -> Tuple[float, float]:
    """Mean and std of `values` within `mask` (both full 3D volumes)."""
    v = values[mask]
    v = v[np.isfinite(v)]
    if v.size == 0:
        return float("nan"), float("nan")
    return float(v.mean()), float(v.std())
