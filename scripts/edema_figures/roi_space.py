#!/usr/bin/env python3
"""
roi_space.py — resample T1-space ROI masks (contra, edema) into each
subject's native DWI/MADI grid, using the per-subject b0->T1 affine
already staged in derivatives/rois/.

The mask files in derivatives/rois/sub-XXX/ are drawn in T1 space
(256x256x208, 1mm) but the MADI kio/rho/V maps are in native DWI space
(104x104x56, ~2.46mm, 2.5mm through-plane). `sub-XXX_desc-b0-2-t1_transform.mat`
maps DWI coordinates -> T1 coordinates (confirmed identical to the source
b0_2T1.mat), so bringing a T1-space mask into DWI space requires its inverse.

These masks are drawn as a SINGLE 1mm-thick T1 slice. Nearest-neighbour
resampling onto the much coarser 2.5mm DWI z-grid point-samples the volume
and mostly misses that thin band entirely (empirically: 1572 T1 voxels ->
27 DWI voxels, centroid off by 14.5mm for sub-187's edema mask). Trilinear
interpolation followed by a low threshold instead area-averages the thin
mask into the coarser grid, which correctly recovers its in-plane extent
(empirically: 472 DWI voxels, in-plane area matching the T1 mask's ~2400mm^2,
centroid off by ~5mm). Because the source is thinner than one target voxel,
trilinear values never exceed ~0.28 here, so the usual 0.5 threshold is not
usable -- THRESHOLD below is calibrated for this specific thickness mismatch.

Usage
-----
    python -m scripts.edema_figures.roi_space --subjects 001 002 003 011 187
    python -m scripts.edema_figures.roi_space --force   # regenerate all
"""

import argparse
import os
import shutil
import subprocess
import tempfile

import nibabel as nib
import numpy as np

from . import config

DESCS = ["contra", "edema"]

# See module docstring: trilinear values here never exceed ~0.28 because the
# source mask is thinner than a DWI voxel, so this must stay well below 0.5.
THRESHOLD = 0.1


def _t1_mask_path(subject: str, desc: str) -> str:
    return os.path.join(config.ROIS_ROOT, f"sub-{subject}", f"sub-{subject}_desc-{desc}_mask.nii.gz")


def _transform_path(subject: str) -> str:
    return os.path.join(config.ROIS_ROOT, f"sub-{subject}", f"sub-{subject}_desc-b0-2-t1_transform.mat")


def _reference_path(subject: str) -> str:
    """Any MADI parameter map defines the DWI-native grid; MAP/kio is
    always present and cheapest to have around."""
    return os.path.join(config.MADI_ROOT, f"sub-{subject}", "dwi", "method-MAP", "kio_map.nii.gz")


def _need_regen(src: str, xfm: str, out: str) -> bool:
    if not os.path.exists(out):
        return True
    out_mtime = os.path.getmtime(out)
    return os.path.getmtime(src) > out_mtime or os.path.getmtime(xfm) > out_mtime


def resample_mask(subject: str, desc: str, force: bool = False) -> str:
    """Resample sub-XXX's T1-space `desc` mask into DWI space. Returns the
    output path. No-op (just returns the path) if the source mask doesn't
    exist for this subject, or a fresh cached copy already exists."""
    src = _t1_mask_path(subject, desc)
    if not os.path.exists(src):
        raise FileNotFoundError(
            f"no T1-space '{desc}' mask for sub-{subject}: {src}"
        )
    xfm = _transform_path(subject)
    if not os.path.exists(xfm):
        raise FileNotFoundError(f"no b0-2-t1 transform for sub-{subject}: {xfm}")
    ref = _reference_path(subject)
    if not os.path.exists(ref):
        raise FileNotFoundError(f"no MADI reference volume for sub-{subject}: {ref}")

    out = os.path.join(config.ROIS_ROOT, f"sub-{subject}", f"sub-{subject}_desc-{desc}-dwi_mask.nii.gz")
    if not force and not _need_regen(src, xfm, out):
        return out

    with tempfile.TemporaryDirectory() as tmp:
        inv = os.path.join(tmp, "t1_2_dwi.mat")
        soft = os.path.join(tmp, "resampled_trilinear.nii.gz")
        subprocess.run(
            ["convert_xfm", "-omat", inv, "-inverse", xfm],
            check=True, capture_output=True,
        )
        subprocess.run(
            [
                "flirt", "-in", src, "-ref", ref, "-applyxfm", "-init", inv,
                "-interp", "trilinear", "-out", soft,
            ],
            check=True, capture_output=True,
        )
        # Binarize at THRESHOLD (see module docstring: the source mask is
        # thinner than a DWI voxel, so trilinear partial-volume values here
        # never approach 0.5).
        img = nib.load(soft)
        data = (np.asarray(img.dataobj) > THRESHOLD).astype(np.uint8)
        nib.save(nib.Nifti1Image(data, img.affine, img.header), out)
    return out


def resample_all(subjects=None, force: bool = False) -> dict:
    """Resample contra + edema masks for every subject that has them.
    Returns {(subject, desc): output_path} for masks actually produced."""
    results = {}
    contra_subjects = subjects or config.CONTRA_MASK_SUBJECTS
    edema_subjects = subjects or config.EDEMA_MASK_SUBJECTS
    for subject in contra_subjects:
        if subject in config.CONTRA_MASK_SUBJECTS:
            try:
                results[(subject, "contra")] = resample_mask(subject, "contra", force=force)
            except FileNotFoundError as e:
                print(f"[roi_space] skip sub-{subject} contra: {e}")
    for subject in edema_subjects:
        if subject in config.EDEMA_MASK_SUBJECTS:
            try:
                results[(subject, "edema")] = resample_mask(subject, "edema", force=force)
            except FileNotFoundError as e:
                print(f"[roi_space] skip sub-{subject} edema: {e}")
    return results


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--subjects", nargs="*", default=None,
                     help="subject ids (no 'sub-' prefix); default = all with masks")
    ap.add_argument("--force", action="store_true", help="regenerate even if cached output looks current")
    args = ap.parse_args()

    if shutil.which("flirt") is None or shutil.which("convert_xfm") is None:
        raise SystemExit("FSL (flirt/convert_xfm) not found on PATH")

    results = resample_all(subjects=args.subjects, force=args.force)
    for (subject, desc), path in sorted(results.items()):
        print(f"sub-{subject} {desc}: {path}")


if __name__ == "__main__":
    main()
