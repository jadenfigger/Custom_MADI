#!/usr/bin/env python3
"""
flair_space.py — resample native-space FLAIR anatomicals into each subject's
native DWI/MADI grid, for use as a reference column in Figure 1.

Unlike the ROI masks (roi_space.py), FLAIR is not already sitting in the
same T1 grid as the b0-2-t1 transform's source/target -- it's a separate
raw anatomical acquisition (own resolution/orientation, e.g. 192x512x512 for
most subjects here) that has never been registered to anything. So this
does a two-step composition:

  1. flirt FLAIR -> T1w (rigid, same session -- 6 dof) to get a
     flair-to-T1 matrix.
  2. Invert the existing `sub-XXX_desc-b0-2-t1_transform.mat` (DWI -> T1,
     staged in derivatives/rois/ by roi_space.py) to get T1 -> DWI, then
     concatenate: flair -> T1 -> DWI.

The composed matrix is applied once with a single trilinear resample
(avoids the double-interpolation blur of resampling FLAIR -> T1 -> DWI in
two separate steps).

Usage
-----
    python -m scripts.edema_figures.flair_space --subjects 001 002 003 011 187
    python -m scripts.edema_figures.flair_space --force   # regenerate all
"""

import argparse
import os
import shutil
import subprocess
import tempfile

from . import config

FLAIR_COST = "corratio"  # FSL default; FLAIR/T1 contrast is similar enough
                          # (both null CSF-bright/dark differently but share
                          # WM/GM ordering) for corratio to lock on reliably.


def _flair_path(subject: str) -> str:
    return os.path.join(config.DATA_ROOT, f"sub-{subject}", "anat", f"sub-{subject}_FLAIR.nii.gz")


def _t1_path(subject: str) -> str:
    return os.path.join(config.DATA_ROOT, f"sub-{subject}", "anat", f"sub-{subject}_T1w.nii.gz")


def _b0_2_t1_path(subject: str) -> str:
    return os.path.join(config.ROIS_ROOT, f"sub-{subject}", f"sub-{subject}_desc-b0-2-t1_transform.mat")


def _reference_path(subject: str) -> str:
    """Any MADI parameter map defines the DWI-native grid; MAP/kio is
    always present and cheapest to have around."""
    return os.path.join(config.MADI_ROOT, f"sub-{subject}", "dwi", "method-MAP", "kio_map.nii.gz")


def flair_dwi_path(subject: str) -> str:
    return os.path.join(config.FLAIR_ROOT, f"sub-{subject}_desc-flair-dwi.nii.gz")


def _need_regen(srcs, out: str) -> bool:
    if not os.path.exists(out):
        return True
    out_mtime = os.path.getmtime(out)
    return any(os.path.getmtime(s) > out_mtime for s in srcs)


def resample_flair(subject: str, force: bool = False) -> str:
    """Register+resample sub-XXX's native-space FLAIR into DWI space.
    Returns the output path. Raises FileNotFoundError if a required input
    is missing for this subject."""
    flair = _flair_path(subject)
    t1 = _t1_path(subject)
    xfm = _b0_2_t1_path(subject)
    ref = _reference_path(subject)
    for p, label in [(flair, "FLAIR"), (t1, "T1w"), (xfm, "b0-2-t1 transform"), (ref, "MADI reference")]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"no {label} for sub-{subject}: {p}")

    out = flair_dwi_path(subject)
    if not force and not _need_regen([flair, t1, xfm], out):
        return out

    os.makedirs(config.FLAIR_ROOT, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        flair_2_t1 = os.path.join(tmp, "flair_2_t1.mat")
        t1_2_dwi = os.path.join(tmp, "t1_2_dwi.mat")
        flair_2_dwi = os.path.join(tmp, "flair_2_dwi.mat")
        flair_in_t1 = os.path.join(tmp, "flair_in_t1.nii.gz")

        subprocess.run(
            [
                "flirt", "-in", flair, "-ref", t1, "-out", flair_in_t1,
                "-omat", flair_2_t1, "-dof", "6", "-cost", FLAIR_COST,
            ],
            check=True, capture_output=True,
        )
        subprocess.run(
            ["convert_xfm", "-omat", t1_2_dwi, "-inverse", xfm],
            check=True, capture_output=True,
        )
        subprocess.run(
            ["convert_xfm", "-omat", flair_2_dwi, "-concat", t1_2_dwi, flair_2_t1],
            check=True, capture_output=True,
        )
        subprocess.run(
            [
                "flirt", "-in", flair, "-ref", ref, "-applyxfm", "-init", flair_2_dwi,
                "-interp", "trilinear", "-out", out,
            ],
            check=True, capture_output=True,
        )
    return out


def resample_all(subjects=None, force: bool = False) -> dict:
    """Resample FLAIR for every subject that has one. Returns
    {subject: output_path} for subjects actually produced."""
    results = {}
    for subject in subjects or config.FLAIR_SUBJECTS:
        if subject not in config.FLAIR_SUBJECTS:
            continue
        try:
            results[subject] = resample_flair(subject, force=force)
        except FileNotFoundError as e:
            print(f"[flair_space] skip sub-{subject}: {e}")
    return results


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--subjects", nargs="*", default=None,
                     help="subject ids (no 'sub-' prefix); default = all with FLAIR")
    ap.add_argument("--force", action="store_true", help="regenerate even if cached output looks current")
    args = ap.parse_args()

    if shutil.which("flirt") is None or shutil.which("convert_xfm") is None:
        raise SystemExit("FSL (flirt/convert_xfm) not found on PATH")

    results = resample_all(subjects=args.subjects, force=args.force)
    for subject, path in sorted(results.items()):
        print(f"sub-{subject} flair: {path}")


if __name__ == "__main__":
    main()
