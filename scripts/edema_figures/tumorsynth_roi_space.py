#!/usr/bin/env python3
"""
tumorsynth_roi_space.py — resample TumorSynth's SRI-24-space tumor/tissue
labels (scripts/segment_tumorsynth.sh output) into each subject's native DWI
grid, mirroring what roi_space.py does for the T1-space edema/contra masks.

Transform chain (two hops, since TumorSynth output lives in template space
but the existing DWI<->T1 transform is defined against each subject's own
brain-extracted T1w):

  1. SRI-24 -> native T1w-brain
     Register the subject's already skull-stripped
     derivatives/preproc/sub-XXX/anat/sub-XXX_desc-brain_T1w.nii.gz to the
     SRI-24 template with mri_robust_register, invert that LTA with
     lta_convert, and use mri_vol2vol --nearest to pull each SRI-24-space
     label map back into T1w-brain space (nearest-neighbour: these are
     integer label maps, not thin single-slice masks, so no threshold trick
     is needed here unlike roi_space.py).

  2. native T1w-brain -> native DWI
     Reuse the existing sub-XXX_desc-b0-2-t1_transform.mat exactly as
     roi_space.py does: invert with convert_xfm, apply with flirt
     -interp nearestneighbour.

ROIs produced (whichever a subject actually has TumorSynth output for):
  - "tumor"       : whole-tumor label (wholetumor dseg == 18) -- TC+NET+edema
  - "tumor-core"  : innertumor label 1 (Tumor Core, TC)
  - "tumor-net"   : innertumor label 2 (Non-Enhancing Tumor, NET)
  - "tumor-edema" : innertumor label 3 (Edema) -- the actual edematous rim,
                    in proper 3D, for any subject with TumorSynth output
                    (kept as a separate desc from the old informal
                    single-slice "edema" mask so it doesn't silently
                    overwrite what fig1/fig3/fig4/fig6 already rely on)
  Label order per mri_tumorsynth --help (verbatim): "Outputs BraTS-compliant
  subclasses: Tumor Core (TC), Non-Enhancing Tumor (NET), and Edema" -> 1/2/3.
  - "grey-matter" / "white-matter" : wholetumor's own healthy-tissue labels
                    (Cerebral-Cortex=2, Cerebral-White-Matter=1), present for
                    every subject with a wholetumor output, tumor or not.

Usage
-----
    conda activate mri   # needs FreeSurfer + FSL sourced in the same shell
    export FREESURFER_HOME=/usr/local/freesurfer/8.2.0
    source "$FREESURFER_HOME/SetUpFreeSurfer.sh"
    PYTHONPATH=. python -m scripts.edema_figures.tumorsynth_roi_space
    PYTHONPATH=. python -m scripts.edema_figures.tumorsynth_roi_space --subjects 001 003
    PYTHONPATH=. python -m scripts.edema_figures.tumorsynth_roi_space --force
"""

import argparse
import glob
import os
import shutil
import subprocess
import tempfile

import nibabel as nib
import numpy as np

from . import config

WHOLE_TUMOR_LABEL = 18
# mri_tumorsynth --help (verbatim, in this order): "Outputs BraTS-compliant
# subclasses: Tumor Core (TC), Non-Enhancing Tumor (NET), and Edema." -> labels
# 1/2/3 in that order. Label 3 IS edema directly -- no derived subtraction
# needed (an earlier version of this script incorrectly treated 1+2+3 as a
# combined "tumor-core" and derived "edema" as whole-tumor minus that union,
# based on a wrong label transcription from memory; fixed here against the
# actual help text).
INNER_TUMOR_LABELS = {"tumor-core": 1, "tumor-net": 2, "tumor-edema": 3}

# Healthy-tissue labels the --wholetumor pass already computes alongside the
# tumor label (see docs/tumorsynth_install.md's full 17-label table) but which
# were previously left unextracted. Available for every subject with a
# wholetumor output, independent of whether a tumor was found.
WHOLE_TUMOR_TISSUE_LABELS = {"grey-matter": 2, "white-matter": 1}

REQUIRED_TOOLS = ["mri_robust_register", "lta_convert", "mri_vol2vol", "convert_xfm", "flirt"]


def _t1w_brain_path(subject: str) -> str:
    return os.path.join(config.PREPROC_ROOT, f"sub-{subject}", "anat", f"sub-{subject}_desc-brain_T1w.nii.gz")


def _b0_2_t1_path(subject: str) -> str:
    return os.path.join(config.ROIS_ROOT, f"sub-{subject}", f"sub-{subject}_desc-b0-2-t1_transform.mat")


def _reference_path(subject: str) -> str:
    """Any MADI parameter map defines the DWI-native grid (see roi_space.py)."""
    return os.path.join(config.MADI_ROOT, f"sub-{subject}", "dwi", "method-MAP", "kio_map.nii.gz")


def _wholetumor_path(subject: str) -> str:
    return os.path.join(config.TUMORSYNTH_ROOT, f"sub-{subject}", f"sub-{subject}_desc-wholetumor_dseg.nii.gz")


def _innertumor_path(subject: str) -> str:
    return os.path.join(config.TUMORSYNTH_ROOT, f"sub-{subject}", f"sub-{subject}_desc-innertumor_dseg.nii.gz")


def _t1w_to_sri24_lta_path(subject: str) -> str:
    return os.path.join(config.ROIS_ROOT, f"sub-{subject}", f"sub-{subject}_desc-t1w-2-sri24_transform.lta")


def _dwi_mask_out_path(subject: str, roi: str) -> str:
    return os.path.join(config.ROIS_ROOT, f"sub-{subject}", f"sub-{subject}_desc-{roi}-dwi_mask.nii.gz")


def _check_tools():
    missing = [t for t in REQUIRED_TOOLS if shutil.which(t) is None]
    if missing:
        raise SystemExit(
            f"missing tool(s) on PATH: {missing} -- source FreeSurfer/FSL first "
            "(see docs/tumorsynth_install.md)"
        )
    if shutil.which("flirt") is None or shutil.which("convert_xfm") is None:
        raise SystemExit("FSL (flirt/convert_xfm) not found on PATH")


def _register_t1w_to_sri24(subject: str, force: bool = False) -> str:
    """Register the subject's T1w-brain to the SRI-24 template; return the
    (cached) inverse LTA path (SRI-24 -> T1w-brain)."""
    t1w_brain = _t1w_brain_path(subject)
    if not os.path.exists(t1w_brain):
        raise FileNotFoundError(f"no T1w-brain for sub-{subject}: {t1w_brain}")

    lta = _t1w_to_sri24_lta_path(subject)
    inv_lta = lta.replace("_transform.lta", "_transform-inv.lta")
    if not force and os.path.exists(inv_lta) and os.path.getmtime(inv_lta) > os.path.getmtime(t1w_brain):
        return inv_lta

    with tempfile.TemporaryDirectory() as tmp:
        moved = os.path.join(tmp, "moved.nii.gz")
        subprocess.run(
            ["mri_robust_register", "--mov", t1w_brain, "--dst", config.SRI24_TEMPLATE,
             "--lta", lta, "--mapmov", moved, "--satit", "--iscale"],
            check=True, capture_output=True,
        )
    subprocess.run(
        ["lta_convert", "--inlta", lta, "--outlta", inv_lta, "--invert"],
        check=True, capture_output=True,
    )
    return inv_lta


def _sri24_label_to_t1w(subject: str, label_path: str, inv_lta: str, out_path: str):
    subprocess.run(
        ["mri_vol2vol", "--mov", label_path, "--targ", _t1w_brain_path(subject),
         "--lta", inv_lta, "--nearest", "--o", out_path],
        check=True, capture_output=True,
    )


def _t1w_mask_to_dwi(subject: str, t1w_mask_path: str, out_path: str):
    xfm = _b0_2_t1_path(subject)
    if not os.path.exists(xfm):
        raise FileNotFoundError(f"no b0-2-t1 transform for sub-{subject}: {xfm}")
    ref = _reference_path(subject)
    if not os.path.exists(ref):
        raise FileNotFoundError(f"no MADI reference volume for sub-{subject}: {ref}")

    with tempfile.TemporaryDirectory() as tmp:
        inv = os.path.join(tmp, "t1_2_dwi.mat")
        subprocess.run(["convert_xfm", "-omat", inv, "-inverse", xfm], check=True, capture_output=True)
        subprocess.run(
            ["flirt", "-in", t1w_mask_path, "-ref", ref, "-applyxfm", "-init", inv,
             "-interp", "nearestneighbour", "-out", out_path],
            check=True, capture_output=True,
        )


def _save_bool_mask(data: np.ndarray, ref_img: nib.Nifti1Image, out_path: str):
    nib.save(nib.Nifti1Image(data.astype(np.uint8), ref_img.affine, ref_img.header), out_path)


def resample_subject(subject: str, force: bool = False) -> dict:
    """Produce every ROI this subject has TumorSynth output for. Returns
    {roi_name: dwi_mask_path} for ROIs actually written."""
    whole_path = _wholetumor_path(subject)
    if not os.path.exists(whole_path):
        print(f"[tumorsynth_roi_space] skip sub-{subject}: no wholetumor output")
        return {}

    out_dir = os.path.join(config.ROIS_ROOT, f"sub-{subject}")
    os.makedirs(out_dir, exist_ok=True)

    all_out = {roi: _dwi_mask_out_path(subject, roi)
               for roi in ["tumor", *INNER_TUMOR_LABELS, *WHOLE_TUMOR_TISSUE_LABELS]}
    if not force and all(os.path.exists(p) and os.path.getmtime(p) > os.path.getmtime(whole_path)
                          for p in [all_out["tumor"]]):
        # "tumor" alone is cheap/always-present; only used as a fast skip when
        # the whole set was already regenerated after the last TumorSynth run.
        if all(os.path.exists(p) for p in all_out.values()) or not os.path.exists(_innertumor_path(subject)):
            print(f"[tumorsynth_roi_space] sub-{subject}: up to date, skipping")
            return {p: q for p, q in all_out.items() if os.path.exists(q)}

    inv_lta = _register_t1w_to_sri24(subject, force=force)
    t1w_img = nib.load(_t1w_brain_path(subject))

    written = {}
    with tempfile.TemporaryDirectory() as tmp:
        whole_t1w = os.path.join(tmp, "whole_t1w.nii.gz")
        _sri24_label_to_t1w(subject, whole_path, inv_lta, whole_t1w)
        whole_data = np.asarray(nib.load(whole_t1w).dataobj)
        tumor_mask = whole_data == WHOLE_TUMOR_LABEL
        tissue_masks = {roi: whole_data == label for roi, label in WHOLE_TUMOR_TISSUE_LABELS.items()}

        inner_path = _innertumor_path(subject)
        inner_masks = {}
        if os.path.exists(inner_path):
            inner_t1w = os.path.join(tmp, "inner_t1w.nii.gz")
            _sri24_label_to_t1w(subject, inner_path, inv_lta, inner_t1w)
            inner_data = np.asarray(nib.load(inner_t1w).dataobj)
            for roi, label in INNER_TUMOR_LABELS.items():
                inner_masks[roi] = inner_data == label

        t1w_rois = {"tumor": tumor_mask, **inner_masks, **tissue_masks}

        for roi, mask in t1w_rois.items():
            t1w_tmp = os.path.join(tmp, f"{roi}_t1w.nii.gz")
            _save_bool_mask(mask, t1w_img, t1w_tmp)
            out = _dwi_mask_out_path(subject, roi)
            _t1w_mask_to_dwi(subject, t1w_tmp, out)
            written[roi] = out

    return written


def resample_all(subjects=None, force: bool = False) -> dict:
    if subjects is None:
        subjects = sorted(
            os.path.basename(p).replace("sub-", "")
            for p in glob.glob(os.path.join(config.TUMORSYNTH_ROOT, "sub-*"))
        )
    results = {}
    for subject in subjects:
        try:
            for roi, path in resample_subject(subject, force=force).items():
                results[(subject, roi)] = path
        except (FileNotFoundError, subprocess.CalledProcessError) as e:
            print(f"[tumorsynth_roi_space] sub-{subject} failed: {e}")
    return results


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--subjects", nargs="*", default=None,
                     help="subject ids (no 'sub-' prefix); default = all with TumorSynth output")
    ap.add_argument("--force", action="store_true", help="regenerate even if cached output looks current")
    args = ap.parse_args()

    _check_tools()
    results = resample_all(subjects=args.subjects, force=args.force)
    for (subject, roi), path in sorted(results.items()):
        print(f"sub-{subject} {roi}: {path}")


if __name__ == "__main__":
    main()
