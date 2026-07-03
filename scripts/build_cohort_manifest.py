#!/usr/bin/env python3
"""
build_cohort_manifest.py — Discover subjects under a BIDS-ish preproc tree
and emit a tab-separated manifest listing one row per (subject x fitting
mode), to be consumed by run_cohort_manifest.py.

Fitting modes emitted per subject: MAP, MAP-fits0, BAYES, BAYES-fits0.

Columns whose value is the sentinel "default" mean: run_cohort_manifest.py
will NOT pass the corresponding --flag to fit_data.py at all, so fit_data.py
falls back to its own built-in default for that parameter.

Usage
-----
    python scripts/build_cohort_manifest.py \\
        --preproc-root /path/to/edema/derivatives/preproc \\
        --out-root     /path/to/edema/derivatives/madi \\
        --manifest-out data/manifests/edema_cohort_manifest.tsv


    python scripts/build_cohort_manifest.py --preproc-root /mnt/c/miscellaneous/coding_projects/python/mri_processing/data_storage/data/edema/derivatives/preproc --out-root     /mnt/c/miscellaneous/coding_projects/python/mri_processing/data_storage/data/edema/derivatives/madi --manifest-out /mnt/c/miscellaneous/coding_projects/python/mri_processing/data_storage/data/edema/derivatives/edema_cohort_manifest.tsv
"""

import argparse
import csv
import glob
import os

# Acquisition metadata for this cohort's single-shot clinical protocol.
# Matches the human-clinical library (data/libraries/madi_dense_human.npz):
# Delta = 50 ms, small delta = 20 ms.
DEFAULT_DELTA_MS = 50.0
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_LIBRARY = os.path.join(REPO_ROOT, "data/libraries/madi_dense_human.npz")

# Sentinel written into manifest columns that should use fit_data.py's own
# built-in defaults rather than an explicit CLI override.
SENTINEL = "default"

# Mask filename "desc-" stems to try, in priority order, each tried with
# both .nii.gz and .nii extensions.
DEFAULT_MASK_PRIORITY = ["desc-nodif-brain-clean_mask", "desc-nodif-brain_mask"]

# (method, fit_s0, run_label, sigma_m)
FIT_MODES = [
    ("map",   False, "MAP", "0.02"),
    ("map",   True,  "MAP-fits0", "0.01"),
    ("bayes", False, "BAYES", "0.02"),
    ("bayes", True,  "BAYES-fits0", "0.01")
]

MANIFEST_COLUMNS = [
    "run_id", "subject_id", "method", "fit_s0",
    "dwi_path", "bval_path", "bvec_path", "delta_ms",
    "mask_path", "library", "out_dir", "device",
    "small_delta", "sigma_m", "vi_min", "vi_max", "rho_max",
    "noise_sigma", "noise_bg_dilate_iters",
    "rician_correct", "avg_s0", "log_space",
]


def find_mask(dwi_dir: str, subject_id: str, priority):
    for stem in priority:
        for ext in (".nii.gz", ".nii"):
            candidate = os.path.join(dwi_dir, f"{subject_id}_{stem}{ext}")
            if os.path.exists(candidate):
                return candidate
    return None


def discover_subjects(preproc_root: str):
    """Yield (subject_id, dwi_path, bval_path, bvec_path, dwi_dir) for every
    subject with a *_desc-preproc_dwi.nii.gz under preproc_root/sub-*/dwi/."""
    pattern = os.path.join(preproc_root, "sub-*", "dwi", "*_desc-preproc_dwi.nii.gz")
    for dwi_path in sorted(glob.glob(pattern)):
        dwi_dir = os.path.dirname(dwi_path)
        subject_id = os.path.basename(dwi_dir.rsplit(os.sep, 2)[-2])
        base = dwi_path[: -len(".nii.gz")]
        bval_path = base + ".bval"
        bvec_path = base + ".eddy_rotated_bvecs"
        if not os.path.exists(bval_path):
            print(f"  ⚠ skipping {subject_id}: missing bval {bval_path}")
            continue
        if not os.path.exists(bvec_path):
            print(f"  ⚠ skipping {subject_id}: missing bvecs {bvec_path}")
            continue
        yield subject_id, dwi_path, bval_path, bvec_path, dwi_dir


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preproc-root", required=True,
                     help="BIDS-ish derivatives/preproc root containing sub-*/dwi/.")
    ap.add_argument("--out-root", required=True,
                     help="BIDS-ish derivatives/madi root; per-subject/per-mode "
                          "output dirs are created under here.")
    ap.add_argument("--manifest-out", required=True,
                     help="Path to write the tab-separated manifest to.")
    ap.add_argument("--library", default=DEFAULT_LIBRARY,
                     help=f"MADI library path (default {DEFAULT_LIBRARY}).")
    ap.add_argument("--delta-ms", type=float, default=DEFAULT_DELTA_MS,
                     help=f"Diffusion time Delta [ms] of the acquisition "
                          f"(default {DEFAULT_DELTA_MS}).")
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "gpu"])
    ap.add_argument("--mask-priority", nargs="+", default=DEFAULT_MASK_PRIORITY,
                     help="Mask filename 'desc-*' stems to try in priority "
                          f"order (default {DEFAULT_MASK_PRIORITY}).")
    args = ap.parse_args()

    rows = []
    print(f"Scanning {args.preproc_root} ...")
    for subject_id, dwi_path, bval_path, bvec_path, dwi_dir in discover_subjects(args.preproc_root):
        mask_path = find_mask(dwi_dir, subject_id, args.mask_priority)
        if mask_path is None:
            print(f"  ⚠ skipping {subject_id}: no mask found "
                  f"(tried {args.mask_priority})")
            continue
        print(f"  {subject_id}: dwi={os.path.basename(dwi_path)}  "
              f"mask={os.path.basename(mask_path)}")

        for method, fit_s0, run_label, sigma_m in FIT_MODES:
            out_dir = os.path.join(args.out_root, subject_id, "dwi", f"method-{run_label}")
            rows.append({
                "run_id": f"{subject_id}_{run_label}",
                "subject_id": subject_id,
                "method": method,
                "fit_s0": "true" if fit_s0 else "false",
                "dwi_path": dwi_path,
                "bval_path": bval_path,
                "bvec_path": bvec_path,
                "delta_ms": args.delta_ms,
                "mask_path": mask_path,
                "library": args.library,
                "out_dir": out_dir,
                "device": args.device,
                # --- tunable parameters, left at fit_data.py's own defaults ---
                "small_delta": SENTINEL,
                "sigma_m": sigma_m,
                "vi_min": SENTINEL,
                "vi_max": SENTINEL,
                "rho_max": SENTINEL,
                "noise_sigma": SENTINEL,
                "noise_bg_dilate_iters": SENTINEL,
                "rician_correct": "true",
                "sigma_m": "0.01" if fit_s0 else "0.02",
                "avg_s0": "false",
                "log_space": "false",
            })

    os.makedirs(os.path.dirname(os.path.abspath(args.manifest_out)), exist_ok=True)
    with open(args.manifest_out, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=MANIFEST_COLUMNS, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)

    n_subjects = len(rows) // len(FIT_MODES) if rows else 0
    print(f"\nWrote {len(rows)} run(s) ({n_subjects} subjects x {len(FIT_MODES)} modes) "
          f"to {args.manifest_out}")


if __name__ == "__main__":
    main()
