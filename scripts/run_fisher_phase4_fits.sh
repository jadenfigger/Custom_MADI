#!/usr/bin/env bash
# Phase 4 fit arms: the unrealistic-cell-volume pathology (plan section 7).
#
# One acquisition, one fixed configuration, one condition varied per arm:
#
#   baseline                 the executed madi_output_glioma_v4.0 configuration, re-run so
#                            every arm comes from one code state (and checked against v4.0)
#   fit_s0                   H4, item 4.1: S0 fitted per voxel (--fit-s0)
#   trust_floor_column       H3, item 4.2: pre-registered S/S0 floor, acquisition columns
#   trust_floor_candidate    H3, item 4.2: pre-registered S/S0 floor, candidate entries
#
# each under MAP and Bayes.  The Bayes free-S0 arm follows the calibration procedure
# docs/fitting_methods.md prescribes: sigma_m is not comparable between the fixed- and
# free-S0 branches, so it is matched on the baseline Bayes run's median n_eff
# (--target-n-eff) rather than left at the Rician auto value.
#
# Runs are SEQUENTIAL by design: the fitter holds the 15 GB library in memory and the
# reference host has 11 GiB of RAM.  A completed arm (sidecar status "complete") is
# skipped, so the script is resumable.
#
# Usage: scripts/run_fisher_phase4_fits.sh <subject> [out_root]
set -euo pipefail

SUBJECT="${1:?usage: run_fisher_phase4_fits.sh <subject> [out_root]}"
OUT_ROOT="${2:-data/outputs/fisher_phase4_${SUBJECT}}"
DS=/mnt/c/miscellaneous/coding_projects/python/mri_processing/data_storage/data/Mayo_Glioma/derivatives/preproc/${SUBJECT}/dwi
LIB=data/libraries/madi_dense_universal_remediated.npz
INPUT="50:${DS}/${SUBJECT}_desc-madi-input_dwi.nii.gz:${DS}/${SUBJECT}_desc-madi-input_dwi.bval:${DS}/${SUBJECT}_desc-madi-input_dwi.bvec"
COMMON=(--fit --input "$INPUT" --mask "${DS}/${SUBJECT}_desc-brain_mask.nii.gz"
        --library "$LIB" --small-delta 20.0 --rician-correct)

export PYTHONPATH=. PYTHONUNBUFFERED=1
mkdir -p "$OUT_ROOT"

complete() {  # <arm_dir> <run_name>
    local sidecar="$1/$2.json"
    [[ -f "$sidecar" ]] && python -c "import json,sys; sys.exit(0 if json.load(open('$sidecar')).get('status')=='completed' else 1)"
}

run_arm() {  # <run_name> <method> [extra flags...]
    local name="$1" method="$2"; shift 2
    local dir="$OUT_ROOT/$name"
    if complete "$dir" "$name"; then echo "[skip] $name (complete)"; return; fi
    mkdir -p "$dir"
    echo "[run ] $name  $(date -Is)"
    python scripts/fit_data.py "${COMMON[@]}" --method "$method" --out "$dir" --run-name "$name" "$@" \
        > "$dir/$name.stdout.log" 2>&1
    echo "[done] $name  $(date -Is)"
}

run_arm baseline_map                 map
# The one fitter code path no smoke run exercised goes early, so a defect in it
# surfaces in the second arm rather than the seventh.
run_arm trust_floor_candidate_map    map   --trust-floor --trust-floor-mode candidate
run_arm trust_floor_column_map       map   --trust-floor --trust-floor-mode column
run_arm fit_s0_map                   map   --fit-s0
run_arm baseline_bayes               bayes
# The run sidecar does not record n_eff, so the median is taken from the written map,
# over the same brain mask the fit used (the fitter's own printed median is over the
# fitted voxels, which are exactly the mask voxels).
N_EFF=$(python - "$OUT_ROOT/baseline_bayes/n_eff.nii.gz" "${DS}/${SUBJECT}_desc-brain_mask.nii.gz" <<'PY'
import sys
import nibabel as nib
import numpy as np
n_eff = np.asarray(nib.load(sys.argv[1]).dataobj, dtype=float)
mask = np.asarray(nib.load(sys.argv[2]).dataobj).astype(bool)
print(float(np.median(n_eff[mask])))
PY
)
echo "[info] baseline Bayes median n_eff = $N_EFF (target for the free-S0 Bayes arm)"
run_arm fit_s0_bayes                 bayes --fit-s0 --target-n-eff "$N_EFF"
run_arm trust_floor_column_bayes     bayes --trust-floor --trust-floor-mode column
run_arm trust_floor_candidate_bayes  bayes --trust-floor --trust-floor-mode candidate
echo "[all ] $(date -Is)"
