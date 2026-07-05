#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# segment_tumorsynth.sh — run mri_TumorSynth on all 10 edema subjects.
#
# Per subject:
#   1. skull-strip each available sequence (mri_synthstrip)
#   2. register each to the SRI-24 template (mri_robust_register --resample)
#   3. --wholetumor  : healthy tissue + whole-tumor label map (multi-sequence)
#   4. isolate the tumor ROI (label 18) and mask the T1CE by it
#   5. --innertumor  : BraTS ET / NET / necrosis substructures
#
# Setup lives in docs/tumorsynth_install.md. This script assumes:
#   - FreeSurfer >= 7.4.1 sourced (mri_synthstrip, mri_robust_register,
#     mri_binarize on PATH) and the mri_tumorsynth launcher on PATH
#   - the nnUNet_v1.7 conda env exists with Task002_Tumor + Task003_InnerTumor
#   - FSL's fslmaths on PATH (for the tumor-ROI masking step)
#
# Usage:
#   bash scripts/segment_tumorsynth.sh            # all subjects
#   bash scripts/segment_tumorsynth.sh 001 187    # a subset
# ---------------------------------------------------------------------------
set -euo pipefail

# ---- configuration --------------------------------------------------------
DATA_ROOT="/mnt/c/miscellaneous/coding_projects/python/mri_processing/data_storage/data/edema"
OUT_ROOT="${DATA_ROOT}/derivatives/tumorsynth"

CONDA_SH="${CONDA_SH:-/home/jaden/miniconda3/etc/profile.d/conda.sh}"
NNUNET_ENV="${NNUNET_ENV:-nnUNet_v1.7}"

# SRI-24 template that --wholetumor expects the input registered to.
# FreeSurfer ships one under $FREESURFER_HOME; override if yours lives elsewhere.
SRI24_TEMPLATE="${SRI24_TEMPLATE:-${FREESURFER_HOME:-/usr/local/freesurfer}/average/SRI24/T1.nii.gz}"

# Use CPU if no GPU is visible; --threads is ignored on GPU.
TS_THREADS="${TS_THREADS:-4}"

ALL_SUBJECTS=(001 002 003 011 132 150 175 187 196 260)
SUBJECTS=("${@:-${ALL_SUBJECTS[@]}}")
if [[ $# -gt 0 ]]; then SUBJECTS=("$@"); else SUBJECTS=("${ALL_SUBJECTS[@]}"); fi

# ---- sanity checks --------------------------------------------------------
[[ -f "$CONDA_SH" ]] || { echo "!! conda.sh not found at $CONDA_SH" >&2; exit 1; }
# shellcheck disable=SC1090
source "$CONDA_SH"
conda activate "$NNUNET_ENV"

for tool in mri_synthstrip mri_robust_register mri_binarize mri_tumorsynth fslmaths; do
  command -v "$tool" >/dev/null 2>&1 || { echo "!! '$tool' not on PATH (see docs/tumorsynth_install.md)" >&2; exit 1; }
done
[[ -f "$SRI24_TEMPLATE" ]] || { echo "!! SRI-24 template not found: $SRI24_TEMPLATE (set SRI24_TEMPLATE=...)" >&2; exit 1; }

# Decide GPU vs CPU once.
GPU_FLAG=()
if ! command -v nvidia-smi >/dev/null 2>&1 || ! nvidia-smi -L >/dev/null 2>&1; then
  echo ".. no GPU detected — running mri_tumorsynth on CPU (--threads $TS_THREADS)"
  GPU_FLAG=(--cpu --threads "$TS_THREADS")
fi

# ---- helpers --------------------------------------------------------------
# Skull-strip + register one sequence to SRI-24. Echoes the SRI-24-space path.
prep_sequence() {  # args: <in_nii> <out_prefix>
  local in_nii="$1" out_prefix="$2"
  local stripped="${out_prefix}_brain.nii.gz"
  local moved="${out_prefix}_sri24.nii.gz"
  local xfm="${out_prefix}_to-sri24.lta"

  mri_synthstrip -i "$in_nii" -o "$stripped" >/dev/null
  mri_robust_register \
    --mov "$stripped" --dst "$SRI24_TEMPLATE" \
    --lta "$xfm" --mapmov "$moved" --satit --iscale >/dev/null
  echo "$moved"
}

# First existing file from a list of candidate basenames in <anat_dir>.
first_existing() {  # args: <anat_dir> <name...>
  local dir="$1"; shift
  for name in "$@"; do
    [[ -f "$dir/$name" ]] && { echo "$dir/$name"; return 0; }
  done
  return 1
}

# ---- main loop ------------------------------------------------------------
for sub in "${SUBJECTS[@]}"; do
  echo "==================================================================="
  echo "== sub-${sub}"
  anat="${DATA_ROOT}/sub-${sub}/anat"
  work="${OUT_ROOT}/sub-${sub}"
  mkdir -p "$work"

  # Primary = T1CE (contrast). Fall back to plain T1w if no contrast scan.
  if ! t1ce=$(first_existing "$anat" "sub-${sub}_ce-Gd_T1w.nii.gz" "sub-${sub}_T1w.nii.gz"); then
    echo "!! sub-${sub}: no T1CE/T1w anatomical — skipping" >&2
    continue
  fi

  # Build the comma-separated multi-sequence input, primary first.
  inputs=()
  inputs+=("$(prep_sequence "$t1ce" "${work}/sub-${sub}_t1ce")")
  if t2=$(first_existing "$anat" "sub-${sub}_T2w.nii.gz"); then
    inputs+=("$(prep_sequence "$t2" "${work}/sub-${sub}_t2")")
  fi
  if flair=$(first_existing "$anat" "sub-${sub}_FLAIR.nii.gz"); then
    inputs+=("$(prep_sequence "$flair" "${work}/sub-${sub}_flair")")
  fi
  joined_inputs=$(IFS=,; echo "${inputs[*]}")

  # 3. whole tumor + healthy tissue
  whole="${work}/sub-${sub}_desc-wholetumor_dseg.nii.gz"
  echo ".. wholetumor: ${#inputs[@]} sequence(s)"
  mri_tumorsynth --i "$joined_inputs" --o "$whole" --wholetumor "${GPU_FLAG[@]}"

  # 4. isolate tumor ROI (label 18) and mask the primary T1CE
  tumor_mask="${work}/sub-${sub}_desc-tumor_mask.nii.gz"
  tumor_roi="${work}/sub-${sub}_desc-tumorroi.nii.gz"
  mri_binarize --i "$whole" --match 18 --o "$tumor_mask"
  fslmaths "${inputs[0]}" -mul "$tumor_mask" "$tumor_roi"

  # 5. inner tumor substructures (only meaningful if a tumor was found)
  inner="${work}/sub-${sub}_desc-innertumor_dseg.nii.gz"
  if [[ "$(fslstats "$tumor_mask" -V 2>/dev/null | awk '{print $1}')" != "0" ]]; then
    echo ".. innertumor"
    mri_tumorsynth --i "$tumor_roi" --o "$inner" --innertumor "${GPU_FLAG[@]}"
  else
    echo ".. no tumor voxels (label 18 empty) — skipping innertumor"
  fi

  echo "== sub-${sub} done -> ${work}"
done

echo "All requested subjects processed. Outputs under ${OUT_ROOT}/"
