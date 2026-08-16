#!/bin/bash
#SBATCH --job-name=madi_v5_post
#SBATCH -p public
#SBATCH -q public
#SBATCH -N 1
#SBATCH -c 1
#SBATCH --mem=32G
#SBATCH -t 0-08:00:00
#SBATCH -o logs/madi_v5_post_%j.out
#SBATCH -e logs/madi_v5_post_%j.err
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=jfigger@asu.edu
#SBATCH --export=NONE

# Read-only hashes followed by the documented 369-shard merge and production
# artifact validation. Submit only after every production array task is
# COMPLETE and its logs pass the explicit error scan. This script never
# touches madi_dense_universal.npz and refuses to overwrite a prior remediated
# merge or checksum record. Do not add set -e, set -u, or pipefail: Sol's
# Mamba activation is not compatible with those shell options.

module load mamba/latest
source activate madiEnv

PYTHON_BIN="$(command -v python)"
echo "${PYTHON_BIN}"
if [[ "${PYTHON_BIN}" != */madiEnv/bin/python ]]; then
  echo "ERROR: source activate madiEnv did not select madiEnv's Python; aborting." >&2
  exit 2
fi
python -c "import numpy, scipy, numba, nibabel; print('imports OK')"

if [[ -z "${SLURM_SUBMIT_DIR:-}" ]]; then
  echo "ERROR: SLURM_SUBMIT_DIR is not set; submit this file with sbatch." >&2
  exit 2
fi
cd "${SLURM_SUBMIT_DIR}"

MERGED="libraries/madi_dense_universal_remediated.npz"
CHECKSUMS="logs/madi_dense_universal_remediated_shards.sha256"
VALIDATION="logs/v5_production_validation.json"
COUNT="$(find libraries -maxdepth 1 -type f -name 'madi_dense_universal_remediated.shard*.npz' | awk 'END {print NR}')"
if [[ "${COUNT}" != "369" ]]; then
  echo "ERROR: expected exactly 369 production shards, found ${COUNT}." >&2
  exit 2
fi
if [[ -e "${MERGED}" || -e "${CHECKSUMS}" || -e "${VALIDATION}" ]]; then
  echo "ERROR: post-processing output already exists; refusing to overwrite evidence." >&2
  exit 2
fi

sha256sum libraries/madi_dense_universal_remediated.shard*.npz > "${CHECKSUMS}"
if [[ $? -ne 0 ]]; then
  echo "ERROR: shard hashing failed." >&2
  exit 2
fi

python -m scripts.merge_shards \
  libraries/madi_dense_universal_remediated.shard*.npz \
  --require-shards 369 \
  --out "${MERGED}"
if [[ $? -ne 0 ]]; then
  echo "ERROR: shard merge failed." >&2
  exit 2
fi

python -m scripts.validate_v5_production \
  --artifact "${MERGED}" \
  --output "${VALIDATION}"
if [[ $? -ne 0 ]]; then
  echo "ERROR: merged production validator failed; do not replace the current library." >&2
  exit 2
fi

echo "POST-BUILD PASS: hashes=${CHECKSUMS} merged=${MERGED} validation=${VALIDATION}"
