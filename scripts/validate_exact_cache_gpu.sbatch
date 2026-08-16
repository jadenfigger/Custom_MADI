#!/bin/bash
#SBATCH --job-name=madi_cache_gold
#SBATCH -p htc
#SBATCH -q public
#SBATCH --array=0-0
#SBATCH -N 1
#SBATCH -c 2
#SBATCH -G a100.20gb:1
#SBATCH --mem=24G
#SBATCH -t 0-01:00:00
#SBATCH -o logs/madi_cache_gold_%A_%a.out
#SBATCH -e logs/madi_cache_gold_%A_%a.err
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=jfigger@asu.edu
#SBATCH --export=NONE

# Small GPU-only equivalence gate for the exact classifier cache.  This is a
# fixed golden replay, not a library build.  Do not add set -e, set -u, or
# pipefail: Sol's Mamba activation is not compatible with them.

module load mamba/latest
source activate madiEnv

PYTHON_BIN="$(command -v python)"
echo "${PYTHON_BIN}"
if [[ "${PYTHON_BIN}" != */madiEnv/bin/python ]]; then
  echo "ERROR: source activate madiEnv did not select madiEnv's Python; aborting." >&2
  exit 2
fi
python -c "import numpy, scipy, numba, nibabel; from numba import cuda; print('imports OK'); print('CUDA:', cuda.is_available()); print('GPU:', cuda.get_current_device().name.decode() if cuda.is_available() else 'NONE')"

if [[ -z "${SLURM_SUBMIT_DIR:-}" ]]; then
  echo "ERROR: SLURM_SUBMIT_DIR is not set; submit this file with sbatch." >&2
  exit 2
fi
cd "${SLURM_SUBMIT_DIR}"
mkdir -p logs sol_outputs

OUTPUT_DIR="sol_outputs/exact_cache_gpu_golden_${SLURM_JOB_ID}"
python -m scripts.run_gpu_golden_check \
  --golden tests/physics_audit/data/cpu_gpu_golden_v1.npz \
  --classifier exact_cached \
  --output-dir "${OUTPUT_DIR}"
