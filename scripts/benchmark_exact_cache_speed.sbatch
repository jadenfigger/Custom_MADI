#!/bin/bash
#SBATCH --job-name=madi_cache_speed
#SBATCH -p htc
#SBATCH -q public
#SBATCH --array=0-2
#SBATCH -N 1
#SBATCH -c 2
#SBATCH -G a100.20gb:1
#SBATCH --mem=24G
#SBATCH -t 0-01:00:00
#SBATCH -o logs/madi_cache_speed_%A_%a.out
#SBATCH -e logs/madi_cache_speed_%A_%a.err
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=jfigger@asu.edu
#SBATCH --export=NONE

# Full-size GPU-only speed gate for the exact classifier cache.  It replays
# the exact and cached paths with 50,000 walkers at low, centre, and high rho
# and writes JSON evidence only; it never writes a library.  The 2.0 um cache
# bound was selected by job 61530730.  Do not add set -e, set -u, or pipefail:
# Sol's Mamba activation is not compatible with those shell options.

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

case "${SLURM_ARRAY_TASK_ID}" in
  0) DENSITY="low" ;;
  1) DENSITY="center" ;;
  2) DENSITY="high" ;;
  *)
    echo "ERROR: cache-speed task ${SLURM_ARRAY_TASK_ID} is outside [0, 2]." >&2
    exit 2
    ;;
esac

OUTPUT_DIR="sol_outputs/exact_cache_speed_${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID}}"
mkdir -p "${OUTPUT_DIR}"
python -m analysis.v5_fast_cache_benchmark \
  --mode speed \
  --density "${DENSITY}" \
  --output-dir "${OUTPUT_DIR}" \
  --geometry-reference data/geometry_reference_si_kappa_0p9.npz \
  --seed 20260803 \
  --cache-option 2.0:0.0:256
