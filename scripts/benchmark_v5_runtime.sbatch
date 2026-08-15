#!/bin/bash
#SBATCH --job-name=madi_v5_runtime
#SBATCH -p public
#SBATCH -q public
#SBATCH --array=0-3%2
#SBATCH -N 1
#SBATCH -c 2
#SBATCH -G a100:1
#SBATCH --mem=24G
#SBATCH -t 0-08:00:00
#SBATCH -o logs/madi_v5_runtime_%A_%a.out
#SBATCH -e logs/madi_v5_runtime_%A_%a.err
#SBATCH --export=NONE

# Diagnostic-only v5 runtime benchmark.  Each task either measures C1/C3 at
# the stencil-probe centre or one exact full-production-walker endpoint replay.
# It writes JSON timing evidence only; it never writes a library artifact.
#
# This follows docs/sol_package_guide.md exactly: clean environment export,
# Mamba module, and source activation.  Do not add set -e, set -u, or pipefail:
# Sol's activation scripts are incompatible with those options.

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
  echo "ERROR: SLURM_SUBMIT_DIR is not set; this launcher must be submitted with sbatch." >&2
  exit 2
fi
cd "${SLURM_SUBMIT_DIR}"
mkdir -p logs sol_outputs

case "${SLURM_ARRAY_TASK_ID}" in
  0) BENCHMARK_CASE="center-c1" ;;
  1) BENCHMARK_CASE="center-c3" ;;
  2) BENCHMARK_CASE="rho-low" ;;
  3) BENCHMARK_CASE="rho-high" ;;
  *)
    echo "ERROR: runtime benchmark task ${SLURM_ARRAY_TASK_ID} is outside [0, 3]." >&2
    exit 2
    ;;
esac

OUTPUT_DIR="sol_outputs/v5_runtime_benchmark_${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID}}"
mkdir -p "${OUTPUT_DIR}"
python analysis/v5_runtime_benchmark.py \
  --case "${BENCHMARK_CASE}" \
  --output-dir "${OUTPUT_DIR}" \
  --geometry-reference data/geometry_reference_si_kappa_0p9.npz \
  --seed 20260803 \
  --profile-walkers 100000
