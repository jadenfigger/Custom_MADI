#!/bin/bash
#SBATCH --job-name=madi_dense
#SBATCH -p public
#SBATCH -q public
#SBATCH --array=0-127
#SBATCH -N 1
#SBATCH -c 16
#SBATCH -G 1
#SBATCH --mem=64G
#SBATCH -t 1-00:00:00
#SBATCH -o logs/madi_dense_%A_%a.out
#SBATCH -e logs/madi_dense_%A_%a.err
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=jfigger@asu.edu
#SBATCH --export=NONE

# This is the single Sol launcher for production, the P0-D golden check and
# the pilot build.  Select a mode at submission time, for example:
#   sbatch --array=0-0 scripts/build_lib.sbatch golden
#
# ``#SBATCH --export=NONE`` is mandatory on Sol.  Mode and shard-count are
# positional arguments, rather than inherited environment variables, so a
# submitting shell can never leak a half-activated Mamba state into the job.
# The command is intentionally non-interactive: the Python process exits
# nonzero on a failed golden check or an invalid library build.
# Do not add ``set -e``, ``set -o pipefail``, or ``set -u`` here: Sol's Mamba
# activation scripts touch unset variables and the cluster guide explicitly
# identifies those shell options as an activation failure mode.

# Sol's documented activation pattern — do not embellish.
module load mamba/latest
source activate madiEnv

PYTHON_BIN="$(command -v python)"
echo "${PYTHON_BIN}"
if [[ "${PYTHON_BIN}" != */madiEnv/bin/python ]]; then
  echo "ERROR: source activate madiEnv did not select madiEnv's Python; aborting." >&2
  exit 2
fi
python -c "import numpy, scipy, numba, nibabel; from numba import cuda; print('imports OK'); print('CUDA:', cuda.is_available()); print('GPU:', cuda.get_current_device().name.decode() if cuda.is_available() else 'NONE')"

REPO_ROOT="${SLURM_SUBMIT_DIR:-/scratch/jfigger/madi/Custom_MADI}"
cd "${REPO_ROOT}"
# Module-style invocation below places this repository root on Python's import
# path.  This avoids inheriting or exporting PYTHONPATH while retaining the
# clean ``--export=NONE`` Sol environment.
mkdir -p logs libraries sol_outputs

MODE="${1:-unset}"
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"

case "${MODE}" in
  golden)
    if [[ "${TASK_ID}" != "0" ]]; then
      echo "golden mode is single-task; submit with --array=0-0" >&2
      exit 2
    fi
    OUTPUT_DIR="sol_outputs/gpu_golden_${SLURM_JOB_ID}"
    python -m scripts.run_gpu_golden_check \
      --golden tests/physics_audit/data/cpu_gpu_golden_v1.npz \
      --output-dir "${OUTPUT_DIR}"
    ;;

  pilot)
    N_SHARDS="${2:-4}"
    if (( TASK_ID < 0 || TASK_ID >= N_SHARDS )); then
      echo "pilot task ${TASK_ID} is outside [0, ${N_SHARDS})" >&2
      exit 2
    fi
    # 4 log-rho × 5 log-V nodes with the 0.4<=vi<=0.99 diagonal mask give
    # four cellular geometry pairs. Five k_io nodes yield 20 cellular rows,
    # plus the one free-water atom in shard 0. Timings cover both published
    # finite-lobe cases while keeping a pilot compact.
    python -m scripts.fit_data \
      --build-library \
      --lib-preset dense \
      --library libraries/madi_remediation_pilot.npz \
      --shard-id "${TASK_ID}" \
      --n-shards "${N_SHARDS}" \
      --seed 20260803 \
      --vi-min 0.4 \
      --vi-max 0.99 \
      --remediation-n-rho 4 \
      --remediation-n-V 5 \
      --remediation-rho-min 100000 \
      --remediation-rho-max 2000000 \
      --remediation-V-min 0.205 \
      --remediation-V-max 12 \
      --remediation-kios 0 5 20 60 130 \
      --sim-walkers 2048 \
      --sim-ensembles 1 \
      --sim-L 80 \
      --sim-grid-spacing 0.75 \
      --sim-T-max 80 \
      --sim-small-deltas 7 20 \
      --sim-big-deltas 25 50 \
      --sim-b-values 0 500 1000 2000 4000 6000 \
      --sim-exchange-calibration-walkers 2048 \
      --sim-exchange-calibration-ms 32
    ;;

  production)
    N_SHARDS="${2:-128}"
    if (( TASK_ID < 0 || TASK_ID >= N_SHARDS )); then
      echo "production task ${TASK_ID} is outside [0, ${N_SHARDS})" >&2
      exit 2
    fi
    # Deliberately not invoked until the P0 acceptance suite and the pilot
    # verification checklist pass. It always writes a new artifact and never
    # overwrites the current production library.
    python -m scripts.fit_data \
      --build-library \
      --lib-preset dense \
      --library libraries/madi_dense_universal_remediated.npz \
      --shard-id "${TASK_ID}" \
      --n-shards "${N_SHARDS}" \
      --seed 20260803 \
      --vi-min 0.4 \
      --vi-max 0.99
    ;;

  *)
    echo "Pass positional mode golden, pilot, or production; got ${MODE}." >&2
    exit 2
    ;;
esac
