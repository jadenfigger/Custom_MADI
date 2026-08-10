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

# This is the single Sol launcher for production, the P0-D golden check, the
# v5 pilot build, and the post-production restricted replicate.  The 24-hour
# production default stays on public/public.  For the sub-four-hour golden and
# pilot jobs, override it at submission with ``--partition=htc --qos=public``
# and request at least 24G host memory.
# Select a mode at submission time, for example:
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

if [[ -z "${SLURM_SUBMIT_DIR:-}" ]]; then
  echo "ERROR: SLURM_SUBMIT_DIR is not set; this launcher must be submitted with sbatch." >&2
  exit 2
fi
cd "${SLURM_SUBMIT_DIR}"
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
    # This pilot is intentionally blocked on the certified SI geometry
    # reference.  It exercises the SI Eq. S8 source-domain and exact
    # full-facet (Eq. S2) per-position KD classifier, so it must not override
    # W or restore the retired candidate cache / tagged-exit calibration
    # options.  The 8x12 log grid leaves eight masked cellular (rho,V) pairs,
    # spanning v_i=0.428..0.966; with kio={0,20,130} plus free water it writes
    # exactly 25 entries across four shards.  Eight independent 512-walker
    # ensembles exercise v5 variance/subset storage only; this is not a
    # production-Monte-Carlo-precision run.  The compact 60-pair timing grid
    # contains every declared v5 diagnostic pair, and the dense preset's
    # unchanged default b grid supplies all 25 stored b-values.
    python -m scripts.fit_data \
      --build-library \
      --lib-preset dense \
      --library libraries/madi_v5_remediation_pilot.npz \
      --shard-id "${TASK_ID}" \
      --n-shards "${N_SHARDS}" \
      --seed 20260803 \
      --vi-min 0.4 \
      --vi-max 0.99 \
      --remediation-n-rho 8 \
      --remediation-n-V 12 \
      --remediation-rho-min 10000 \
      --remediation-rho-max 10000000 \
      --remediation-V-min 0.01 \
      --remediation-V-max 200 \
      --remediation-kios 0 20 130 \
      --sim-walkers 512 \
      --sim-ensembles 8 \
      --geometry-reference data/geometry_reference_si_kappa_0p9.npz \
      --sim-T-max 128 \
      --sim-small-deltas 5 7 10 12 15 20 25 30 \
      --sim-big-deltas 15 25 30 36 40 50 60 80
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
      --vi-max 0.99 \
      --geometry-reference data/geometry_reference_si_kappa_0p9.npz
    ;;

  replicate)
    N_SHARDS="${2:-8}"
    if (( TASK_ID < 0 || TASK_ID >= N_SHARDS )); then
      echo "replicate task ${TASK_ID} is outside [0, ${N_SHARDS})" >&2
      exit 2
    fi
    # W4 is deliberately a post-production, independently seeded restricted
    # build.  Its declared 190-entry subset spans the P0 (rho,V) plane and
    # k_io axis.  Do not submit it until the production artifact exists and a
    # separate approval is given; it compares only against that production
    # artifact and never overwrites it.
    if [[ ! -f libraries/madi_dense_universal_remediated.npz ]]; then
      echo "ERROR: replicate mode is post-production; the v5 production artifact is absent." >&2
      exit 2
    fi
    python -m scripts.fit_data \
      --build-library \
      --lib-preset dense \
      --library libraries/madi_dense_v5_replicate_subset.npz \
      --shard-id "${TASK_ID}" \
      --n-shards "${N_SHARDS}" \
      --seed 20260804 \
      --vi-min 0.4 \
      --vi-max 0.99 \
      --remediation-entry-subset data/madi_v5_replicate_entry_subset.json \
      --geometry-reference data/geometry_reference_si_kappa_0p9.npz
    ;;

  *)
    echo "Pass positional mode golden, pilot, production, or replicate; got ${MODE}." >&2
    exit 2
    ;;
esac
