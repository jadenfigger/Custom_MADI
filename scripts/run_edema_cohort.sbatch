#!/bin/bash
#SBATCH --job-name=edema_madi_fit
#SBATCH -p htc
#SBATCH -q public
#SBATCH --array=0-10%4
#SBATCH -N 1
#SBATCH -c 4
#SBATCH -G 1
#SBATCH --mem=32G
#SBATCH -t 0-04:00:00
#SBATCH -o logs/edema_fit_%A_%a.out
#SBATCH -e logs/edema_fit_%A_%a.err
#SBATCH --export=NONE

# One array task = a contiguous block of ROWS_PER_SHARD manifest rows.
# Shard N covers data rows [N*ROWS_PER_SHARD, (N+1)*ROWS_PER_SHARD).
#
# --array upper bound must be ceil(n_data_rows / ROWS_PER_SHARD) - 1.
# With 41 data rows and 4 per shard: ceil(41/4) - 1 = 10  ->  --array=0-10
# Rather than editing this by hand, submit via submit_edema_fit.sh (below),
# which computes the bound from the manifest. Shards that land past the end
# of the manifest exit 0 immediately, so an over-long range is harmless.
#
# %4 throttles concurrent tasks so this doesn't grab every GPU on the
# partition -- raise/lower depending on queue pressure (squeue -u $USER / sinfo).
#
# Rows whose output already exists are skipped, so this is safe to resubmit
# over the full range after a partial run.

set -uo pipefail

# Sol's documented activation pattern — do not embellish.
module load mamba/latest
source activate madiEnv

# Sanity check — the first line should be /home/jfigger/.conda/envs/madi/bin/python
which python
python -c "import numpy, scipy, numba, nibabel; from numba import cuda; print('imports OK'); print('CUDA:', cuda.is_available()); print('GPU:', cuda.get_current_device().name.decode() if cuda.is_available() else 'NONE')"

# Adjust if the repo lives somewhere else on Sol.
REPO_ROOT=/scratch/jfigger/custom_madi/Custom_MADI
MANIFEST=/scratch/jfigger/edema/derivatives/edema_cohort_manifest.tsv

# <<< SET THIS: parent dir that run_cohort_manifest.py writes per-run output into.
# Used only for the "already done?" check.
OUT_ROOT=/scratch/jfigger/edema/derivatives/madi

ROWS_PER_SHARD=4

cd "${REPO_ROOT}"
mkdir -p logs
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

# Data rows only (skip header, ignore trailing blanks).
N_ROWS=$(awk -F'\t' 'NR>1 && NF' "${MANIFEST}" | wc -l)

START=$(( SLURM_ARRAY_TASK_ID * ROWS_PER_SHARD ))
END=$(( START + ROWS_PER_SHARD ))
(( END > N_ROWS )) && END=${N_ROWS}

if (( START >= N_ROWS )); then
    echo "Shard ${SLURM_ARRAY_TASK_ID}: start row ${START} >= ${N_ROWS} data rows — nothing to do."
    exit 0
fi

echo "Shard ${SLURM_ARRAY_TASK_ID}: data rows ${START}..$((END - 1)) of ${N_ROWS}"

shard_status=0
for (( i=START; i<END; i++ )); do
    # Manifest row 1 is the header; data row i -> file line i+2.
    RUN_ID=$(awk -F'\t' -v n=$((i + 2)) 'NR==n {print $1}' "${MANIFEST}")
    if [ -z "${RUN_ID}" ]; then
        echo "ERROR: no run_id at manifest line $((i + 2)) (data row ${i}) in ${MANIFEST}"
        shard_status=1
        continue
    fi

    # Skip rows already processed by an earlier submission.
    # Adjust the glob to match whatever run_cohort_manifest.py actually emits.
    if compgen -G "${OUT_ROOT}/${RUN_ID}/*madi*.nii*" > /dev/null 2>&1; then
        echo "[$(date +%T)] row ${i} (${RUN_ID}): SKIP — output already present"
        continue
    fi

    echo "[$(date +%T)] row ${i} -> run_id ${RUN_ID}"
    python scripts/run_cohort_manifest.py \
        --manifest "${MANIFEST}" \
        --run-id "${RUN_ID}"
    rc=$?
    if (( rc != 0 )); then
        echo "[$(date +%T)] row ${i} (${RUN_ID}): FAILED with exit ${rc}" >&2
        shard_status=1
    fi
done

echo "Shard ${SLURM_ARRAY_TASK_ID} done (status ${shard_status})"
exit ${shard_status}