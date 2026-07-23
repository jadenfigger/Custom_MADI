#!/bin/bash
#SBATCH --job-name=edema_madi_fit
#SBATCH -p public
#SBATCH -q public
#SBATCH --array=0-39%10
#SBATCH -N 1
#SBATCH -c 4
#SBATCH -G 1
#SBATCH --mem=32G
#SBATCH -t 0-01:00:00
#SBATCH -o logs/edema_fit_%A_%a.out
#SBATCH -e logs/edema_fit_%A_%a.err
#SBATCH --export=NONE

# One array task = one manifest row (run_id), selected by matching
# SLURM_ARRAY_TASK_ID to the manifest's row order. --array=0-39 must match
# the manifest's row count (see MANIFEST below) minus the header --
# regenerate the manifest first if the subject count ever changes, and keep
# the array range and row count in sync (same rule build_lib.sbatch follows
# for --array vs --n-shards).
#
# %10 throttles to 10 concurrent tasks so this doesn't grab every GPU on the
# partition at once -- raise/lower depending on queue pressure (squeue -u
# $USER / sinfo).

# Sol's documented activation pattern — do not embellish.
module load mamba/latest
source activate madiEnv

# Sanity check — the first line should be /home/jfigger/.conda/envs/madi/bin/python
which python
python -c "import numpy, scipy, numba, nibabel; from numba import cuda; print('imports OK'); print('CUDA:', cuda.is_available()); print('GPU:', cuda.get_current_device().name.decode() if cuda.is_available() else 'NONE')"

# Adjust if the repo lives somewhere else on Sol.
REPO_ROOT=/scratch/jfigger/madi/Custom_MADI
MANIFEST=/scratch/jfigger/edema/derivatives/edema_cohort_manifest.tsv

cd "${REPO_ROOT}"
mkdir -p logs
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH}"

# Manifest row 1 is the header; array task N -> file line N+2.
RUN_ID=$(awk -F'\t' -v n=$((SLURM_ARRAY_TASK_ID + 2)) 'NR==n {print $1}' "${MANIFEST}")
if [ -z "${RUN_ID}" ]; then
    echo "ERROR: no run_id at manifest row $((SLURM_ARRAY_TASK_ID + 2)) (array task ${SLURM_ARRAY_TASK_ID}) in ${MANIFEST}"
    exit 1
fi
echo "Array task ${SLURM_ARRAY_TASK_ID} -> run_id ${RUN_ID}"

python scripts/run_cohort_manifest.py \
    --manifest "${MANIFEST}" \
    --run-id "${RUN_ID}"
