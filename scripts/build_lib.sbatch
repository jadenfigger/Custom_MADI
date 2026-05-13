#!/bin/bash
#SBATCH --job-name=madi_dense
#SBATCH -p public
#SBATCH -q public
#SBATCH --array=0-63
#SBATCH -N 1
#SBATCH -c 16
#SBATCH -G 1
#SBATCH --mem=64G
#SBATCH -t 1-00:00:00
#SBATCH -o logs/madi_dense_%A_%a.out
#SBATCH -e logs/madi_dense_%A_%a.err
#SBATCH --export=NONE

# Sol's documented activation pattern — do not embellish.
module load mamba/latest
source activate madiEnv

# Sanity check — the first line should be /home/jfigger/.conda/envs/madi/bin/python
which python
python -c "import numpy, scipy, numba, nibabel; from numba import cuda; print('imports OK'); print('CUDA:', cuda.is_available()); print('GPU:', cuda.get_current_device().name.decode() if cuda.is_available() else 'NONE')"

cd "${SLURM_SUBMIT_DIR}"
mkdir -p logs libraries

python /scratch/jfigger/madi/Custom_MADI/scripts/fit_data.py \
    --build-library \
    --lib-preset dense \
    --library libraries/madi_dense.npz \
    --shard-id "${SLURM_ARRAY_TASK_ID}" \
    --n-shards 64