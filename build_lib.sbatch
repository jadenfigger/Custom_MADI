#!/bin/bash
#SBATCH --job-name=madi_dense
#SBATCH --partition=general
#SBATCH --qos=public
#
# Array: 32 shards.  Each task builds ~3–4 (ρ,V) groups of the dense
# preset and writes its own .npz.  Sharding is by (ρ,V) pair, round-
# robin on ρ·V, so cheap/expensive groups are mixed across tasks.
#
#SBATCH --array=0-31
#
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16          # scipy Voronoi is single-threaded, but
                                    # give headroom for BLAS + ensemble build
#SBATCH --gres=gpu:a100:1           # 1 A100 per task — run_simulation is
                                    # single-GPU, so 4/node wastes allocation
#SBATCH --mem=64G
#
#SBATCH --time=1-00:00:00           # 24h walltime; bump if a task times out
#
#SBATCH --output=logs/madi_dense_%A_%a.out
#SBATCH --error=logs/madi_dense_%A_%a.err
#SBATCH --mail-type=END,FAIL        # optional
# #SBATCH --mail-user=you@asu.edu

set -euo pipefail

# ---------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------
module purge
module load mamba/latest            # or whatever you use on Sol
module load cuda-12.4.1-gcc-12.1.0  # match numba-cuda expectations

# Activate your env — adjust the name to match yours
source activate madi

# Sanity: confirm CUDA visible
nvidia-smi -L
python -c "from numba import cuda; print('CUDA:', cuda.is_available()); print(cuda.get_current_device().name)"

# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------
# Assumes you git-cloned the repo to $HOME/madi and run sbatch from
# $HOME/madi.  Adjust if you put it elsewhere.
REPO_ROOT="${SLURM_SUBMIT_DIR}"
cd "${REPO_ROOT}"

mkdir -p logs libraries

N_SHARDS=32
SHARD_ID=${SLURM_ARRAY_TASK_ID}
OUT_LIB="libraries/madi_dense.npz"      # fit_data.py will suffix .shard###.npz

echo "============================================================"
echo " MADI dense library — shard ${SHARD_ID}/${N_SHARDS}"
echo " host    : $(hostname)"
echo " started : $(date)"
echo " repo    : ${REPO_ROOT}"
echo "============================================================"

# ---------------------------------------------------------------------
# Run the shard
# ---------------------------------------------------------------------
python fit_data.py \
    --build-library \
    --lib-preset dense \
    --library "${OUT_LIB}" \
    --shard-id "${SHARD_ID}" \
    --n-shards "${N_SHARDS}"

echo "============================================================"
echo " shard ${SHARD_ID} finished at $(date)"
echo "============================================================"
