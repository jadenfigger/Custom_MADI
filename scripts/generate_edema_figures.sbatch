#!/bin/bash
#SBATCH --job-name=edema_figures
#SBATCH -p htc
#SBATCH -q public
#SBATCH -N 1
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH -t 0-02:00:00
#SBATCH -o logs/edema_figures_%j.out
#SBATCH -e logs/edema_figures_%j.err
#SBATCH --export=NONE

# No GPU / job array here -- run_all.py is a single serial script (ROI/FLAIR
# resampling via FSL, then figures 1/3/4/6 via matplotlib, Agg backend
# already set in each fig*.py). Needs step 1 (run_edema_cohort.sbatch) done
# first: it reads derivatives/madi/sub-*/dwi/method-*/ fit output.

module load mamba/latest
source activate madiEnv

# FSL provides `flirt`/`convert_xfm`, used by roi_space.py / flair_space.py.
# No established Sol module name for this yet -- find it once, then hardcode
# it here:
#   module spider fsl
#   module avail 2>&1 | grep -i fsl
# module load <whatever that reports>
which flirt || { echo "ERROR: flirt not on PATH -- load the FSL module first (see comment above)"; exit 1; }

# Sanity check — the first line should be /home/jfigger/.conda/envs/madi/bin/python
which python
python -c "import numpy, matplotlib, nibabel; print('imports OK')"

# Adjust if the repo/data live somewhere else on Sol.
REPO_ROOT=/scratch/jfigger/custom_madi/Custom_MADI
export MADI_EDEMA_DATA_ROOT=/scratch/jfigger/edema

cd "${REPO_ROOT}"
mkdir -p logs
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH}"

python -m scripts.edema_figures.run_all
