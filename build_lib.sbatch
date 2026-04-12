#!/bin/bash
#SBATCH -p public
#SBATCH -q public
#SBATCH -G a100:1
#SBATCH -c 8
#SBATCH --mem=128G
#SBATCH -t 0-12:00
#SBATCH -J madi_lib
#SBATCH -o madi_lib.%j.out
#SBATCH -e madi_lib.%j.err

module load mamba/latest
source activate madi

cd /scratch/jfigger/madi/custom_madi

python -u fit_data.py --build-library --lib-preset calibration \
                      --library madi_output_calibration.npz