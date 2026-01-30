#!/bin/sh
#SBATCH --job-name=al_emu_pg
#SBATCH --qos=ng
#SBATCH --gpus=1
#SBATCH --mem=256G
#SBATCH --time=6:00:00

#module purge
#conda activate xgbnew

module purge
module load python3/3.12.9-01 cuda

python3 main.py
