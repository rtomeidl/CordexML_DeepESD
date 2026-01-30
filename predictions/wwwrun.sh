#!/bin/bash
#SBATCH --job-name=cnn_preds3
#SBATCH --partition=mp64
#SBATCH --nodes=1
#SBATCH --ntasks=64
#SBATCH --ntasks-per-node=64
#SBATCH --exclusive
#SBATCH --mail-type=ALL
#SBATCH --mail-user=rftome@fc.ul.pt

export TF_ENABLE_ONEDNN_OPTS=0

source /home/rtome/.bash_profile 
conda activate xgb

XPATH=${PWD}
for FILE in `seq -w 001 001 012`; do
  echo "Running ... runfile_${FILE}"

  cd ${XPATH}/config
  cp runfile_${FILE} config.json

  cd ${XPATH}
  python3 main_predictions.py

done
