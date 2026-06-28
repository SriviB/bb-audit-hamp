#!/bin/bash
#SBATCH -J hamp-attack
#SBATCH -o status/hamp_attack_%j.o
#SBATCH -e status/hamp_attack_%j.e
#SBATCH -p gh
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH -t 1:00:00
#SBATCH -A ASC25081

SEED=42
NUM_SHADOW=64
NUM_CANARIES=500
CANARY_TYPE="label_noise"
NUM_POISON=0
POISON_TYPE="random_images"
LOGREG_C="1.0"

REPO_DIR="/work/09636/srivibalaji/vista/bb-audit-hamp/misleading-privacy-evals"
EXPERIMENT_DIR="${REPO_DIR}/experiments"
DATA_DIR="${REPO_DIR}/data"
EXPERIMENT="label_noise"
RUN="train_4"

echo "Running attack with C=${LOGREG_C}"
export LD_LIBRARY_PATH="/work/09636/srivibalaji/miniconda3/envs/misleading-privacy-evals/lib:$LD_LIBRARY_PATH"
export PYTHONPATH="${PYTHONPATH}:/work/09636/srivibalaji/vista/bb-audit-hamp/misleading-privacy-evals/src"
export MLFLOW_ALLOW_FILE_STORE=true

# Activate Conda environment
eval "$(conda shell.bash hook)"
conda activate misleading-privacy-evals

# Run from misleading-privacy-evals/ to prevent local models/ folder import collision
cd "${REPO_DIR}"

python -u -m experiments.hamp \
  --experiment-dir "${EXPERIMENT_DIR}" \
  --experiment "${EXPERIMENT}" \
  --data-dir "${DATA_DIR}" \
  --seed "${SEED}" --num-shadow "${NUM_SHADOW}" \
  --run-suffix "${RUN}" \
  --num-canaries "${NUM_CANARIES}" --canary-type "${CANARY_TYPE}" \
  --num-poison "${NUM_POISON}" --poison-type "${POISON_TYPE}" \
  attack --logreg-c "${LOGREG_C}"
