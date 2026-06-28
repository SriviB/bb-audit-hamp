#!/bin/bash
#SBATCH -J filter-attack
#SBATCH -o status/filter_attack_%j.o
#SBATCH -e status/filter_attack_%j.e
#SBATCH -p gh
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH -t 1:00:00
#SBATCH -A ASC25081

SEED=0
NUM_SHADOW=64
NUM_CANARIES=500
CANARY_TYPE="label_noise"
NUM_POISON=0
POISON_TYPE="random_images"
LOGREG_C="1.0"

ROOT_DIR="/work/09636/srivibalaji/vista/bb-audit-hamp"
REPO_DIR="${ROOT_DIR}/misleading-privacy-evals"
EXPERIMENT_DIR="${REPO_DIR}/experiments"
DATA_DIR="${REPO_DIR}/data"
EXPERIMENT="filter_noise"
RUN="filter_train_1"

echo "Running attack with C=${LOGREG_C}"
export LD_LIBRARY_PATH="/work/09636/srivibalaji/miniconda3/envs/bb-audit-hamp/lib:$LD_LIBRARY_PATH"
# Include ROOT_DIR so we can import utils.dpsgd
export PYTHONPATH="${PYTHONPATH}:${ROOT_DIR}:${REPO_DIR}/src"

# Use absolute path to python to bypass conda shell activation issues in Slurm
PYTHON_BIN="/work/09636/srivibalaji/miniconda3/envs/bb-audit-hamp/bin/python"

$PYTHON_BIN -u -m experiments.filter \
  --experiment-dir "${EXPERIMENT_DIR}" \
  --experiment "${EXPERIMENT}" \
  --data-dir "${DATA_DIR}" \
  --seed "${SEED}" --num-shadow "${NUM_SHADOW}" \
  --run-suffix "${RUN}" \
  --num-canaries "${NUM_CANARIES}" --canary-type "${CANARY_TYPE}" \
  --num-poison "${NUM_POISON}" --poison-type "${POISON_TYPE}" \
  attack --logreg-c "${LOGREG_C}"
