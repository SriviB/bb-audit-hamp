#!/bin/bash
#SBATCH -J hamp-sanity-attack
#SBATCH -o status/hamp_sanity_attack_%j.o
#SBATCH -e status/hamp_sanity_attack_%j.e
#SBATCH -p gh
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH -t 1:00:00
#SBATCH -A ASC25081

SEED=0
NUM_SHADOW=4
NUM_CANARIES=500
CANARY_TYPE="clean"
NUM_POISON=0
POISON_TYPE="random_images"
LOGREG_C="1.0"

REPO_DIR="/work/09636/srivibalaji/vista/bb-audit-hamp/misleading-privacy-evals"
EXPERIMENT_DIR="${REPO_DIR}/experiments_sanity"
DATA_DIR="${REPO_DIR}/data"
EXPERIMENT="clean"
RUN="train_0"

echo "Running sanity attack with C=${LOGREG_C}"

export LD_LIBRARY_PATH="/work/09636/srivibalaji/miniconda3/envs/bb-audit-hamp/lib:$LD_LIBRARY_PATH"
export PYTHONPATH="${PYTHONPATH}:${REPO_DIR}/src"

PYTHON_BIN="/work/09636/srivibalaji/miniconda3/envs/bb-audit-hamp/bin/python"

$PYTHON_BIN -u -m experiments.hamp \
  --experiment-dir "${EXPERIMENT_DIR}" \
  --experiment "${EXPERIMENT}" \
  --data-dir "${DATA_DIR}" \
  --seed "${SEED}" --num-shadow "${NUM_SHADOW}" \
  --run-suffix "${RUN}" \
  --num-canaries "${NUM_CANARIES}" --canary-type "${CANARY_TYPE}" \
  --num-poison "${NUM_POISON}" --poison-type "${POISON_TYPE}" \
  attack --logreg-c "${LOGREG_C}"
