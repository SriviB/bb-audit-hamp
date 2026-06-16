#!/bin/bash
#SBATCH -J hamp-sanity-train
#SBATCH -o status/hamp_sanity_train_%A_%a.o
#SBATCH -e status/hamp_sanity_train_%A_%a.e
#SBATCH -p gh
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH -t 3:00:00
#SBATCH -A ASC25081
#SBATCH --array=0-3

SEED=0
NUM_SHADOW=4
NUM_CANARIES=500
CANARY_TYPE="clean"
NUM_POISON=0
POISON_TYPE="random_images"
ENTROPY_THRESHOLD="0.95"
REGULARIZATION_STRENGTH="0.001"

SHADOW_MODEL_IDX=$SLURM_ARRAY_TASK_ID

REPO_DIR="/work/09636/srivibalaji/vista/bb-audit-hamp/misleading-privacy-evals"
EXPERIMENT_DIR="${REPO_DIR}/experiments_sanity"
DATA_DIR="${REPO_DIR}/data"
EXPERIMENT="clean"
RUN="train_0"

echo "Training shadow model ${SHADOW_MODEL_IDX} / ${NUM_SHADOW}"

conda activate bb-audit-hamp

export PYTHONPATH="${PYTHONPATH}:${REPO_DIR}/src"
export DOWNLOAD_DATA=1

python -u -m experiments.hamp \
  --experiment-dir "${EXPERIMENT_DIR}" \
  --experiment "${EXPERIMENT}" \
  --data-dir "${DATA_DIR}" \
  --seed "${SEED}" --num-shadow "${NUM_SHADOW}" \
  --run-suffix "${RUN}" \
  --num-canaries "${NUM_CANARIES}" --canary-type "${CANARY_TYPE}" \
  --num-poison "${NUM_POISON}" --poison-type "${POISON_TYPE}" \
  train --shadow-model-idx "${SHADOW_MODEL_IDX}" \
  --entropy-threshold "${ENTROPY_THRESHOLD}" \
  --regularization-strength "${REGULARIZATION_STRENGTH}"
