#!/bin/bash
#SBATCH -J hamp-train
#SBATCH -o status/hamp_train_%A_%a.o
#SBATCH -e status/hamp_train_%A_%a.e
#SBATCH -p gh
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH -t 3:00:00
#SBATCH -A ASC25081
#SBATCH --array=0-15

SEED=0
NUM_SHADOW=64
NUM_CANARIES=500
CANARY_TYPE="label_noise"
NUM_POISON=0
POISON_TYPE="random_images"
ENTROPY_THRESHOLD="0.9996837722339832"
REGULARIZATION_STRENGTH="0.005"

REPO_DIR="/work/09636/srivibalaji/vista/bb-audit-hamp/misleading-privacy-evals"
EXPERIMENT_DIR="${REPO_DIR}/experiments"
DATA_DIR="${REPO_DIR}/data"
EXPERIMENT="label_noise"
RUN="train_4"

echo "Training shadow models for array task ${SLURM_ARRAY_TASK_ID}"
export PYTHONPATH="${PYTHONPATH}:${REPO_DIR}/src"
export DOWNLOAD_DATA=1

# Use absolute path to python to bypass conda shell activation issues in Slurm
PYTHON_BIN="/work/09636/srivibalaji/miniconda3/envs/bb-audit-hamp/bin/python"

# Calculate start index for this task (each task trains 4 models sequentially)
START_IDX=$((SLURM_ARRAY_TASK_ID * 4))

for i in {0..3}; do
  SHADOW_MODEL_IDX=$((START_IDX + i))
  echo "--------------------------------------------------"
  echo "Starting shadow model ${SHADOW_MODEL_IDX}..."
  echo "--------------------------------------------------"
  
  $PYTHON_BIN -u -m experiments.hamp \
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
done
