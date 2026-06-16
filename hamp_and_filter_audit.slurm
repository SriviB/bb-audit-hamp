#!/bin/bash
#SBATCH -J hamp_and_filter_audit
#SBATCH -o status/hamp_and_filter_audit_%A_%a.o
#SBATCH -e status/hamp_and_filter_audit_%A_%a.e
#SBATCH -p gh
#SBATCH -N 5
#SBATCH --ntasks-per-node=1
#SBATCH -t 5:00:00
#SBATCH -A ASC25081
#SBATCH --array=0-2

module load cuda/12.4

set -e
cd $WORK
eval "$(conda shell.bash hook)"
conda activate misleading-privacy-evals
cd bb-audit-dpsgd

mkdir -p status

DATA_NAME="mnist"
MODEL_NAME="cnn"
N_EPOCHS=100
BATCH_SIZE=4000
BLOCK_SIZE=4000
NUM_SHADOW=64
NUM_CANARIES=500
SEED=0
PARENT_OUT="hamp_and_filter_audit"

mkdir -p ${PARENT_OUT}

# SLURM array mapping:
# 0: no defense
# 1: HAMP defense
# 2: Filter defense
experiments=(
    "none|mnist_audit_no_defense"
    "hamp|mnist_audit_hamp_defense"
    "filter|mnist_audit_filter_defense"
)

IFS='|' read -r DEFENSE_TYPE OUT_NAME <<< "${experiments[$SLURM_ARRAY_TASK_ID]}"

# Select learning rates and arguments based on the defense
if [ "${DEFENSE_TYPE}" = "filter" ]; then
    # Use larger learning rate and standard DP-SGD arguments for gradient-norm filtering
    LR="3.0"
    EXTRA_ARGS="--epsilon 10.0 --max_grad_norm 1.0 --defense_k 5 --defense_score_fn grad_norm_unclipped --defense_score_norm linf"
elif [ "${DEFENSE_TYPE}" = "hamp" ]; then
    # HAMP default learning rate and entropy regularization arguments (matched to paper's sweep)
    LR="0.05"
    EXTRA_ARGS="--hamp_gamma 0.9997 --hamp_alpha_entropy 0.005"
else
    # Baseline with no defense
    LR="0.05"
    EXTRA_ARGS=""
fi

echo "=========================================="
echo "Job Array Task ID : ${SLURM_ARRAY_TASK_ID}"
echo "Defense Type      : ${DEFENSE_TYPE}"
echo "Learning Rate     : ${LR}"
echo "Output Directory  : ${OUT_NAME}"
echo "=========================================="

srun --ntasks=$SLURM_NTASKS --nodes=$SLURM_JOB_NUM_NODES bash -c '
  MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1);
  MASTER_PORT=$((29600 + '"${SLURM_ARRAY_TASK_ID}"'));
  torchrun --nnodes=$SLURM_JOB_NUM_NODES --nproc_per_node=1 \
    --rdzv_backend=c10d \
    --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} \
    parallel_audit_model_hamp.py \
    --data_name '"${DATA_NAME}"' \
    --model_name '"${MODEL_NAME}"' \
    --num_shadow '"${NUM_SHADOW}"' \
    --num_canaries '"${NUM_CANARIES}"' \
    --n_epochs '"${N_EPOCHS}"' \
    --lr '"${LR}"' \
    --batch_size '"${BATCH_SIZE}"' \
    --block_size '"${BLOCK_SIZE}"' \
    --seed '"${SEED}"' \
    --target_type mislabeled \
    --defense_type '"${DEFENSE_TYPE}"' \
    '"${EXTRA_ARGS}"' \
    --out '"${PARENT_OUT}/${OUT_NAME}"''

echo "Task ${SLURM_ARRAY_TASK_ID} completed: ${OUT_NAME}"
