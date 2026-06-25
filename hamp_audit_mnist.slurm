#!/bin/bash
#SBATCH -J hamp_audit_mnist
#SBATCH -o status/hamp_audit_mnist_%j.o
#SBATCH -e status/hamp_audit_mnist_%j.e
#SBATCH -p gh
#SBATCH -N 5
#SBATCH --ntasks-per-node=1
#SBATCH -t 2:00:00
#SBATCH -A ASC25081

module load cuda/12.4

cd /work/09636/srivibalaji/vista/bb-audit-hamp
eval "$(conda shell.bash hook)"
conda activate bb-audit-hamp
export LD_LIBRARY_PATH="/work/09636/srivibalaji/miniconda3/envs/bb-audit-hamp/lib:$LD_LIBRARY_PATH"

mkdir -p status

DATA_NAME="mnist"
MODEL_NAME="cnn"
N_EPOCHS=100
BATCH_SIZE=4000
BLOCK_SIZE=4000
NUM_SHADOW=64
NUM_CANARIES=500
SEED=0
DEFENSE_TYPE="hamp"
PARENT_OUT="hamp_audit_mnist"
OUT_NAME="mnist_audit_hamp_defense"

mkdir -p ${PARENT_OUT}

# HAMP default learning rate and entropy regularization arguments (matched to paper's sweep)
LR="0.05"
EXTRA_ARGS="--hamp_gamma 0.9997 --hamp_alpha_entropy 0.005"

echo "=========================================="
echo "Defense Type      : ${DEFENSE_TYPE}"
echo "Learning Rate     : ${LR}"
echo "Batch / Block Size: ${BATCH_SIZE}"
echo "Output Directory  : ${OUT_NAME}"
echo "=========================================="

srun --ntasks=$SLURM_NTASKS --nodes=$SLURM_JOB_NUM_NODES bash -c '
  MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1);
  MASTER_PORT=29600;
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

echo "HAMP audit completed: ${OUT_NAME}"
