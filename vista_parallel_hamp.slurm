#!/bin/bash
#SBATCH -J parallel-hamp
#SBATCH -o status/parallel_hamp_%j.o
#SBATCH -e status/parallel_hamp_%j.e
#SBATCH -p gh
#SBATCH -N 20
#SBATCH --ntasks-per-node=1
#SBATCH -t 3:00:00
#SBATCH -A ASC25081

module load cuda/12.4

# Navigate to repo root
cd /work/09636/srivibalaji/vista/bb-audit-hamp
mkdir -p status

# Setup environment variables
export PYTHONPATH="${PYTHONPATH}:/work/09636/srivibalaji/vista/bb-audit-hamp/misleading-privacy-evals/src"
export MLFLOW_ALLOW_FILE_STORE=true

# Activate Conda environment
eval "$(conda shell.bash hook)"
conda activate misleading-privacy-evals
export LD_LIBRARY_PATH="/work/09636/srivibalaji/miniconda3/envs/misleading-privacy-evals/lib:$LD_LIBRARY_PATH"

echo "======================================================="
echo "Running HAMP Train & Attack on Our Codebase"
echo "======================================================="

srun --ntasks=$SLURM_NTASKS --nodes=$SLURM_JOB_NUM_NODES bash -c '
  MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1);
  MASTER_PORT=29500;
  torchrun --nnodes=$SLURM_JOB_NUM_NODES --nproc_per_node=1 \
    --rdzv_backend=c10d \
    --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} \
    parallel_audit_model_hamp.py \
    --data_name cifar10 \
    --model_name wideresnet_np \
    --defense_type hamp \
    --target_type mislabeled \
    --num_shadow 64 \
    --num_canaries 500 \
    --n_epochs 200 \
    --batch_size 64 \
    --lr 0.5 \
    --momentum 0.99 \
    --weight_decay 1e-5 \
    --seed 42 \
    --sampling shuffle \
    --hamp_gamma 0.9996837722339832 \
    --hamp_alpha_entropy 0.005 \
    --out debug_audit_hamp_64
'
