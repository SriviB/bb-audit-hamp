#!/bin/bash
#SBATCH -J parallel-filter
#SBATCH -o status/parallel_filter_%j.o
#SBATCH -e status/parallel_filter_%j.e
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
echo "Running Filter Train & Attack on Our Codebase"
echo "======================================================="

srun --ntasks=$SLURM_NTASKS --nodes=$SLURM_JOB_NUM_NODES bash -c '
  MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1);
  MASTER_PORT=29500;
  torchrun --nnodes=$SLURM_JOB_NUM_NODES --nproc_per_node=1 \
    --rdzv_backend=c10d \
    --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} \
    parallel_audit_model_hamp.py \
    --data_name cifar10 \
    --model_name wideresnet \
    --defense_type filter \
    --target_type mislabeled \
    --num_shadow 64 \
    --num_canaries 500 \
    --n_epochs 200 \
    --batch_size 64 \
    --lr 0.5 \
    --momentum 0.99 \
    --seed 42 \
    --sampling shuffle \
    --max_grad_norm -1 \
    --defense_k 5 \
    --defense_score_fn grad_norm_unclipped \
    --defense_score_norm linf \
    --out debug_audit_filter_64
'
