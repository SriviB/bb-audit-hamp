#!/bin/bash
#SBATCH -J download-cifar10
#SBATCH -o status/download_cifar10_%j.o
#SBATCH -e status/download_cifar10_%j.e
#SBATCH -p gh
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH -t 1:00:00
#SBATCH -A ASC25081

cd /work/09636/srivibalaji/vista/bb-audit-hamp/misleading-privacy-evals
eval "$(conda shell.bash hook)"
conda activate bb-audit-hamp
export LD_LIBRARY_PATH="/work/09636/srivibalaji/miniconda3/envs/bb-audit-hamp/lib:$LD_LIBRARY_PATH"

echo "Starting CIFAR-10 download..."
python -c "from torchvision.datasets import CIFAR10; CIFAR10(root='data', train=True, download=True); CIFAR10(root='data', train=False, download=True)"
echo "Download completed successfully!"
