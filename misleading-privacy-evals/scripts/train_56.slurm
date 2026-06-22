#!/bin/bash
#SBATCH -J hamp-train-56
#SBATCH -o status/hamp_train_56_%j.o
#SBATCH -e status/hamp_train_56_%j.e
#SBATCH -p gh
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH -t 1:00:00
#SBATCH -A ASC25081

export LD_LIBRARY_PATH="/work/09636/srivibalaji/miniconda3/envs/bb-audit-hamp/lib:$LD_LIBRARY_PATH"
export PYTHONPATH="${PYTHONPATH}:/work/09636/srivibalaji/vista/bb-audit-hamp/misleading-privacy-evals/src"

PYTHON_BIN="/work/09636/srivibalaji/miniconda3/envs/bb-audit-hamp/bin/python"

$PYTHON_BIN -u -m experiments.hamp \
  --experiment-dir "/work/09636/srivibalaji/vista/bb-audit-hamp/misleading-privacy-evals/experiments" \
  --experiment "label_noise" \
  --data-dir "/work/09636/srivibalaji/vista/bb-audit-hamp/misleading-privacy-evals/data" \
  --seed 0 --num-shadow 64 \
  --run-suffix "train_4" \
  --num-canaries 500 --canary-type "label_noise" \
  --num-poison 0 --poison-type "random_images" \
  train --shadow-model-idx 56 \
  --entropy-threshold "0.9996837722339832" \
  --regularization-strength "0.005"
