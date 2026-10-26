#!/bin/bash
# Example SLURM job for training; adapt partition, GPU request and paths to your cluster.
#SBATCH --job-name=dose-denoising-train
#SBATCH --gres=gpu:1                 # A100 80 GB used in the paper
#SBATCH --cpus-per-task=8
#SBATCH --output=slurm-%x-%j.out

# Usage: sbatch slurm/train.sbatch /path/to/dataset_2mm
DATA_ROOT=${1:?dataset directory}

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
srun python scripts/train.py --data-root "$DATA_ROOT" --output-dir runs
