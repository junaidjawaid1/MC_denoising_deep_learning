#!/bin/bash
# Example SLURM job for the gamma evaluation. PyMedPhys gamma runs on the CPU and needs ~20 GB RAM
# (evaluation.ram_available_gb); the GPU is only used for the network prediction.
#SBATCH --job-name=dose-denoising-eval
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --output=slurm-%x-%j.out

# Usage:
#   sbatch slurm/evaluate.sbatch /path/to/dataset_2mm results/2mm_prediction --checkpoint best_model.pth
#   sbatch slurm/evaluate.sbatch /path/to/dataset_2mm results/2mm_baseline --baseline
DATA_ROOT=${1:?dataset directory}
OUTPUT_DIR=${2:?output directory}
shift 2

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
srun python scripts/evaluate.py --data-root "$DATA_ROOT" --output-dir "$OUTPUT_DIR" "$@"
