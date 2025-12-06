#!/bin/bash

# SLURM job submission script for Baseline Transformer training
# Supports both Conv1D variants via --use_temporal_conv flag
# NOTE: Uncomment and adjust SLURM directives below for your HPC environment
#
# Usage:
#   sbatch slurm/train_baseline.sh                    # Search both variants
#   sbatch slurm/train_baseline.sh --use_temporal_conv=False  # No Conv1D
#   sbatch slurm/train_baseline.sh --use_temporal_conv=True   # With Conv1D

#SBATCH --job-name=baseline_train
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

# Uncomment and adjust these system-specific directives:
# #SBATCH --partition=gpu
# #SBATCH --account=your_account
# #SBATCH --qos=standard
# #SBATCH --mail-type=ALL
# #SBATCH --mail-user=your_email@example.com

# Activate Python environment (if using a virtual environment)
# Uncomment and adjust the path below for your environment:
# source /path/to/your/venv/bin/activate
# Or for conda:
# source /path/to/your/conda/env/bin/activate

# Load required modules (if needed for your cluster)
# Uncomment and adjust for your cluster:
# module load your_module_name

# Change to the project directory (automatically detects project root)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT" || exit 1

# Set environment variables for reproducibility
export PYTHONHASHSEED=42
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONUNBUFFERED=1

# Optional: Set up wandb (uncomment and customize if using)
# export WANDB_ENTITY="your-entity"
# export WANDB_PROJECT="your-project"
# export WANDB_RUN_GROUP="baseline-training"

echo "========================================"
echo "BASELINE TRANSFORMER TRAINING"
echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Time: $(date)"
echo "Project Root: ${PROJECT_ROOT}"
echo "========================================"

# Run training script
# Pass through any additional arguments (e.g., --use_temporal_conv)
# NOTE: Update --data_path to point to your data directory
python scripts/train/baseline_train.py \
    --data_path "${PROJECT_ROOT}/data" \
    --output_dir "${PROJECT_ROOT}/results/baseline" \
    --lstm_path "${PROJECT_ROOT}/models/checkpoints/lstm_baseline.pth" \
    --max_trials 32 \
    --epochs 1000 \
    --seed 42 \
    "$@"  # Pass through any additional arguments

echo "========================================"
echo "BASELINE TRAINING COMPLETED"
echo "Time: $(date)"
echo "========================================"
