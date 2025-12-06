#!/bin/bash

# SLURM job submission script for evaluating all experiments
# NOTE: Uncomment and adjust SLURM directives below for your HPC environment

#SBATCH --job-name=evaluate_all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:2
#SBATCH --time=2:00:00
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

# Set up environment variables
export PYTHONUNBUFFERED=1

echo "========================================"
echo "EVALUATING ALL EXPERIMENTS"
echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Time: $(date)"
echo "Project Root: ${PROJECT_ROOT}"
echo "========================================"

# Run evaluation script
# Pass through any additional arguments (e.g., --test_ars)
# NOTE: Update --data_path to point to your data directory
python scripts/eval/evaluate_all_experiments.py \
    --data_path "${PROJECT_ROOT}/data" \
    --output_dir "${PROJECT_ROOT}/results" \
    --test_ars 11698 11726 13165 13179 13183 \
    "$@"  # Pass through any additional arguments

echo "========================================"
echo "EVALUATION COMPLETED"
echo "Time: $(date)"
echo "========================================"
