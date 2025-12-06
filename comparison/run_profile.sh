#!/bin/bash

# SLURM job submission script for model complexity profiling
# NOTE: Uncomment and adjust SLURM directives below for your HPC environment

#SBATCH --job-name=profile_complexity
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=1:00:00
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

# Change to the project directory (adjust path as needed)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT" || exit 1

# Set up environment variables
export PYTHONUNBUFFERED=1
export CUDA_LAUNCH_BLOCKING=0
export TORCH_SHOW_CPP_STACKTRACES=0

echo "========================================"
echo "PROFILING ALL MODELS FOR COMPLEXITY"
echo "========================================"
echo "Models: LSTM, Baseline, Baseline+Conv1D, EarlyDetect, EarlyDetect+Conv1D"
echo "Metrics: Parameters, FLOPs, Peak Memory, Forward Pass Time"
echo "Output: complexity_table.csv"
echo "========================================"

# Install thop if not available
echo "Checking for thop library..."
python -c "import thop" 2>/dev/null || {
    echo "Installing thop library..."
    pip install thop
}

# Run profiling script
echo "Starting complexity profiling..."
python comparison/profile_complexity.py --output complexity_table.csv

echo "========================================"
echo "PROFILING COMPLETED"
echo "Results saved in: comparison/complexity_table.csv"
echo "========================================"

