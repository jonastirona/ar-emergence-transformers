# Model Complexity Profiling

This directory contains a single script to profile all models and generate the complexity metrics table used in the paper.

## Overview

The profiling script measures computational complexity for all 5 models:
- **LSTM**
- **Baseline** (no Conv1D)
- **Baseline+Conv1D**
- **EarlyDetect** (no Conv1D)
- **EarlyDetect+Conv1D**

Metrics measured:
- **Parameters**: Total model parameters
- **FLOPs**: Floating Point Operations (in Giga)
- **Peak Memory**: Maximum GPU memory usage (MB)
- **Forward Pass Time**: Inference time (ms)

These metrics are used to generate Table 5 (tab:complexity_results) in the paper, which compares computational efficiency across all models.

## Usage

### Run Profiling

```bash
# Submit SLURM job
sbatch comparison/run_profile.sh

# Or run directly (if on GPU node)
python comparison/profile_complexity.py --output complexity_table.csv
```

### Output

The script generates `complexity_table.csv` with columns:
- Model
- Parameters (K/M format)
- FLOPs (G)
- Peak Memory (MB)
- Fwd. Pass Time (ms)

## Model Configurations

The script uses the optimal model configurations from the paper (Table 2):

- **LSTM**: 3 layers, 64 hidden units, 110 input timesteps
- **Baseline**: 3 layers, 128 d_model, 4 heads, no Conv1D
- **Baseline+Conv1D**: 5 layers, 256 d_model, 4 heads, with Conv1D
- **EarlyDetect**: 6 layers, 256 d_model, 8 heads, no Conv1D
- **EarlyDetect+Conv1D**: 4 layers, 512 d_model, 4 heads, with Conv1D

These match the best-performing configurations identified through hyperparameter search (see paper Appendix A).

## Requirements

- Python 3.7+
- PyTorch with CUDA support
- thop library for FLOPs/MACs measurement
- pandas for table generation

## Notes

- Profiling uses dummy input data (batch_size=32, seq_len=128)
- Warmup iterations (5) and measurement iterations (10) for accurate timing
- Results include median values for robust statistics
- Memory measurements include peak usage tracking
- SLURM script is generic and can be adapted to any HPC environment
