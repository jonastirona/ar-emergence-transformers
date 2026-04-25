# Reproducibility Guide

This guide explains how to reproduce the results from the paper using this repository.

## Prerequisites

1. **Python Environment**: Python 3.10+ with the required packages (see `requirements.txt`)
2. **Data**: Processed SDO/HMI data in the format described in `DATA_FORMAT.md`
3. **Hardware**: GPU recommended for training and evaluation (CUDA-compatible)

## Quick Start: Reproducing Paper Results

### Step 1: Install Dependencies

```bash
# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install requirements
pip install -r requirements.txt
```

### Step 2: Download and Prepare Data

**Download from SolARED Portal:**
1. Visit https://sun.njit.edu/sarportal/
2. Download data for the 5 test ARs: 11698, 11726, 13165, 13179, 13183
3. Convert to `.npz` format (see [`DATA_FORMAT.md`](DATA_FORMAT.md) for format specifications)

**Data Structure:**
Place your processed data in the expected format:

```bash
# Example structure
mkdir -p data
# Copy your AR data directories to data/
# data/AR11698/, data/AR11726/, etc.
```

Each AR directory must contain 3 `.npz` files:
- `mean_pmdop{AR}_flat.npz` (power maps)
- `mean_mag{AR}_flat.npz` (magnetic flux)
- `mean_int{AR}_flat.npz` (continuum intensity)

### Step 3: Run Evaluation (Using Pre-trained Models)

The repository includes pre-trained checkpoints. To reproduce the paper's evaluation results:

```bash
cd scripts/eval
python evaluate_all_experiments.py \
  --data_path /path/to/data \
  --output_dir ../../results \
  --test_ars 11698 11726 13165 13179 13183
```

This will generate:
- PDF plots for each AR and model
- CSV files with metrics
- Outputs in `results/unified_evaluations/`

### Step 4: Verify Results

Compare your outputs with the expected results structure:

```
results/
├── unified_evaluations/
│   ├── all_ARs_emergence_timing_table.csv
│   ├── AR11698_*.pdf
│   ├── AR11726_*.pdf
│   └── ...
├── metrics.csv
└── ...
```

## Reproducing Training

If you want to retrain the models from scratch:

### Baseline Models

```bash
# Baseline (no Conv1D)
python scripts/train/baseline_train.py \
  --data_path /path/to/data \
  --output_dir results/baseline \
  --max_trials 32 \
  --epochs 1000 \
  --seed 42

# Baseline + Conv1D
python scripts/train/baseline_train.py \
  --data_path /path/to/data \
  --output_dir results/baseline \
  --use_temporal_conv \
  --max_trials 32 \
  --epochs 1000 \
  --seed 42
```

### EarlyDetect Models

```bash
# EarlyDetect (no Conv1D)
python scripts/train/earlydetect_train.py \
  --data_path /path/to/data \
  --output_dir results/earlydetect \
  --max_trials 16 \
  --epochs 1000 \
  --seed 42

# EarlyDetect + Conv1D
python scripts/train/earlydetect_train.py \
  --data_path /path/to/data \
  --output_dir results/earlydetect \
  --use_temporal_conv \
  --max_trials 16 \
  --epochs 1000 \
  --seed 42
```

## Expected Metrics

The evaluation script computes several metrics per model and AR (matching paper Section 2.3):

1. **Overall RMSE**: Root Mean Squared Error for intensity prediction (normalized for Table 1, denormalized for per-AR tables)
2. **Operational Lead Time (T_lead)**: `T_lead = (t_onset_obs − t_onset_pred) + 12h`
   - Positive values = genuine early alert (predicted before observed onset)
   - Negative values = late prediction (missed the window)
   - TP window: T_lead ∈ [0, 24] h; emergence threshold: derivative < −0.01 for k ≥ 4 consecutive hours
3. **Emergence RMSE**: RMSE computed only within the 24-hour emergence window
4. **Model Complexity**: Parameters, FLOPs, Peak Memory, Forward Pass Time (from profiling script)

## Reproducibility Notes

### Training Configuration (from paper Section 2.3)
- **Framework**: PyTorch 2.1.0 with CUDA 12.1
- **Hardware**: NVIDIA A100 GPUs (40 GB memory)
- **Training**: Mixed-precision (FP16), batch size 32, AdamW optimizer, OneCycleLR scheduling
- **Early Stopping**: Patience of 50 epochs
- **Gradient Clipping**: Max norm 1.0
- **Random Seeds**: `seed=42` by default
- **Environment Variables**: `PYTHONHASHSEED=42` and `CUBLAS_WORKSPACE_CONFIG=:4096:8` for reproducibility

### Expected Variations
- **Hardware Differences**: Small numerical differences may occur due to GPU/CPU differences, but results should be qualitatively similar
- **Package Versions**: Use the exact versions in `requirements.txt` for best reproducibility
- **Normalized vs Denormalized**: Paper Table 1 uses normalized RMSE (0.1189), per-AR tables use denormalized (328.27)

## Troubleshooting

### Data Not Found
- Verify data paths match the expected structure
- Check that AR numbers match (11698, 11726, 13165, 13179, 13183)
- Ensure `.npz` files are readable with `np.load()`

### CUDA Out of Memory
- Reduce batch size in training scripts
- Use CPU mode (slower but works): set `device='cpu'` in scripts

### Checkpoint Loading Issues
- Verify checkpoint files exist in `models/checkpoints/`
- Check that model architecture matches checkpoint (Conv1D flag)

## Paper Details

This repository reproduces results from:

**"Forecasting Continuum Intensity for Solar Active Region Emergence Prediction using Transformers"**

**Key Experimental Design:**
- **Factorial Ablation Study**: 2×2 design evaluating:
  - Conv1D front-end: Yes/No
  - Architecture: Baseline vs. Early Detection
- **Result**: 4 Transformer configurations + LSTM baseline = 5 total models
- **Best Model**: EarlyDetect (no Conv1D) - RMSE: 0.1189, highest operational T_lead

**Dataset:**
- **Source**: SolARED (Solar Active Region Emergence Dataset)
- **Size**: 46 ARs (41 train/val, 5 test)
- **Download**: https://sun.njit.edu/sarportal/
- **Reference**: Kasapis et al. (2025)

## Citation

If you use this code or models, please cite:

```bibtex
@article{tirona2025forecasting,
  title={Forecasting Continuum Intensity for Solar Active Region Emergence Prediction using Transformers},
  author={Tirona, Jonas and Patil, Sarang and Kasapis, Spiridon and Dogan, Eren and Stefan, John and Kitiashvili, Irina N. and Kosovichev, Alexander G. and Xu, Mengjia},
  journal={Journal of Geophysical Research: Machine Learning and Computation},
  year={2025}
}
```

**Dataset Citation:**
```bibtex
@article{kasapis2025,
  title={SolARED: Solar Active Region Emergence Dataset},
  author={Kasapis, S. and Kitiashvili, I. N. and Kosovichev, A. G. and Stefan, J. T.},
  journal={ApJS},
  year={2025}
}
```
