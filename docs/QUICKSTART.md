# Quick Start Guide

Get up and running with AR Emergence Transformers in 5 minutes.

## 1. Installation

```bash
# Clone the repository
git clone <repository-url>
cd ar-emergence-transformers

# Install dependencies
pip install -r requirements.txt
```

## 2. Download and Prepare Data

### Download from SolARED Portal

Data can be downloaded from the [SolARED Portal](https://sun.njit.edu/sarportal/):

1. Visit https://sun.njit.edu/sarportal/
2. Select Active Regions (at minimum, the 5 test ARs: 11698, 11726, 13165, 13179, 13183)
3. Download timeline data for each AR
4. Convert to the required `.npz` format (see [`DATA_FORMAT.md`](DATA_FORMAT.md))

### Data Structure

Organize your data files in this structure:

```
/path/to/your/data/
├── AR11698/
│   ├── mean_pmdop11698_flat.npz
│   ├── mean_mag11698_flat.npz
│   └── mean_int11698_flat.npz
├── AR11726/
│   ├── mean_pmdop11726_flat.npz
│   ├── mean_mag11726_flat.npz
│   └── mean_int11726_flat.npz
└── ... (other ARs)
```

**Key points:**
- Create a parent directory (e.g., `/path/to/your/data`)
- Inside it, create subdirectories named `AR{number}/` for each Active Region
- Each AR directory needs 3 `.npz` files (see [`DATA_FORMAT.md`](DATA_FORMAT.md) for details)

## 3. Run Evaluation

```bash
cd scripts/eval
python evaluate_all_experiments.py \
  --data_path /path/to/your/data \
  --output_dir ../../results \
  --test_ars 11698 11726 13165 13179 13183
```

Replace `/path/to/your/data` with your actual data directory path.

## 4. View Results

Results are saved in `results/unified_evaluations/`:

- **CSV files**: `csv/all_ARs_emergence_timing_table.csv`
- **PDF plots**: `pdf/AR{AR}_all_models_comparison.pdf`

## Troubleshooting

### "FileNotFoundError: Could not find AR{number} data files"
- Check that your `--data_path` points to the directory containing `AR*/` folders
- Verify each AR directory has all 3 required `.npz` files
- Check file naming: must be exactly `mean_pmdop{AR}_flat.npz`, etc.

### "CUDA out of memory"
- The script uses 2 GPUs by default. If you have only 1 GPU, edit `slurm/eval.sh` to use `--gres=gpu:1`
- Or run on CPU (slower): modify the script to use `device='cpu'`

### "ModuleNotFoundError"
- Make sure you've installed all dependencies: `pip install -r requirements.txt`
- Activate your conda/virtual environment if using one

## Next Steps

- See [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md) for detailed instructions to reproduce paper results
- See [`DATA_FORMAT.md`](DATA_FORMAT.md) for complete data format specifications and download instructions
- See [`comparison/README.md`](../comparison/README.md) for model complexity profiling
