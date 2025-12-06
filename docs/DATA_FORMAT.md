# Data Format & Download Guide

This document describes the expected data format and how to obtain the data for reproducing the paper results.

## Data Source

The data used in this paper comes from the **SolARED (Solar Active Region Emergence Dataset)**, a machine learning-ready dataset comprising 50 active regions observed with the SDO/HMI instrument. The dataset includes:

- **50 Active Regions** observed between 2010-2023
- **46 ARs used** (4 excluded due to data gaps)
- **41 ARs for training/validation**, **5 ARs for testing** (11698, 11726, 13165, 13179, 13183)
- **10-day tracking period** per AR (before, during, and after emergence)
- **9×9 grid** partitioning (81 tiles per AR, central row analyzed)

### Downloading Data

**Option 1: SolARED Portal (Recommended)**

Data can be downloaded from the SolARED web portal:
- **URL**: https://sun.njit.edu/sarportal/
- **Format**: Download as `.fits` files or use the portal's export functionality
- **Documentation**: See Kasapis et al. (2024, 2025) for detailed dataset description

**Steps:**
1. Visit https://sun.njit.edu/sarportal/
2. Select desired Active Region (AR)
3. Choose tiles and time range
4. Download timeline data in `.fits` format
5. Convert to `.npz` format (see preprocessing below)

**Option 2: Pre-processed Data**

If you have access to pre-processed `.npz` files matching the format below, you can use them directly.

### Data Preprocessing

The SolARED dataset provides 2D spatial maps (512×512 pixels). For this repository, data must be preprocessed into 1D tile-averaged timelines:

1. **Tile Extraction**: Extract central row (9 tiles) from 9×9 grid
2. **Averaging**: Average pixel values within each tile
3. **Format Conversion**: Convert to `.npz` format with specific structure (see below)

**Note**: The preprocessing pipeline that converts SolARED data to the `.npz` format used here is not included in this repository. Users need to either:
- Use pre-processed data in the correct format
- Implement the preprocessing pipeline based on the format specification below

## Data Structure

The repository expects processed SDO/HMI data organized by Active Region (AR) number. The data should be placed in a directory structure as follows:

```
data/
├── AR11698/
│   ├── mean_pmdop11698_flat.npz
│   ├── mean_mag11698_flat.npz
│   └── mean_int11698_flat.npz
├── AR11726/
│   ├── mean_pmdop11726_flat.npz
│   ├── mean_mag11726_flat.npz
│   └── mean_int11726_flat.npz
├── AR13165/
│   ├── mean_pmdop13165_flat.npz
│   ├── mean_mag13165_flat.npz
│   └── mean_int13165_flat.npz
├── AR13179/
│   ├── mean_pmdop13179_flat.npz
│   ├── mean_mag13179_flat.npz
│   └── mean_int13179_flat.npz
└── AR13183/
    ├── mean_pmdop13183_flat.npz
    ├── mean_mag13183_flat.npz
    └── mean_int13183_flat.npz
```

## File Format

Each AR directory contains three `.npz` files:

### 1. Power Maps (`mean_pmdop{AR}_flat.npz`)

Contains Doppler power map data with 5 arrays:
- `arr_0`: Power map for 2-3 mHz band
- `arr_1`: Power map for 3-4 mHz band
- `arr_2`: Power map for 4-5 mHz band
- `arr_3`: Power map for 5-6 mHz band
- `arr_4`: Time array (timestamps)

Shape: `(num_tiles, num_timesteps)` for each power map array

### 2. Magnetic Flux (`mean_mag{AR}_flat.npz`)

Contains magnetic flux data:
- `arr_0`: Magnetic flux values

Shape: `(num_tiles, num_timesteps)`

### 3. Continuum Intensity (`mean_int{AR}_flat.npz`)

Contains continuum intensity data (target variable):
- `arr_0`: Continuum intensity values

Shape: `(num_tiles, num_timesteps)`

## Data Preprocessing

The scripts automatically handle:
- Trimming edge tiles (controlled by `rid_of_top` parameter, typically 1)
- Min-max normalization per AR
- NaN handling (replaced with 0)
- Sequence preparation for transformer models

## Data Details from Paper

Based on the paper methodology:

- **Input Sequence Length**: $L_{in} = 110$ hours
- **Prediction Horizon**: $L_{out} = 12$ hours
- **Features**: 5 total
  - 4 acoustic power maps (frequency bands: 2-3, 3-4, 4-5, 5-6 mHz)
  - 1 line-of-sight magnetic field ($B_{los}$)
- **Normalization**: Min-max normalization per AR: $X' = \frac{X - X_{min}}{X_{max} - X_{min}}$
- **Target Variable**: Continuum intensity ($I_c$)

The data is derived from SDO/HMI observables:
- **Doppler velocity** ($v_D$) → acoustic power maps ($P_a$)
- **Line-of-sight magnetic field** ($B_{los}$)
- **Continuum intensity** ($I_c$)

For full details on geometric correction, tiling, and preprocessing, see:
- Kasapis et al. (2025) - SolARED dataset paper
- The paper methodology section (Section 2.1)

## Test Active Regions

The paper uses 5 held-out test ARs:
- **AR11698** (2013-03-15 to 2013-03-17)
- **AR11726** (2013-04-20 to 2013-04-22)
- **AR13165** (2022-12-12 to 2022-12-14)
- **AR13179** (2022-12-30 to 2023-01-01)
- **AR13183** (2023-01-05 to 2023-01-07)

These ARs should be included in your data directory for full reproducibility.

## Notes

- All arrays should be NumPy arrays
- Time arrays should be compatible with `astropy.time` or `datetime` objects
- Data should be preprocessed to remove edge effects and normalize appropriately
- The number of tiles and timesteps may vary per AR
- The scripts automatically handle per-AR normalization and edge tile trimming
