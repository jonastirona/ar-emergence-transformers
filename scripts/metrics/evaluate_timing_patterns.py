#!/usr/bin/env python3
"""
Script to calculate timing pattern statistics for all models.

This script evaluates all models (LSTM, Baseline, Baseline+Conv1D, EarlyDetect, EarlyDetect+Conv1D)
on all test ARs and calculates timing pattern statistics:
- Mean Timing Δ (hrs)
- Median Timing Δ (hrs)
- Std. Dev. of Timing Δ
- % Early Forecasts (< 0h)
- % Late Forecasts (> 0h)

This addresses the reviewer's request for a "prediction forecasting pattern" table
to support claims about model consistency and operational usefulness.
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import json
from pathlib import Path
import argparse
from datetime import datetime
import time
import os
import re
from collections import OrderedDict
import sys
from pathlib import Path

# Define ROOT path (3 levels up from scripts/metrics/)
ROOT = Path(__file__).resolve().parents[2]

# Add paths for imports
sys.path.extend([
    str(ROOT / "scripts"),
    str(ROOT / "scripts" / "models"),
    str(ROOT / "scripts" / "data"),
])

from functions_spyros import (
    LSTM, lstm_ready, min_max_scaling, smooth_with_numpy, 
    emergence_indication, recalibrate, calculate_metrics
)

# Import transformer models
try:
    from baseline_transformer import SARTransformerLocalTile as SARTransformerLocalTile_B
    print("✅ Successfully imported Baseline transformer model")
except ImportError as e:
    print(f"⚠️  Warning: Could not import baseline_transformer: {e}")
    print("   Baseline models will not be loaded")
    SARTransformerLocalTile_B = None

from early_detect_transformer import SARTransformerLocalTile
from data_loader import load_ar_data_enhanced, cross_ar_tile_data_preparation_attention


def safe_gradient(arr, min_length=2):
    """Safely compute gradient, handling short arrays"""
    arr = np.atleast_1d(arr).flatten()
    if len(arr) < min_length:
        return np.zeros_like(arr)
    return np.gradient(arr)


def find_first_emergence_window(signal, threshold=-0.01, min_duration=4):
    """Find the first 24-hour emergence window"""
    emergence_indices = emergence_indication(signal, threshold, min_duration)
    
    first_emergence_start = None
    for i, val in enumerate(emergence_indices):
        if val != 0:
            first_emergence_start = i
            break
    
    if first_emergence_start is None:
        return None, None
    
    window_size = 24
    emergence_end = min(first_emergence_start + window_size, len(signal))
    
    return first_emergence_start, emergence_end


def get_ar_settings_fixed(test_AR, rid_of_top):
    """Get AR-specific settings"""
    if test_AR == 11698:
        starting_tile = 46 - rid_of_top * 9
        before_plot = 50
        num_in = 96
        NOAA_first = datetime(2013, 3, 15)
        NOAA_second = datetime(2013, 3, 17)
    elif test_AR == 11726:
        starting_tile = 37 - rid_of_top * 9
        before_plot = 50
        num_in = 72
        NOAA_first = datetime(2013, 4, 20)
        NOAA_second = datetime(2013, 4, 22)
    elif test_AR == 13165:
        rid_of_top = 1
        starting_tile = 28 - rid_of_top * 9
        before_plot = 40
        num_in = 96
        NOAA_first = datetime(2022, 12, 12)
        NOAA_second = datetime(2022, 12, 14)
    elif test_AR == 13179:
        starting_tile = 37 - rid_of_top * 9
        before_plot = 40
        num_in = 96
        NOAA_first = datetime(2022, 12, 30)
        NOAA_second = datetime(2023, 1, 1)
    elif test_AR == 13183:
        starting_tile = 37 - rid_of_top * 9
        before_plot = 40
        num_in = 96
        NOAA_first = datetime(2023, 1, 6)
        NOAA_second = datetime(2023, 1, 8)
    else:
        raise ValueError("Invalid test_AR value")
    return starting_tile, before_plot, num_in, NOAA_first, NOAA_second


def load_and_preprocess_ar_eval_template(test_AR, data_path, rid_of_top, size):
    """Load and preprocess AR data with per-AR normalization"""
    base = f'{data_path}/AR{test_AR}'
    power = np.load(os.path.join(base, f'mean_pmdop{test_AR}_flat.npz'), allow_pickle=True)
    mag   = np.load(os.path.join(base, f'mean_mag{test_AR}_flat.npz'),   allow_pickle=True)
    cont  = np.load(os.path.join(base, f'mean_int{test_AR}_flat.npz'),   allow_pickle=True)

    pm23, pm34, pm45, pm56, time_arr = (
        power['arr_0'], power['arr_1'], power['arr_2'], power['arr_3'], power['arr_4']
    )
    mf = mag['arr_0']; ii = cont['arr_0']

    # Store raw data BEFORE trimming
    ii_raw_full = ii.copy()
    
    sl = slice(rid_of_top*size, -rid_of_top*size)
    pm23, pm34, pm45, pm56 = pm23[sl,:], pm34[sl,:], pm45[sl,:], pm56[sl,:]
    mf = mf[sl,:]; ii = ii[sl,:]
    mf[np.isnan(mf)] = 0; ii[np.isnan(ii)] = 0

    stacked = np.stack([pm23,pm34,pm45,pm56],axis=1)
    mp,Mp = stacked.min(), stacked.max()
    mm,Mm = mf.min(), mf.max()
    mi,Mi = ii.min(), ii.max()
    stacked = (stacked - mp)/(Mp-mp)
    mf = (mf - mm)/(Mm-mm)
    ii = (ii - mi)/(Mi-mi)

    inputs = np.concatenate([stacked, np.expand_dims(mf,1)], axis=1)
    
    return inputs, ii, time_arr, (mp, Mp, mm, Mm, mi, Mi), ii_raw_full


def lstm_ready_eval_template(tile, size, power_maps, intensities, num_in, num_pred, model_seq_len=None):
    """LSTM ready function"""
    final_maps = np.transpose(power_maps, axes=(2, 1, 0))
    final_ints = np.transpose(intensities, axes=(1,0))
    X_trans = final_maps[:,:,tile]
    y_trans = final_ints[:,tile]
    
    available_time_steps = len(X_trans)
    max_possible_num_in = available_time_steps - num_pred
    
    if max_possible_num_in <= 0:
        raise ValueError(f"Not enough data for tile {tile}")
    
    effective_num_in = min(num_in, max_possible_num_in)
    X_ss, y_mm = split_sequences(X_trans, y_trans, effective_num_in, num_pred)
    
    target_seq_len = model_seq_len if model_seq_len is not None else effective_num_in
    if effective_num_in < target_seq_len and len(X_ss) > 0:
        padding_length = target_seq_len - effective_num_in
        padding_shape = (len(X_ss), padding_length, X_ss.shape[2])
        padding = np.zeros(padding_shape)
        X_ss = np.concatenate([padding, X_ss], axis=1)
    
    X = torch.Tensor(X_ss)
    y = torch.Tensor(y_mm)
    return X, y


def split_sequences(input_sequences, output_sequences, n_steps_in, n_steps_out):
    """Split sequences"""
    X, y = list(), list()
    for i in range(len(input_sequences)):
        end_ix = i + n_steps_in
        out_end_ix = end_ix + n_steps_out - 1
        
        if out_end_ix > len(input_sequences):
            break
            
        seq_x = input_sequences[i:end_ix]
        seq_y = output_sequences[end_ix-1:out_end_ix]
        
        X.append(seq_x)
        y.append(seq_y)
    
    return np.array(X), np.array(y)


def load_model_config(model_path):
    """Load model configuration from best_config JSON file"""
    model_path_obj = Path(model_path)
    
    # Try multiple locations for config file
    search_paths = [
        model_path_obj.parent.parent,  # experiments/experiment_X/
        model_path_obj.parent.parent.parent,  # experiments/
        model_path_obj.parent  # models/trial_XXX/
    ]
    
    config_files = []
    for search_path in search_paths:
        if search_path.exists():
            found = list(search_path.glob('best_config*.json'))
            if found:
                config_files.extend(found)
                break
    
    if config_files:
        config_file = config_files[0]
        try:
            with open(config_file, 'r') as f:
                config_data = json.load(f)
                if 'best_config' in config_data:
                    config = config_data['best_config'].copy()
                else:
                    config = config_data.copy()
                
                required_keys = ['d_model', 'nhead', 'num_layers', 'dropout', 'output_len', 'use_temporal_conv']
                for key in required_keys:
                    if key not in config:
                        print(f"    Warning: Config missing key '{key}', using default")
                
                if 'timing_bias_weight' not in config:
                    config['timing_bias_weight'] = 0.1
                if 'early_detection_weight' not in config:
                    config['early_detection_weight'] = 0.2
                
                return config
        except Exception as e:
            print(f"    ⚠️  Warning: Could not load config from {config_file}: {e}")
    
    # Default config
    return {
        'd_model': 256,
        'nhead': 8,
        'num_layers': 4,
        'dropout': 0.1,
        'output_len': 12,
        'use_temporal_conv': True,
        'timing_bias_weight': 0.1,
        'early_detection_weight': 0.2
    }


def evaluate_timing_patterns_on_ar(
    test_AR,
    model_configs,
    data_path,
    device
):
    """Evaluate all models on a single AR and collect timing differences for 7 central tiles
    (same tiles as used in hyperparameter search evaluation)"""
    
    print(f"\n{'='*100}")
    print(f"Evaluating AR {test_AR} for timing patterns")
    print(f"{'='*100}")
    
    rid_of_top = 1
    size = 9
    start_tile, before_plot, num_in, NOAA_first, NOAA_second = get_ar_settings_fixed(test_AR, rid_of_top)
    
    # Load data (using same loader as experiment_f_hyperparam_search.py for consistency)
    inputs, ii, time_arr, _, _ = load_and_preprocess_ar_eval_template(test_AR, data_path, rid_of_top, size)
    
    # Load all models
    models = {}
    lstm_fut = None
    
    # Load LSTM
    if 'lstm' in model_configs:
        lstm_config = model_configs['lstm']
        lstm_path = lstm_config['path']
        pat = r't(\d+)_r(\d+)_i(\d+)_n(\d+)_h(\d+)_e(\d+)_l([0-9.]+)\.pth'
        match = re.findall(pat, lstm_path)
        if match:
            lstm_num_pred, _, _, lstm_num_layers, lstm_hidden_size, n_epochs, lr = (
                int(x) if i!=6 else float(x) for i,x in enumerate(match[0])
            )
            lstm = LSTM(inputs.shape[1], lstm_hidden_size, lstm_num_layers, lstm_num_pred).to(device)
            sd = torch.load(lstm_path, map_location=device)
            new_sd = OrderedDict((k[7:] if k.startswith('module.') else k, v) for k,v in sd.items())
            lstm.load_state_dict(new_sd)
            lstm.eval()
            models['lstm'] = {
                'model': lstm,
                'name': 'LSTM',
                'fut_idx': lstm_num_pred - 1,
                'config': {'output_len': lstm_num_pred}
            }
            lstm_fut = lstm_num_pred - 1
            print(f"  Loaded LSTM")
    
    # Load Transformer models
    for exp_name, exp_config in model_configs.items():
        if exp_name == 'lstm':
            continue
        
        try:
            model_path = exp_config['path']
            config = load_model_config(model_path)
            
            if 'config' in exp_config:
                config.update(exp_config['config'])
            
            print(f"    Loading {exp_config['name']}")
            # Debug: Print key config values to verify correct loading
            if exp_name in ['exp_d', 'exp_f']:  # EarlyDetect models
                print(f"      Config: d_model={config.get('d_model')}, nhead={config.get('nhead')}, "
                      f"num_layers={config.get('num_layers')}, use_temporal_conv={config.get('use_temporal_conv')}, "
                      f"timing_bias={config.get('timing_bias_weight')}, early_detection={config.get('early_detection_weight')}")
            
            if config.get('d_model') is None:
                raise ValueError(f"Config missing d_model for {exp_name}")
            if config.get('nhead') is None:
                raise ValueError(f"Config missing nhead for {exp_name}")
            if config.get('num_layers') is None:
                raise ValueError(f"Config missing num_layers for {exp_name}")
            
            if exp_config.get('use_experiment_b_model', False):
                if SARTransformerLocalTile_B is None:
                    print(f"    Error: Experiment B model class not available. Skipping {exp_name}.")
                    continue
                
                transformer_class = SARTransformerLocalTile_B
                transformer = transformer_class(
                    input_dim=inputs.shape[1],
                    d_model=config['d_model'],
                    nhead=config['nhead'],
                    num_layers=config['num_layers'],
                    dropout=config.get('dropout', 0.0),
                    output_len=config['output_len'],
                    max_seq_len=150,
                    use_temporal_conv=config.get('use_temporal_conv', True)
                ).to(device)
            else:
                transformer = SARTransformerLocalTile(
                    input_dim=inputs.shape[1],
                    d_model=config['d_model'],
                    nhead=config['nhead'],
                    num_layers=config['num_layers'],
                    dropout=config.get('dropout', 0.0),
                    output_len=config['output_len'],
                    max_seq_len=150,
                    use_temporal_conv=config.get('use_temporal_conv', True),
                    timing_bias_weight=config.get('timing_bias_weight', 0.1),
                    early_detection_weight=config.get('early_detection_weight', 0.2)
                ).to(device)
            
            state_dict = torch.load(model_path, map_location=device)
            
            try:
                transformer.load_state_dict(state_dict, strict=True)
            except RuntimeError as e:
                error_msg = str(e)
                if "size mismatch" in error_msg:
                    print(f"    ❌ Architecture mismatch! Config may be incorrect.")
                    raise RuntimeError(f"Model architecture doesn't match checkpoint. Check config values.")
                else:
                    print(f"    ⚠️  Strict loading failed, trying lenient loading...")
                    transformer.load_state_dict(state_dict, strict=False)
            
            transformer.eval()
            
            models[exp_name] = {
                'model': transformer,
                'name': exp_config['name'],
                'fut_idx': config.get('output_len', 12) - 1,
                'config': config
            }
            print(f"  ✅ Loaded {exp_config['name']}")
        except Exception as e:
            print(f"  ❌ Warning: Could not load {exp_name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Evaluate on 7 tiles (same as hyperparameter search evaluation)
    # Only evaluate the central tiles where emergence is most prominent
    thr = -0.01
    st = 4
    all_tile_timing_diffs = {model_name: [] for model_name in models.keys()}
    
    for i in range(7):
        tile_idx = start_tile + i
        disp = tile_idx + 10  # Display index (matching hyperparameter search format)
        print(f"    Processing Tile {disp} (tile_idx {tile_idx})")
        
        try:
            # Prepare test data (using same function as experiment_f_hyperparam_search.py for consistency)
            X_test, y_test = lstm_ready_eval_template(
                tile_idx, size, inputs, ii, num_in, 12, model_seq_len=num_in
            )
            
            if X_test is None or y_test is None or len(X_test) == 0:
                continue
            
            X_test = X_test.to(device)
            Xt = X_test.view(X_test.size(0), num_in, X_test.size(2))
            y_true_tensor = y_test.to(device)
            
            # Process all data at once (same as experiment_f_hyperparam_search.py for consistent timing calculations)
            # Get predictions from all models
            predictions = {}
            
            with torch.no_grad():
                for model_name, model_info in models.items():
                    if model_name == 'lstm':
                        pred = model_info['model'](X_test)[:, model_info['fut_idx']].cpu().numpy()
                    else:
                        pred = model_info['model'](Xt)[:, model_info['fut_idx']].cpu().numpy()
                    
                    predictions[model_name] = pred
            
            true_values = y_true_tensor[:, lstm_fut if lstm_fut is not None else 11].cpu().numpy()
            
            # Recalibrate predictions and ground truth (EXACTLY as experiment_f_hyperparam_search.py)
            # Note: predictions are already 1D arrays from model output [:, fut_idx], so no need to flatten
            last_idx = ii.shape[1] - true_values.shape[0] - 1
            for model_name in predictions.keys():
                pred_values = predictions[model_name]
                # pred_values is already 1D (batch_size,), but ensure it's flattened to match experiment_f
                pred_1d = pred_values.flatten() if pred_values.ndim > 1 else pred_values
                pred_calibrated = recalibrate(pred_1d, ii[tile_idx, last_idx])
                predictions[model_name] = pred_calibrated
            
            # CRITICAL FIX: Recalibrate ground truth (same as experiment_f_hyperparam_search.py line 439)
            # true_values is already 1D (batch_size,), but ensure it's flattened to match experiment_f
            true_1d = true_values.flatten() if true_values.ndim > 1 else true_values
            true_calibrated = recalibrate(true_1d, ii[tile_idx, last_idx])
            
            # Calculate observed emergence window using recalibrated data (same as experiment_f_hyperparam_search.py)
            # Use np.gradient directly (not safe_gradient) to match experiment_f_hyperparam_search.py
            d_obs = np.gradient(smooth_with_numpy(true_calibrated))
            obs_start, obs_end = find_first_emergence_window(d_obs, thr, st)
            
            # Calculate timing difference for each model
            for model_name in models.keys():
                pred_calibrated = predictions[model_name]
                
                # Timing metrics using recalibrated data (same as experiment_f_hyperparam_search.py)
                # Use np.gradient directly to match experiment_f_hyperparam_search.py
                d_pred = np.gradient(pred_calibrated)
                pred_start, pred_end = find_first_emergence_window(d_pred, thr, st)
                
                # CRITICAL: Match exact logic from experiment_f_hyperparam_search.py line 199-203
                # Only require obs_start to be not None, then check pred_start separately
                timing_diff = None
                if obs_start is not None:
                    if pred_start is not None:
                        timing_diff = pred_start - obs_start
                
                # Only record timing differences when both observed and predicted emergence are detected
                # (same as experiment_f_hyperparam_search.py - it returns None if either is None)
                if timing_diff is not None:
                    all_tile_timing_diffs[model_name].append(timing_diff)
            
            # Cleanup after successful processing
            del X_test, y_test, y_true_tensor, Xt
            if tile_idx % 10 == 0:
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
                
        except RuntimeError as e:
            error_msg = str(e)
            # Check if this is a CUDA error that should be retried
            is_cuda_error = (
                "CUDA" in error_msg.upper() and 
                ("busy" in error_msg.lower() or "unavailable" in error_msg.lower() or 
                 "out of memory" in error_msg.lower() or "cuda" in error_msg.lower())
            )
            if is_cuda_error:
                # Propagate CUDA errors to outer retry loop
                raise
            # For other errors, skip this tile
            print(f"      ⚠️  Error processing tile {disp}: {error_msg[:100]}")
            continue
        except Exception as e:
            # Skip tiles with other errors
            print(f"      ⚠️  Error processing tile {disp}: {str(e)[:100]}")
            continue
    
    # Clean up
    for model_info in models.values():
        del model_info['model']
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # Calculate per-AR averages (matching experiment_f_hyperparam_search.py two-level averaging)
    # First level: average across tiles for this AR
    ar_avg_timing_diffs = {}
    for model_name, timing_diffs in all_tile_timing_diffs.items():
        if len(timing_diffs) > 0:
            # Filter out None values (same as mean_metric in experiment_f_hyperparam_search.py)
            valid_diffs = [td for td in timing_diffs if td is not None and not np.isnan(td)]
            ar_avg_timing_diffs[model_name] = np.mean(valid_diffs) if valid_diffs else None
        else:
            ar_avg_timing_diffs[model_name] = None
    
    return ar_avg_timing_diffs


def calculate_timing_statistics(all_ar_avg_timing_diffs):
    """Calculate timing pattern statistics for each model
    Uses two-level averaging to match experiment_f_hyperparam_search.py:
    1. First level: Average timing differences across tiles for each AR (done in evaluate_timing_patterns_on_ar)
    2. Second level: Average per-AR averages across all ARs (done here)
    This matches how avg_transformer_timing_diff is calculated in the CSV.
    """
    stats = {}
    
    for model_name, ar_avg_timing_diffs in all_ar_avg_timing_diffs.items():
        if len(ar_avg_timing_diffs) == 0:
            stats[model_name] = {
                'mean': None,
                'median': None,
                'std': None,
                'pct_early': None,
                'pct_late': None,
                'count': 0
            }
            continue
        
        # Second level: Average per-AR averages across all ARs
        # This matches experiment_f_hyperparam_search.py line 987: np.mean(all_transformer_timing)
        timing_array = np.array(ar_avg_timing_diffs)
        
        stats[model_name] = {
            'mean': np.mean(timing_array),  # This matches avg_transformer_timing_diff in CSV
            'median': np.median(timing_array),
            'std': np.std(timing_array),
            'pct_early': 100.0 * np.sum(timing_array < 0) / len(timing_array),
            'pct_late': 100.0 * np.sum(timing_array > 0) / len(timing_array),
            'count': len(timing_array)  # Number of ARs with valid timing differences
        }
    
    return stats


def main():
    parser = argparse.ArgumentParser(description='Calculate Timing Pattern Statistics for All Models')
    parser.add_argument('--data_path', type=str,
                       default='data',
                       help='Path to data directory')
    parser.add_argument('--output_dir', type=str,
                       default='results/timing_patterns',
                       help='Output directory')
    parser.add_argument('--test_ars', type=int, nargs='+',
                       default=[11698, 11726, 13165, 13179, 13183],
                       help='Test ARs to evaluate')
    
    args = parser.parse_args()
    
    # Check CUDA environment
    import os
    cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')
    print(f"CUDA_VISIBLE_DEVICES: {cuda_visible}")
    print(f"torch.cuda.is_available(): {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"torch.cuda.device_count(): {torch.cuda.device_count()}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Verify CUDA is actually working if using GPU
    if device.type == 'cuda':
        print(f"  Checking CUDA availability...")
        # Wait a moment for CUDA to be ready in SLURM environment
        time.sleep(2)
        
        # Try to explicitly initialize CUDA context by setting current device
        try:
            # Force CUDA initialization by setting current device
            torch.cuda.set_device(0)
            print(f"  CUDA device set to 0")
        except Exception as e:
            print(f"  Note: Setting CUDA device returned: {e}")
        
        try:
            # Test CUDA with a simple operation
            # Try creating tensor on CPU first, then move to GPU
            test_tensor_cpu = torch.zeros(1)
            test_tensor = test_tensor_cpu.to(device)
            device_name = torch.cuda.get_device_name(0)
            print(f"✅ CUDA device verified: {device_name}")
            
            # Try a simple computation to ensure context is working
            result = (test_tensor * 2).cpu()
            print(f"✅ CUDA computation test passed")
            
        except RuntimeError as e:
            error_msg = str(e)
            if "busy" in error_msg.lower() or "unavailable" in error_msg.lower():
                print(f"⚠️  Warning: CUDA device is busy or unavailable")
                print(f"   Error: {error_msg[:200]}")
                print(f"   Attempting workaround: trying CPU fallback for this run...")
                print(f"   (Note: This will be slower but should work)")
                # Fall back to CPU if CUDA is persistently unavailable
                device = torch.device('cpu')
                print(f"   Switched to CPU device")
            else:
                print(f"⚠️  Warning: CUDA device error: {error_msg[:200]}")
                print("   Will attempt to continue, but errors may occur...")
    else:
        print(f"⚠️  Warning: CUDA not available, but GPU was requested in SLURM")
        print(f"   This may indicate a SLURM allocation issue")
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Define model configurations (using ROOT relative paths)
    model_configs = {
        'lstm': {
            'path': str(ROOT / 'models' / 'checkpoints' / 'lstm_baseline.pth'),
            'name': 'LSTM',
        },
        'exp_b_no_conv1d': {
            'path': str(ROOT / 'models' / 'checkpoints' / 'baseline_no_conv1d.pth'),
            'name': 'Baseline',
            'use_experiment_b_model': True,
            'config': {'use_temporal_conv': False}
        },
        'exp_b_conv1d': {
            'path': str(ROOT / 'models' / 'checkpoints' / 'baseline_conv1d.pth'),
            'name': 'Baseline+Conv1D',
            'use_experiment_b_model': True,
            'config': {'use_temporal_conv': True}
        },
        'exp_d': {
            # EarlyDetect: timing loss without Conv1D
            'path': str(ROOT / 'models' / 'checkpoints' / 'earlydetect_no_conv1d.pth'),
            'name': 'EarlyDetect',
            'config': {'use_temporal_conv': False}
        },
        'exp_f': {
            # EarlyDetect+Conv1D: timing loss with Conv1D
            'path': str(ROOT / 'models' / 'checkpoints' / 'earlydetect_conv1d.pth'),
            'name': 'EarlyDetect+Conv1D',
            'config': {'use_temporal_conv': True}
        }
    }
    
    print(f"\n{'='*100}")
    print("TIMING PATTERN ANALYSIS FOR ALL MODELS")
    print(f"{'='*100}")
    print("Models to evaluate:")
    for key, config in model_configs.items():
        print(f"  - {config['name']}: {config['path']}")
    print(f"{'='*100}\n")
    
    # Collect per-AR averages (matching experiment_f_hyperparam_search.py two-level averaging approach)
    # This matches how avg_transformer_timing_diff is calculated in the CSV:
    # 1. Average timing differences across tiles for each AR
    # 2. Average those per-AR averages across all ARs
    all_ar_avg_timing_diffs = {}
    for model_name in model_configs.keys():
        all_ar_avg_timing_diffs[model_name] = []
    
    # Evaluate on all ARs (same 5 evaluation ARs as experiment_f_hyperparam_search.py)
    # Training ARs: 41 ARs (not used for evaluation)
    # Test/Evaluation ARs: [11698, 11726, 13165, 13179, 13183] (same as CSV)
    for test_AR in args.test_ars:
        max_retries = 5  # Increased retries
        retry_delay = 30  # Start with longer delay (30 seconds)
        
        # Longer initial delay to let CUDA settle
        if torch.cuda.is_available():
            print(f"  Waiting 5 seconds before starting AR {test_AR}...")
            time.sleep(5)
        
        for attempt in range(max_retries):
            try:
                # Minimal cleanup - avoid operations that might fail when CUDA is busy
                if torch.cuda.is_available():
                    try:
                        # Only do minimal cleanup - don't synchronize if CUDA is busy
                        torch.cuda.empty_cache()
                        import gc
                        gc.collect()
                    except RuntimeError:
                        pass  # Skip all cleanup if CUDA is busy
                
                ar_avg_timing_diffs = evaluate_timing_patterns_on_ar(
                    test_AR, model_configs, args.data_path, device
                )
                
                # Collect per-AR averages (second level of averaging will be done in calculate_timing_statistics)
                for model_name, ar_avg in ar_avg_timing_diffs.items():
                    if ar_avg is not None:
                        all_ar_avg_timing_diffs[model_name].append(ar_avg)
                
                break  # Success
                
            except RuntimeError as e:
                error_msg = str(e)
                # Check for various CUDA error patterns
                is_cuda_error = (
                    "CUDA" in error_msg.upper() and 
                    ("busy" in error_msg.lower() or "unavailable" in error_msg.lower() or 
                     "out of memory" in error_msg.lower() or "cuda" in error_msg.lower())
                )
                
                if is_cuda_error:
                    if attempt < max_retries - 1:
                        print(f"  ⚠️  CUDA busy for AR {test_AR}, attempt {attempt + 1}/{max_retries}")
                        print(f"  Waiting {retry_delay} seconds before retry...")
                        # Minimal cleanup - avoid operations that might fail
                        if torch.cuda.is_available():
                            try:
                                torch.cuda.empty_cache()
                                import gc
                                gc.collect()
                            except RuntimeError:
                                pass  # Skip cleanup if CUDA is busy
                        time.sleep(retry_delay)
                        retry_delay = min(int(retry_delay * 1.5), 120)  # Increase delay, cap at 2 minutes
                        continue
                    else:
                        print(f"  ❌ Failed to evaluate AR {test_AR} after {max_retries} attempts")
                        print(f"  Error: {error_msg[:200]}")
                        print(f"  💡 Tip: Check if another process is using the GPU with: nvidia-smi")
                        continue
                else:
                    # Not a CUDA error, re-raise
                    raise
            except Exception as e:
                print(f"  ❌ Error evaluating AR {test_AR}: {e}")
                import traceback
                traceback.print_exc()
                continue
    
    # Calculate statistics
    print(f"\n{'='*100}")
    print("CALCULATING TIMING PATTERN STATISTICS")
    print(f"{'='*100}")
    
    stats = calculate_timing_statistics(all_ar_avg_timing_diffs)
    
    # Create results table
    results_data = []
    for model_key, model_config in model_configs.items():
        model_name = model_config['name']
        if model_key in stats:
            s = stats[model_key]
            results_data.append({
                'Model': model_name,
                'Mean Timing Δ (hrs)': f"{s['mean']:.2f}" if s['mean'] is not None else 'N/A',
                'Median Timing Δ (hrs)': f"{s['median']:.2f}" if s['median'] is not None else 'N/A',
                'Std. Dev. of Timing Δ': f"{s['std']:.2f}" if s['std'] is not None else 'N/A',
                '% Early Forecasts (< 0h)': f"{s['pct_early']:.1f}" if s['pct_early'] is not None else 'N/A',
                '% Late Forecasts (> 0h)': f"{s['pct_late']:.1f}" if s['pct_late'] is not None else 'N/A',
                'Count': s['count']
            })
    
    # Create DataFrame
    df = pd.DataFrame(results_data)
    
    # Print table
    print("\n" + "="*100)
    print("TIMING PATTERN STATISTICS TABLE")
    print("="*100)
    print(df.to_string(index=False))
    print("="*100)
    
    # Save results
    df.to_csv(output_dir / 'timing_patterns_table.csv', index=False)
    
    # Also save detailed data
    detailed_data = {
        'model_names': {k: model_configs[k]['name'] for k in model_configs.keys()},
        'per_ar_avg_timing_differences': {model_configs[k]['name']: all_ar_avg_timing_diffs[k] for k in model_configs.keys()},
        'statistics': {model_configs[k]['name']: stats[k] for k in model_configs.keys() if k in stats},
        'evaluation_ars': args.test_ars,
        'averaging_method': 'two-level: (tiles → AR) then (ARs → final), matching experiment_f_hyperparam_search.py'
    }
    
    with open(output_dir / 'timing_patterns_detailed.json', 'w') as f:
        json.dump(detailed_data, f, indent=2, default=str)
    
    print(f"\nResults saved to: {output_dir}")
    print(f"  - timing_patterns_table.csv")
    print(f"  - timing_patterns_detailed.json")


if __name__ == '__main__':
    main()

