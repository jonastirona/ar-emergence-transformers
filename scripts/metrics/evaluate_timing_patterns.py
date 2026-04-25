#!/usr/bin/env python3
"""
Script to calculate operational lead-time pattern statistics for all models.

This script evaluates all models (LSTM, Baseline, Baseline+Conv1D, EarlyDetect, EarlyDetect+Conv1D)
on all test ARs and calculates lead-time pattern statistics:
- Mean Lead Time (hrs)
- Median Lead Time (hrs)
- Std. Dev. of Lead Time
- % Early Warnings (> 0h)
- % Late Warnings (< 0h)

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


def find_emergence_windows(signal, threshold=-0.01, min_duration=4):
    """Find all sustained emergence windows as (start, end_exclusive)."""
    emergence_indices = emergence_indication(signal, threshold, min_duration)
    windows = []
    in_window = False
    start_idx = None
    for i, val in enumerate(emergence_indices):
        if val != 0 and not in_window:
            in_window = True
            start_idx = i
        elif val == 0 and in_window:
            windows.append((start_idx, i))
            in_window = False
            start_idx = None
    if in_window and start_idx is not None:
        windows.append((start_idx, len(emergence_indices)))
    return windows


def select_onset_from_windows(windows, prediction_start_idx=0, lookback_tolerance=12):
    """Select onset relative to prediction start, prioritizing ongoing-at-start events.

    lookback_tolerance: if an event ended within this many indices before
    prediction_start_idx, anchor obs_start to prediction_start_idx rather than
    skipping to the next (potentially distant) event.
    """
    if not windows:
        return None, None, "no_window"

    # 1. Event active at prediction start → anchor onset to prediction_start_idx.
    for start, end in windows:
        if start <= prediction_start_idx < end:
            return prediction_start_idx, end, "window_contains_prediction_start"

    # 2. Event that ended just before prediction start (within lookback_tolerance) →
    #    anchor onset to prediction_start_idx so the nearby event is not skipped.
    #    Pick the most recent such window.
    best_recent = None
    for start, end in windows:
        if end <= prediction_start_idx and (prediction_start_idx - end) <= lookback_tolerance:
            if best_recent is None or end > best_recent[1]:
                best_recent = (start, end)
    if best_recent is not None:
        return prediction_start_idx, prediction_start_idx, "window_just_before_prediction_start"

    # 3. First event starting at or after prediction start.
    for start, end in windows:
        if start >= prediction_start_idx:
            return start, end, "first_window_after_prediction_start"

    return None, None, "only_windows_before_prediction_start"


def classify_operational_outcome(has_observed, has_alert, lead_time, tp_min=0, tp_max=24):
    """Classify tile outcome for operational early warning."""
    if not has_observed and not has_alert:
        return "TN"
    if not has_observed and has_alert:
        return "FP_no_event"
    if has_observed and not has_alert:
        return "FN_no_alert"
    if lead_time is None:
        return "FN_no_lead_time"
    if tp_min <= lead_time <= tp_max:
        return "TP"
    if lead_time < tp_min:
        return "FN_too_late"
    return "FP_too_early"


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

    search_paths = [
        model_path_obj.parent.parent,
        model_path_obj.parent.parent.parent,
        model_path_obj.parent
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
    """Evaluate all models on a single AR and collect tile-level lead times for 7 central tiles."""

    print(f"\n{'='*100}")
    print(f"Evaluating AR {test_AR} for timing patterns")
    print(f"{'='*100}")

    rid_of_top = 1
    size = 9
    start_tile, before_plot, num_in, NOAA_first, NOAA_second = get_ar_settings_fixed(test_AR, rid_of_top)

    inputs, ii, time_arr, _, _ = load_and_preprocess_ar_eval_template(test_AR, data_path, rid_of_top, size)

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
            if exp_name in ['exp_d', 'exp_f']:
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
                    print(f"    Error: Baseline model class not available. Skipping {exp_name}.")
                    continue

                transformer = SARTransformerLocalTile_B(
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

    thr = -0.01
    st = 4
    forecast_horizon_hours = 12
    all_tile_lead_times = {model_name: [] for model_name in models.keys()}
    all_tile_outcomes = {
        model_name: {
            'TP': 0,
            'FP_no_event': 0,
            'FP_too_early': 0,
            'FN_no_alert': 0,
            'FN_no_lead_time': 0,
            'FN_too_late': 0,
            'TN': 0
        } for model_name in models.keys()
    }
    tile_diagnostics = []

    for i in range(7):
        tile_idx = start_tile + i
        disp = tile_idx + 10
        print(f"    Processing Tile {disp} (tile_idx {tile_idx})")

        try:
            X_test, y_test = lstm_ready_eval_template(
                tile_idx, size, inputs, ii, num_in, 12, model_seq_len=num_in
            )

            if X_test is None or y_test is None or len(X_test) == 0:
                continue

            X_test = X_test.to(device)
            Xt = X_test.view(X_test.size(0), num_in, X_test.size(2))
            y_true_tensor = y_test.to(device)

            predictions = {}

            with torch.no_grad():
                for model_name, model_info in models.items():
                    if model_name == 'lstm':
                        pred = model_info['model'](X_test)[:, model_info['fut_idx']].cpu().numpy()
                    else:
                        pred = model_info['model'](Xt)[:, model_info['fut_idx']].cpu().numpy()

                    predictions[model_name] = pred

            true_values = y_true_tensor[:, lstm_fut if lstm_fut is not None else 11].cpu().numpy()

            # Recalibrate predictions
            last_idx = ii.shape[1] - true_values.shape[0] - 1
            recal_point = ii[tile_idx, last_idx]
            for model_name in predictions.keys():
                pred_values = predictions[model_name]
                pred_1d = pred_values.flatten() if pred_values.ndim > 1 else pred_values
                predictions[model_name] = recalibrate(pred_1d, recal_point)

            # Build observed signal: concat(before_norm, recalibrated_ground_truth)
            true_calibrated_obs = recalibrate(true_values, recal_point)
            before_norm_obs = ii[tile_idx, last_idx - before_plot:last_idx]
            combined_obs = np.concatenate((before_norm_obs, true_calibrated_obs))
            d_obs = np.gradient(smooth_with_numpy(combined_obs))
            obs_windows = find_emergence_windows(d_obs, thr, st)
            obs_start_comb, obs_end_comb, obs_reason = select_onset_from_windows(
                obs_windows,
                prediction_start_idx=before_plot
            )

            # Convert combined-array coords → prediction-window-relative coords
            if obs_start_comb is not None:
                obs_start = obs_start_comb - before_plot
                obs_end = obs_end_comb - before_plot if obs_end_comb is not None else None
            else:
                obs_start, obs_end = None, None

            tile_diag = {
                'tile_display': disp,
                'tile_index': tile_idx,
                'last_idx': last_idx,
                'before_plot': before_plot,
                'observed_windows_combined': obs_windows,
                'observed_selected_combined': (obs_start_comb, obs_end_comb),
                'observed_selected_relative': (obs_start, obs_end),
                'observed_selected': (obs_start, obs_end),
                'observed_selection_reason': obs_reason,
                'models': {}
            }

            for model_name in models.keys():
                pred_calibrated = predictions[model_name]

                d_pred = np.gradient(pred_calibrated)
                pred_windows = find_emergence_windows(d_pred, thr, st)
                pred_start, pred_end, pred_reason = select_onset_from_windows(pred_windows, prediction_start_idx=0)

                # T_lead = (obs_start - pred_start) + forecast_horizon_hours
                lead_time = None
                if obs_start is not None and pred_start is not None:
                    lead_time = (obs_start - pred_start) + forecast_horizon_hours
                    all_tile_lead_times[model_name].append(lead_time)

                has_observed = obs_start is not None
                has_alert = pred_start is not None
                outcome = classify_operational_outcome(has_observed, has_alert, lead_time, tp_min=0, tp_max=24)
                all_tile_outcomes[model_name][outcome] += 1

                tile_diag['models'][model_name] = {
                    'predicted_windows': pred_windows,
                    'predicted_selected': (pred_start, pred_end),
                    'prediction_selection_reason': pred_reason,
                    'lead_time_hours': lead_time,
                    'outcome': outcome
                }

            tile_diagnostics.append(tile_diag)

            del X_test, y_test, y_true_tensor, Xt
            if tile_idx % 10 == 0:
                torch.cuda.empty_cache() if torch.cuda.is_available() else None

        except RuntimeError as e:
            error_msg = str(e)
            is_cuda_error = (
                "CUDA" in error_msg.upper() and
                ("busy" in error_msg.lower() or "unavailable" in error_msg.lower() or
                 "out of memory" in error_msg.lower() or "cuda" in error_msg.lower())
            )
            if is_cuda_error:
                raise
            print(f"      ⚠️  Error processing tile {disp}: {error_msg[:100]}")
            continue
        except Exception as e:
            print(f"      ⚠️  Error processing tile {disp}: {str(e)[:100]}")
            continue

    for model_info in models.values():
        del model_info['model']
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    return all_tile_lead_times, all_tile_outcomes, tile_diagnostics


def calculate_lead_time_statistics(all_pooled_lead_times):
    """Calculate pooled tile-level lead-time statistics for each model."""
    stats = {}

    for model_name, pooled_lead_times in all_pooled_lead_times.items():
        valid_leads = [lt for lt in pooled_lead_times if lt is not None and not np.isnan(lt)]
        if len(valid_leads) == 0:
            stats[model_name] = {
                'mean': None,
                'median': None,
                'std': None,
                'pct_early': None,
                'pct_late': None,
                'pct_tp_window': None,
                'count': 0
            }
            continue

        lead_array = np.array(valid_leads)

        stats[model_name] = {
            'mean': np.mean(lead_array),
            'median': np.median(lead_array),
            'std': np.std(lead_array),
            'pct_early': 100.0 * np.sum(lead_array > 0) / len(lead_array),
            'pct_late': 100.0 * np.sum(lead_array < 0) / len(lead_array),
            'pct_tp_window': 100.0 * np.sum((lead_array >= 0) & (lead_array <= 24)) / len(lead_array),
            'count': len(lead_array)
        }

    return stats


def main():
    parser = argparse.ArgumentParser(description='Calculate Operational Lead-Time Pattern Statistics for All Models')
    parser.add_argument('--data_path', type=str,
                       default='data',
                       help='Path to data directory')
    parser.add_argument('--output_dir', type=str,
                       default='results/lead_time_patterns',
                       help='Output directory')
    parser.add_argument('--test_ars', type=int, nargs='+',
                       default=[11698, 11726, 13165, 13179, 13183],
                       help='Test ARs to evaluate')

    args = parser.parse_args()

    import os
    cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')
    print(f"CUDA_VISIBLE_DEVICES: {cuda_visible}")
    print(f"torch.cuda.is_available(): {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"torch.cuda.device_count(): {torch.cuda.device_count()}")

    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Apple Silicon (MPS) GPU!")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    if device.type == 'cuda':
        print(f"  Checking CUDA availability...")
        time.sleep(2)

        try:
            torch.cuda.set_device(0)
            print(f"  CUDA device set to 0")
        except Exception as e:
            print(f"  Note: Setting CUDA device returned: {e}")

        try:
            test_tensor_cpu = torch.zeros(1)
            test_tensor = test_tensor_cpu.to(device)
            device_name = torch.cuda.get_device_name(0)
            print(f"✅ CUDA device verified: {device_name}")
            result = (test_tensor * 2).cpu()
            print(f"✅ CUDA computation test passed")
        except RuntimeError as e:
            error_msg = str(e)
            if "busy" in error_msg.lower() or "unavailable" in error_msg.lower():
                print(f"⚠️  Warning: CUDA device is busy or unavailable")
                print(f"   Error: {error_msg[:200]}")
                print(f"   Falling back to CPU...")
                device = torch.device('cpu')
                print(f"   Switched to CPU device")
            else:
                print(f"⚠️  Warning: CUDA device error: {error_msg[:200]}")
                print("   Will attempt to continue, but errors may occur...")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

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
            'path': str(ROOT / 'models' / 'checkpoints' / 'earlydetect_no_conv1d.pth'),
            'name': 'EarlyDetect',
            'config': {'use_temporal_conv': False}
        },
        'exp_f': {
            'path': str(ROOT / 'models' / 'checkpoints' / 'earlydetect_conv1d.pth'),
            'name': 'EarlyDetect+Conv1D',
            'config': {'use_temporal_conv': True}
        }
    }

    script_start = time.time()
    print(f"\n{'='*100}")
    print("TIMING PATTERN ANALYSIS FOR ALL MODELS")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*100}")
    print("Models to evaluate:")
    for key, config in model_configs.items():
        print(f"  - {config['name']}: {config['path']}")
    print(f"{'='*100}\n")

    all_pooled_lead_times = {}
    aggregated_outcomes = {}
    per_ar_tile_diagnostics = {}
    for model_name in model_configs.keys():
        all_pooled_lead_times[model_name] = []
        aggregated_outcomes[model_name] = {
            'TP': 0,
            'FP_no_event': 0,
            'FP_too_early': 0,
            'FN_no_alert': 0,
            'FN_no_lead_time': 0,
            'FN_too_late': 0,
            'TN': 0
        }

    for ar_num, test_AR in enumerate(args.test_ars, 1):
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] AR {test_AR} ({ar_num}/{len(args.test_ars)})")
        ar_start = time.time()
        max_retries = 5
        retry_delay = 30

        for attempt in range(max_retries):
            try:
                if torch.cuda.is_available():
                    try:
                        torch.cuda.empty_cache()
                        import gc
                        gc.collect()
                    except RuntimeError:
                        pass

                ar_tile_lead_times, ar_outcomes, ar_tile_diagnostics = evaluate_timing_patterns_on_ar(
                    test_AR, model_configs, args.data_path, device
                )

                for model_name, lead_times in ar_tile_lead_times.items():
                    valid = [lt for lt in lead_times if lt is not None and not np.isnan(lt)]
                    all_pooled_lead_times[model_name].extend(valid)

                for model_name, counts in ar_outcomes.items():
                    for outcome_name, count in counts.items():
                        aggregated_outcomes[model_name][outcome_name] += count

                per_ar_tile_diagnostics[str(test_AR)] = ar_tile_diagnostics

                ar_elapsed = time.time() - ar_start
                print(f"  ✅ AR {test_AR} done in {ar_elapsed:.1f}s")
                break

            except RuntimeError as e:
                error_msg = str(e)
                is_cuda_error = (
                    "CUDA" in error_msg.upper() and
                    ("busy" in error_msg.lower() or "unavailable" in error_msg.lower() or
                     "out of memory" in error_msg.lower() or "cuda" in error_msg.lower())
                )

                if is_cuda_error:
                    if attempt < max_retries - 1:
                        print(f"  ⚠️  CUDA busy for AR {test_AR}, attempt {attempt + 1}/{max_retries}")
                        print(f"  Waiting {retry_delay} seconds before retry...")
                        if torch.cuda.is_available():
                            try:
                                torch.cuda.empty_cache()
                                import gc
                                gc.collect()
                            except RuntimeError:
                                pass
                        time.sleep(retry_delay)
                        retry_delay = min(int(retry_delay * 1.5), 120)
                        continue
                    else:
                        print(f"  ❌ Failed to evaluate AR {test_AR} after {max_retries} attempts")
                        print(f"  Error: {error_msg[:200]}")
                        print(f"  💡 Tip: Check if another process is using the GPU with: nvidia-smi")
                        continue
                else:
                    raise
            except Exception as e:
                print(f"  ❌ Error evaluating AR {test_AR}: {e}")
                import traceback
                traceback.print_exc()
                continue

    print(f"\n{'='*100}")
    print("CALCULATING LEAD-TIME PATTERN STATISTICS")
    print(f"{'='*100}")

    stats = calculate_lead_time_statistics(all_pooled_lead_times)

    results_data = []
    for model_key, model_config in model_configs.items():
        model_name = model_config['name']
        if model_key in stats:
            s = stats[model_key]
            results_data.append({
                'Model': model_name,
                'Mean Lead Time (hrs)': f"{s['mean']:.2f}" if s['mean'] is not None else 'N/A',
                'Median Lead Time (hrs)': f"{s['median']:.2f}" if s['median'] is not None else 'N/A',
                'Std. Dev. of Lead Time': f"{s['std']:.2f}" if s['std'] is not None else 'N/A',
                '% Early Warnings (> 0h)': f"{s['pct_early']:.1f}" if s['pct_early'] is not None else 'N/A',
                '% Late Warnings (< 0h)': f"{s['pct_late']:.1f}" if s['pct_late'] is not None else 'N/A',
                '% Within TP Window (0-24h)': f"{s['pct_tp_window']:.1f}" if s['pct_tp_window'] is not None else 'N/A',
                'Count': s['count']
            })

    df = pd.DataFrame(results_data)

    print("\n" + "="*100)
    print("LEAD-TIME PATTERN STATISTICS TABLE")
    print("="*100)
    print(df.to_string(index=False))
    print("="*100)

    df.to_csv(output_dir / 'lead_time_patterns_table.csv', index=False)

    detailed_data = {
        'model_names': {k: model_configs[k]['name'] for k in model_configs.keys()},
        'pooled_tile_lead_times': {model_configs[k]['name']: all_pooled_lead_times[k] for k in model_configs.keys()},
        'statistics': {model_configs[k]['name']: stats[k] for k in model_configs.keys() if k in stats},
        'operational_outcome_counts': {model_configs[k]['name']: aggregated_outcomes[k] for k in model_configs.keys()},
        'tile_level_diagnostics': per_ar_tile_diagnostics,
        'evaluation_ars': args.test_ars,
        'forecast_horizon_hours': 12,
        'tp_window_hours': [0, 24],
        'averaging_method': 'pooled tile-level: (all valid tiles across ARs) → final'
    }

    with open(output_dir / 'lead_time_patterns_detailed.json', 'w') as f:
        json.dump(detailed_data, f, indent=2, default=str)

    total_elapsed = time.time() - script_start
    print(f"\nResults saved to: {output_dir}")
    print(f"  - lead_time_patterns_table.csv")
    print(f"  - lead_time_patterns_detailed.json")
    print(f"\nTotal runtime: {total_elapsed/60:.1f} min ({total_elapsed:.0f}s)")
    print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == '__main__':
    main()
