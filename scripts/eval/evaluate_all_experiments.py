#!/usr/bin/env python3

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
import io
import gzip
import zipfile
from collections import OrderedDict
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib import gridspec
from matplotlib.ticker import MaxNLocator
import sys

from astropy.io import fits
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.extend(
    [
        str(ROOT / "scripts"),
        str(ROOT / "scripts/models"),
        str(ROOT / "scripts/data"),
    ]
)

from functions_spyros import (
    LSTM,
    lstm_ready,
    min_max_scaling,
    smooth_with_numpy,
    emergence_indication,
    recalibrate,
    calculate_metrics,
    add_grid_lines,
    highlight_tile,
)
from baseline_transformer import SARTransformerLocalTile as BaselineTransformer
from early_detect_transformer import SARTransformerLocalTile as EarlyDetectTransformer
from data_loader import load_ar_data_enhanced, cross_ar_tile_data_preparation_attention

FONT_SCALE = 1


def scaled_font(size):
    return size * FONT_SCALE


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
    """Select onset relative to prediction start, prioritizing ongoing-at-start events."""
    if not windows:
        return None, None, "no_window"

    for start, end in windows:
        if start <= prediction_start_idx < end:
            return prediction_start_idx, end, "window_contains_prediction_start"

    best_recent = None
    for start, end in windows:
        if end <= prediction_start_idx and (prediction_start_idx - end) <= lookback_tolerance:
            if best_recent is None or end > best_recent[1]:
                best_recent = (start, end)
    if best_recent is not None:
        return prediction_start_idx, prediction_start_idx, "window_just_before_prediction_start"

    for start, end in windows:
        if start >= prediction_start_idx:
            return start, end, "first_window_after_prediction_start"

    return None, None, "only_windows_before_prediction_start"


def calculate_emergence_timing_normalized(true_norm, pred_lstm_norm, pred_transformer_norm, threshold=-0.01, min_duration=4):
    """Calculate emergence timing using normalized data (for accurate timing)"""
    d_obs = np.gradient(smooth_with_numpy(true_norm))
    d_lstm = np.gradient(pred_lstm_norm)
    d_transformer = np.gradient(pred_transformer_norm)
    
    obs_start, obs_end, _ = select_onset_from_windows(find_emergence_windows(d_obs, threshold, min_duration))
    lstm_start, lstm_end, _ = select_onset_from_windows(find_emergence_windows(d_lstm, threshold, min_duration))
    transformer_start, transformer_end, _ = select_onset_from_windows(find_emergence_windows(d_transformer, threshold, min_duration))
    
    lstm_lead_time = None
    transformer_lead_time = None
    
    if obs_start is not None:
        if lstm_start is not None:
            lstm_lead_time = (obs_start - lstm_start) + 12
        if transformer_start is not None:
            transformer_lead_time = (obs_start - transformer_start) + 12
    
    return {
        'lstm': {
            'emergence_lead_time': lstm_lead_time,
            'emergence_window': (lstm_start, lstm_end) if lstm_start is not None else None
        },
        'transformer': {
            'emergence_lead_time': transformer_lead_time,
            'emergence_window': (transformer_start, transformer_end) if transformer_start is not None else None
        },
        'observed': {
            'emergence_window': (obs_start, obs_end) if obs_start is not None else None
        }
    }

def calculate_accuracy_metrics_denormalized(true_raw, pred_lstm_raw, pred_transformer_raw, timing_window):
    """Calculate accuracy metrics using denormalized data"""
    def calc_basic_metrics(y_true, y_pred):
        mae = np.mean(np.abs(y_true - y_pred))
        mse = np.mean((y_true - y_pred) ** 2)
        rmse = np.sqrt(mse)
        r2 = 1 - np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2)
        return mae, rmse, r2
    
    lstm_mae, lstm_rmse, lstm_r2 = calc_basic_metrics(true_raw, pred_lstm_raw)
    transformer_mae, transformer_rmse, transformer_r2 = calc_basic_metrics(true_raw, pred_transformer_raw)
    
    lstm_emerg_mae, lstm_emerg_rmse, lstm_emerg_r2 = None, None, None
    transformer_emerg_mae, transformer_emerg_rmse, transformer_emerg_r2 = None, None, None
    
    if timing_window is not None:
        obs_start, obs_end = timing_window
        if obs_start is not None and obs_end is not None:
            window_true = true_raw[obs_start:obs_end]
            window_lstm = pred_lstm_raw[obs_start:obs_end]
            window_transformer = pred_transformer_raw[obs_start:obs_end]
            
            if len(window_true) > 0:
                lstm_emerg_mae, lstm_emerg_rmse, lstm_emerg_r2 = calc_basic_metrics(window_true, window_lstm)
                transformer_emerg_mae, transformer_emerg_rmse, transformer_emerg_r2 = calc_basic_metrics(window_true, window_transformer)
    
    return {
        'lstm': {
            'MAE': lstm_mae,
            'RMSE': lstm_rmse,
            'R2': lstm_r2,
            'emerg_MAE': lstm_emerg_mae,
            'emerg_RMSE': lstm_emerg_rmse,
            'emerg_R2': lstm_emerg_r2
        },
        'transformer': {
            'MAE': transformer_mae,
            'RMSE': transformer_rmse,
            'R2': transformer_r2,
            'emerg_MAE': transformer_emerg_mae,
            'emerg_RMSE': transformer_emerg_rmse,
            'emerg_R2': transformer_emerg_r2
        }
    }

def calculate_emergence_metrics_detailed(true_norm, pred_lstm_norm, pred_transformer_norm, true_raw, pred_lstm_raw, pred_transformer_raw, time_arr, threshold=-0.01, min_duration=4):
    """Calculate emergence metrics with timing from normalized data and accuracy from denormalized data"""
    # Phase 1: Calculate timing on NORMALIZED data (for accurate timing)
    timing_metrics = calculate_emergence_timing_normalized(
        true_norm, pred_lstm_norm, pred_transformer_norm, 
        threshold, min_duration
    )
    
    # Phase 2: Calculate accuracy metrics on DENORMALIZED data
    accuracy_metrics = calculate_accuracy_metrics_denormalized(
        true_raw, pred_lstm_raw, pred_transformer_raw,
        timing_metrics['observed']['emergence_window']
    )
    
    # Combine both
    return {
        'lstm': {
            **timing_metrics['lstm'],
            **accuracy_metrics['lstm']
        },
        'transformer': {
            **timing_metrics['transformer'],
            **accuracy_metrics['transformer']
        },
        'observed': timing_metrics['observed']
    }

def format_emergence_status(lead_time, has_observed, has_predicted):
    """Format emergence lead time into table status string

    Status meanings:
    - "Quiet": No emergence in reality
    - "FP": Model predicted emergence but there was none (False Positive)
    - "FN": Model didn't predict emergence when there was one (False Negative)
    - "Xh Alarm": Lead time in hours (positive = early alert, negative = late)
    """
    if not has_observed:
        if has_predicted:
            return "FP"
        else:
            return "Quiet"
    else:
        if not has_predicted or lead_time is None:
            return "FN"
        else:
            return f"{int(lead_time)}h Alarm"


def export_emergence_timing_table(all_tile_metrics, test_AR, start_tile, output_dir, model_configs):
    """Export emergence lead times per tile per model to CSV in table format"""
    csv_dir = Path(output_dir) / 'unified_evaluations' / 'csv'
    csv_dir.mkdir(parents=True, exist_ok=True)
    
    # Get all model keys (excluding 'observed')
    all_model_keys = set()
    for tile_metrics in all_tile_metrics:
        all_model_keys.update([k for k in tile_metrics.keys() if k != 'observed'])
    
    # Get model display names
    model_names_map = {}
    for key in all_model_keys:
        if key in model_configs:
            model_names_map[key] = model_configs[key]['name']
        else:
            model_names_map[key] = key
    
    # Create table data: rows = models, columns = tiles + overall
    table_data = []
    
    for model_key in sorted(all_model_keys):
        model_name = model_names_map[model_key]
        row = {'Model': model_name}
        
        # Process each tile
        tile_lead_times = []
        for i, tile_metrics in enumerate(all_tile_metrics):
            tile_num = start_tile + i + 10  # Display tile number (1-indexed)
            
            # Check if observed emergence exists
            obs_metrics = tile_metrics.get('observed', {})
            has_observed = obs_metrics.get('emergence_window') is not None
            
            # Check model prediction
            model_metrics = tile_metrics.get(model_key, {})
            lead_time = model_metrics.get('emergence_lead_time')
            # Check if model predicted emergence (has emergence_window)
            pred_window = model_metrics.get('emergence_window')
            has_predicted = pred_window is not None
            
            # Format status
            status = format_emergence_status(lead_time, has_observed, has_predicted)
            row[f'Tile {tile_num}'] = status
            
            # Collect lead times for overall calculation (only if both observed and predicted)
            if has_observed and has_predicted and lead_time is not None:
                tile_lead_times.append(lead_time)
        
        # Calculate overall status
        if len(tile_lead_times) > 0:
            # Use mean lead time for overall
            overall_lead_time = np.mean(tile_lead_times)
            overall_status = format_emergence_status(overall_lead_time, True, True)
        else:
            # Check if any tile had observed emergence
            any_observed = any(
                tile_metrics.get('observed', {}).get('emergence_window') is not None
                for tile_metrics in all_tile_metrics
            )
            if any_observed:
                overall_status = "No pred"
            else:
                overall_status = "Quiet"
        
        row['Overall'] = overall_status
        table_data.append(row)
    
    # Create DataFrame and save
    df = pd.DataFrame(table_data)
    csv_path = csv_dir / f"AR{test_AR}_emergence_timing_table.csv"
    df.to_csv(csv_path, index=False)
    
    print(f"  Saved emergence timing table to {csv_path}")
    return csv_path, df


def export_metrics_to_csv(all_tile_metrics, test_AR, output_dir, model_configs):
    """Export aggregated metrics (mean ± std) across all tiles to CSV"""
    # Get all model keys that have metrics
    all_model_keys = set()
    for tile_metrics in all_tile_metrics:
        all_model_keys.update([k for k in tile_metrics.keys() if k != 'observed'])
    
    # Get model display names
    model_names_map = {}
    for key in all_model_keys:
        if key in model_configs:
            model_names_map[key] = model_configs[key]['name']
        else:
            model_names_map[key] = key
    
    # Collect all metrics for each model across tiles
    metrics_by_model = {}
    for model_key in all_model_keys:
        metrics_by_model[model_key] = {
            'MAE': [],
            'RMSE': [],
            'R2': [],
            'emerg_MAE': [],
            'emerg_RMSE': [],
            'emerg_R2': [],
            'emergence_lead_time': []
        }
    
    # Aggregate metrics from all tiles
    for tile_metrics in all_tile_metrics:
        for model_key in all_model_keys:
            if model_key in tile_metrics:
                model_metrics = tile_metrics[model_key]
                for metric_name in metrics_by_model[model_key].keys():
                    value = model_metrics.get(metric_name)
                    if value is not None:
                        metrics_by_model[model_key][metric_name].append(value)
    
    # Calculate mean ± std for each metric
    csv_data = []
    for model_key in sorted(all_model_keys):
        model_name = model_names_map[model_key]
        row = {'Model': model_name}
        
        for metric_name in ['MAE', 'RMSE', 'R2', 'emerg_MAE', 'emerg_RMSE', 'emerg_R2', 'emergence_lead_time']:
            values = metrics_by_model[model_key][metric_name]
            if len(values) > 0:
                mean_val = np.mean(values)
                std_val = np.std(values)
                row[metric_name] = f'{mean_val:.4f} ± {std_val:.4f}'
            else:
                row[metric_name] = 'N/A'
        
        csv_data.append(row)
    
    # Create DataFrame and save
    df = pd.DataFrame(csv_data)
    csv_dir = Path(output_dir) / 'unified_evaluations' / 'csv'
    csv_dir.mkdir(parents=True, exist_ok=True)
    csv_path = csv_dir / f"AR{test_AR}_metrics.csv"
    df.to_csv(csv_path, index=False)
    
    print(f"  Saved metrics CSV to {csv_path}")
    return csv_path, df


def export_combined_emergence_timing_table(all_results, output_dir, model_configs):
    """Export combined emergence timing table for all ARs in wide format"""
    csv_dir = Path(output_dir) / 'unified_evaluations' / 'csv'
    csv_dir.mkdir(parents=True, exist_ok=True)
    
    # Get all unique model keys across all ARs
    all_model_keys = set()
    for all_tile_metrics in all_results.values():
        for tile_metrics in all_tile_metrics:
            all_model_keys.update([k for k in tile_metrics.keys() if k != 'observed'])
    
    # Get model display names
    model_names_map = {}
    for key in all_model_keys:
        if key in model_configs:
            model_names_map[key] = model_configs[key]['name']
        else:
            model_names_map[key] = key
    
    # Collect all tile numbers across all ARs to determine column structure
    all_tile_cols = set()
    for test_AR, all_tile_metrics in all_results.items():
        rid_of_top = 1
        start_tile, _, _, _, _ = get_ar_settings_fixed(test_AR, rid_of_top)
        for i in range(len(all_tile_metrics)):
            tile_num = start_tile + i + 10
            all_tile_cols.add(f'Tile {tile_num}')
    all_tile_cols = sorted(all_tile_cols, key=lambda x: int(x.split()[1]))
    
    # Create table data: rows = (AR, Model), columns = tiles + overall
    table_data = []
    
    for test_AR in sorted(all_results.keys()):
        all_tile_metrics = all_results[test_AR]
        rid_of_top = 1
        start_tile, _, _, _, _ = get_ar_settings_fixed(test_AR, rid_of_top)
        
        for model_key in sorted(all_model_keys):
            model_name = model_names_map[model_key]
            row = {'AR': test_AR, 'Model': model_name}
            
            # Process each tile for this AR
            tile_lead_times = []
            for i, tile_metrics in enumerate(all_tile_metrics):
                tile_num = start_tile + i + 10
                tile_col = f'Tile {tile_num}'
                
                # Check if observed emergence exists
                obs_metrics = tile_metrics.get('observed', {})
                has_observed = obs_metrics.get('emergence_window') is not None
                
                # Check model prediction
                model_metrics = tile_metrics.get(model_key, {})
                lead_time = model_metrics.get('emergence_lead_time')
                pred_window = model_metrics.get('emergence_window')
                has_predicted = pred_window is not None
                
                # Format status
                status = format_emergence_status(lead_time, has_observed, has_predicted)
                row[tile_col] = status
                
                # Collect lead times for overall calculation
                if has_observed and has_predicted and lead_time is not None:
                    tile_lead_times.append(lead_time)
            
            # Calculate overall status
            if len(tile_lead_times) > 0:
                overall_timing = np.mean(tile_lead_times)
                overall_status = format_emergence_status(overall_timing, True, True)
            else:
                any_observed = any(
                    tile_metrics.get('observed', {}).get('emergence_window') is not None
                    for tile_metrics in all_tile_metrics
                )
                overall_status = "No pred" if any_observed else "Quiet"
            
            row['Overall'] = overall_status
            table_data.append(row)
    
    # Create DataFrame
    df = pd.DataFrame(table_data)
    
    # Reorder columns: AR, Model, then all tiles, then Overall
    cols = ['AR', 'Model'] + all_tile_cols + ['Overall']
    # Only include columns that exist in the dataframe
    cols = [c for c in cols if c in df.columns]
    df = df[cols]
    
    # Save
    csv_path = csv_dir / "all_ARs_emergence_timing_table.csv"
    df.to_csv(csv_path, index=False)
    
    print(f"  Saved combined emergence timing table to {csv_path}")
    return csv_path


def export_combined_metrics_to_csv(all_results, output_dir, model_configs):
    """Export combined metrics (mean ± std) across all ARs and tiles to CSV"""
    csv_dir = Path(output_dir) / 'unified_evaluations' / 'csv'
    csv_dir.mkdir(parents=True, exist_ok=True)
    
    # Get all model keys across all ARs
    all_model_keys = set()
    for all_tile_metrics in all_results.values():
        for tile_metrics in all_tile_metrics:
            all_model_keys.update([k for k in tile_metrics.keys() if k != 'observed'])
    
    # Get model display names
    model_names_map = {}
    for key in all_model_keys:
        if key in model_configs:
            model_names_map[key] = model_configs[key]['name']
        else:
            model_names_map[key] = key
    
    # Collect all metrics for each model across all ARs and tiles
    metrics_by_model = {}
    for model_key in all_model_keys:
        metrics_by_model[model_key] = {
            'MAE': [],
            'RMSE': [],
            'R2': [],
            'emerg_MAE': [],
            'emerg_RMSE': [],
            'emerg_R2': [],
            'emergence_lead_time': []
        }
    
    # Aggregate metrics from all ARs and tiles
    for all_tile_metrics in all_results.values():
        for tile_metrics in all_tile_metrics:
            for model_key in all_model_keys:
                if model_key in tile_metrics:
                    model_metrics = tile_metrics[model_key]
                    for metric_name in metrics_by_model[model_key].keys():
                        value = model_metrics.get(metric_name)
                        if value is not None:
                            metrics_by_model[model_key][metric_name].append(value)
    
    # Calculate mean ± std for each metric
    csv_data = []
    for model_key in sorted(all_model_keys):
        model_name = model_names_map[model_key]
        row = {'Model': model_name}
        
        for metric_name in ['MAE', 'RMSE', 'R2', 'emerg_MAE', 'emerg_RMSE', 'emerg_R2', 'emergence_lead_time']:
            values = metrics_by_model[model_key][metric_name]
            if len(values) > 0:
                mean_val = np.mean(values)
                std_val = np.std(values)
                row[metric_name] = f'{mean_val:.4f} ± {std_val:.4f}'
            else:
                row[metric_name] = 'N/A'
        
        csv_data.append(row)
    
    # Create DataFrame and save
    df = pd.DataFrame(csv_data)
    csv_path = csv_dir / "all_ARs_metrics.csv"
    df.to_csv(csv_path, index=False)
    
    print(f"  Saved combined metrics CSV to {csv_path}")
    return csv_path


def create_emergence_metrics_table(ax, all_metrics, model_keys, model_names_display):
    """Create emergence metrics table - metrics as rows, models as columns"""
    obs_metrics = all_metrics.get('observed', {})
    has_emergence_window = obs_metrics.get('emergence_window') is not None
    
    # Create shorter column names for better formatting
    short_names = []
    for name in model_names_display:
        if name == 'LSTM' or name == 'LSTM Baseline':
            short_names.append('LSTM')
        elif name == 'Baseline':
            short_names.append('Baseline')
        elif name == 'Baseline+Conv1D':
            short_names.append('Baseline+Conv1D')
        elif name == 'EarlyDetect':
            short_names.append('EarlyDetect')
        elif name == 'EarlyDetect+Conv1D':
            short_names.append('EarlyDetect+Conv1D')
        else:
            short_names.append(name)
    
    # Create header with model names (metrics as rows, models as columns)
    header = ['Metric'] + short_names
    data = [header]
    
    # Overall metrics (each metric is a row)
    data.append(['Overall MAE'] + [
        f'{all_metrics[m]["MAE"]:.4f}' if m in all_metrics and all_metrics[m].get("MAE") is not None else 'N/A'
        for m in model_keys
    ])
    data.append(['Overall RMSE'] + [
        f'{all_metrics[m]["RMSE"]:.4f}' if m in all_metrics and all_metrics[m].get("RMSE") is not None else 'N/A'
        for m in model_keys
    ])
    data.append(['Overall R2'] + [
        f'{all_metrics[m]["R2"]:.4f}' if m in all_metrics and all_metrics[m].get("R2") is not None else 'N/A'
        for m in model_keys
    ])
    
    # Window metrics
    if has_emergence_window:
        data.append(['Window MAE'] + [
            f'{all_metrics[m]["emerg_MAE"]:.4f}' if m in all_metrics and all_metrics[m].get("emerg_MAE") is not None else 'N/A'
            for m in model_keys
        ])
        data.append(['Window RMSE'] + [
            f'{all_metrics[m]["emerg_RMSE"]:.4f}' if m in all_metrics and all_metrics[m].get("emerg_RMSE") is not None else 'N/A'
            for m in model_keys
        ])
        data.append(['Window R2'] + [
            f'{all_metrics[m]["emerg_R2"]:.4f}' if m in all_metrics and all_metrics[m].get("emerg_R2") is not None else 'N/A'
            for m in model_keys
        ])
    
    # Lead time
    data.append(['T_lead (hrs)'] + [
        f'{all_metrics[m]["emergence_lead_time"]:+.0f}' if m in all_metrics and all_metrics[m].get("emergence_lead_time") is not None else 'N/A'
        for m in model_keys
    ])
    
    # Position table in upper right (where legend used to be)
    table_height = 0.8 if has_emergence_window else 0.6
    table_y_position = -0.6 if has_emergence_window else -0.4
    table_width = min(0.4 + len(model_keys) * 0.08, 0.7)
    
    table = ax.table(
        cellText=data,
        loc='upper left',
        bbox=[1.02, table_y_position, table_width, table_height],
        cellLoc='center',
        colLoc='center'
    )
    
    table.auto_set_font_size(False)
    table.set_fontsize(scaled_font(16))
    
    for (row, col), cell in table.get_celld().items():
        cell.set_text_props(color='black')
        cell.set_facecolor('white')
        cell.set_edgecolor('#CCCCCC')
        cell.set_linewidth(0.5)
        
        if row == 0:
            cell.set_text_props(weight='bold')
            cell.set_facecolor('#e0e0e0')
        elif row <= 3:
            if row % 2 == 0:
                cell.set_facecolor('#f9f9f9')
        elif has_emergence_window and row <= 6:
            if row % 2 == 1:
                cell.set_facecolor('#fff2cc')
            else:
                cell.set_facecolor('#ffe599')
        else:
            cell.set_facecolor('#d9ead3')
        
        if 'T_lead' in str(cell.get_text().get_text()) and col == 0:
            cell.set_text_props(fontsize=scaled_font(16))
        else:
            cell.set_text_props(fontsize=scaled_font(18))


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


AR_GRAPH_TIME_SPANS = {
    11698: {
        'start': datetime(2013, 3, 10, 1, 59, 37),
        'end': datetime(2013, 3, 17, 16, 59, 37),
        'length': 184
    },
    11726: {
        'start': datetime(2013, 4, 16, 21, 59, 37),
        'end': datetime(2013, 4, 25, 12, 59, 37),
        'length': 200
    },
    13165: {
        'start': datetime(2022, 12, 8, 22, 59, 37),
        'end': datetime(2022, 12, 16, 3, 59, 37),
        'length': 174
    },
    13179: {
        'start': datetime(2022, 12, 26, 20, 59, 37),
        'end': datetime(2023, 1, 3, 1, 59, 37),
        'length': 174
    },
    13183: {
        'start': datetime(2023, 1, 3, 3, 59, 37),
        'end': datetime(2023, 1, 10, 8, 59, 37),
        'length': 174
    }
}

FITS_FILE_CACHE = {}


def parse_fits_timestamp_from_name(fits_name):
    """Extract datetime from FITS filename (expects *_YYYY.MM.DD_HH-MM-SS.sss_* pattern)."""
    base = os.path.basename(fits_name)
    match = re.search(r'(\d{4}\.\d{2}\.\d{2})_(\d{2}-\d{2}-\d{2}\.\d+)', base)
    if not match:
        return None
    date_part, time_part = match.groups()
    try:
        return datetime.strptime(f"{date_part} {time_part}", "%Y.%m.%d %H-%M-%S.%f")
    except ValueError:
        return None


def list_fits_files_with_datetimes(zip_path):
    """Return sorted list of (datetime, filename) for FITS entries, cached per archive."""
    cache_key = str(zip_path)
    if cache_key in FITS_FILE_CACHE:
        return FITS_FILE_CACHE[cache_key]
    
    if not zip_path.exists():
        print(f"  ⚠️  Intensity map archive not found: {zip_path}")
        FITS_FILE_CACHE[cache_key] = []
        return FITS_FILE_CACHE[cache_key]
    
    entries = []
    with zipfile.ZipFile(zip_path) as z:
        for name in z.namelist():
            if not name.lower().endswith('.fits'):
                continue
            dt = parse_fits_timestamp_from_name(name)
            if dt is not None:
                entries.append((dt, name))
    
    entries.sort(key=lambda item: item[0])
    FITS_FILE_CACHE[cache_key] = entries
    return entries


def _load_from_json(test_AR, target_datetime, data_path):
    """Load intensity map from downloaded JSON files (full-resolution 513x513 data).
    For AR11726, uses the exact JSON files provided."""
    if test_AR != 11726:
        return None
    
    # Exact file paths for AR11726
    json_files = {
        datetime(2013, 4, 19, 23, 59, 37): ROOT / 'data' / 'AR11726_intensity_2013_04_19_23_59_37.json',
        datetime(2013, 4, 21, 23, 59, 37): ROOT / 'data' / 'AR11726_intensity_2013_04_21_23_59_37.json',
    }
    
    # Find closest JSON file to target datetime
    closest_json = None
    min_diff = float('inf')
    
    for json_datetime, json_path in json_files.items():
        diff = abs((target_datetime - json_datetime).total_seconds())
        if diff < min_diff:
            min_diff = diff
            closest_json = json_path
    
    if closest_json is None or not closest_json.exists():
        return None
    
    json_path = closest_json
    
    try:
        with open(json_path, 'r') as f:
            json_data = json.load(f)
        
        # Extract data array (513 rows, each row is an object with keys "0" through "512")
        data_array = json_data.get('data', [])
        if not data_array or len(data_array) != 513:
            print(f"  ⚠️  JSON file {json_path} has unexpected structure")
            return None
        
        # Convert to numpy array: 513 rows × 513 columns
        image = np.zeros((513, 513), dtype=np.float32)
        for i, row_obj in enumerate(data_array):
            if isinstance(row_obj, dict):
                for j in range(513):
                    key = str(j)
                    if key in row_obj:
                        image[i, j] = float(row_obj[key])
            elif isinstance(row_obj, list) and len(row_obj) == 513:
                # Handle case where row is already a list
                image[i, :] = np.array(row_obj, dtype=np.float32)
        
        # Crop to 512x512 if needed (remove last row/col)
        if image.shape == (513, 513):
            image = image[:512, :512]
        
        time_delta_hours = min_diff / 3600.0
        print(f"  ✓ Loaded intensity map from JSON {json_path.name} (Δ={time_delta_hours:.2f}h from target, shape={image.shape}, range=[{np.nanmin(image):.2f}, {np.nanmax(image):.2f}])")
        return image
    except Exception as e:
        print(f"  ⚠️  Could not load JSON file {json_path}: {e}")
        return None


def load_intensity_map_for_ar(test_AR, data_path, target_datetime):
    """Load the continuum intensity map from PNG files (noaa1.png or noaa2.png).
    Determines which PNG to load based on proximity to NOAA_first or NOAA_second."""
    if target_datetime is None:
        return None
    
    # Get NOAA dates for this AR to determine which PNG to load
    rid_of_top = 1
    _, _, _, NOAA_first, NOAA_second = get_ar_settings_fixed(test_AR, rid_of_top)
    
    # Determine which PNG file to load based on which NOAA date is closer
    diff_first = abs((target_datetime - NOAA_first).total_seconds())
    diff_second = abs((target_datetime - NOAA_second).total_seconds())
    
    if diff_first < diff_second:
        png_filename = 'noaa1.png'
        target_noaa = NOAA_first
    else:
        png_filename = 'noaa2.png'
        target_noaa = NOAA_second
    
    # Construct path to PNG file
    ar_dir = Path(data_path) / f'AR{test_AR}'
    png_path = ar_dir / png_filename
    
    if not png_path.exists():
        print(f"  ⚠️  PNG file not found: {png_path}")
        return None
    
    try:
        # Load PNG image
        img = Image.open(png_path)
        # Convert to numpy array (grayscale)
        image = np.array(img.convert('L'), dtype=np.float32)
        
        # If image is RGB, convert to grayscale
        if len(image.shape) == 3:
            # Convert RGB to grayscale using standard weights
            image = np.dot(image[...,:3], [0.2989, 0.5870, 0.1140]).astype(np.float32)
        
        # Normalize to 0-1 range if needed (PNG is typically 0-255)
        if image.max() > 1.0:
            image = image / 255.0
        
        time_delta_hours = abs((target_datetime - target_noaa).total_seconds()) / 3600.0
        print(f"  ✓ Loaded intensity map from {png_filename} (Δ={time_delta_hours:.2f}h from target, shape={image.shape}, range=[{np.nanmin(image):.2f}, {np.nanmax(image):.2f}])")
        return image
    except Exception as e:
        print(f"  ⚠️  Could not load PNG file {png_path}: {e}")
        return None


def _load_from_npz(npz_data, target_datetime, test_AR):
    """Helper function to load intensity map from npz data (tile-averaged, lower resolution)"""
    try:
        datetimes = npz_data['arr_1']  # Shape: (232,)
        intensity_data = npz_data['arr_0']  # Shape: (81, 232)
        
        # Find the closest datetime to target_datetime
        closest_idx = None
        min_time_diff = float('inf')
        for i, dt in enumerate(datetimes):
            if isinstance(dt, datetime):
                time_diff = abs((target_datetime - dt).total_seconds())
                if time_diff < min_time_diff:
                    min_time_diff = time_diff
                    closest_idx = i
        
        if closest_idx is not None:
            # Extract intensity values for all 81 tiles at this time step
            tile_intensities = intensity_data[:, closest_idx]  # Shape: (81,)
            closest_dt = datetimes[closest_idx]
            
            # Reshape to 9x9 grid (tiles are arranged in a 9x9 grid)
            image = tile_intensities.reshape(9, 9).astype(np.float32)
            
            # Upsample to a reasonable image size (e.g., 512x512) for visualization
            # Use numpy repeat to preserve tile structure (nearest neighbor)
            tile_size = 512 // 9  # ~56 pixels per tile
            image_upsampled = np.repeat(np.repeat(image, tile_size, axis=0), tile_size, axis=1)
            # Crop to exactly 512x512 if needed
            if image_upsampled.shape[0] > 512:
                image_upsampled = image_upsampled[:512, :512]
            if image_upsampled.shape[1] > 512:
                image_upsampled = image_upsampled[:, :512]
            
            time_delta_hours = min_time_diff / 3600.0
            print(f"  ✓ Loaded intensity map from npz file (tile-averaged, lower resolution)")
            print(f"    Target: {target_datetime.strftime('%Y-%m-%d %H:%M:%S')}, Using: {closest_dt.strftime('%Y-%m-%d %H:%M:%S')} (Δ={time_delta_hours:.2f}h)")
            print(f"    Shape: {image_upsampled.shape}, range=[{np.nanmin(image_upsampled):.2f}, {np.nanmax(image_upsampled):.2f}]")
            return image_upsampled
        else:
            print(f"  ⚠️  No valid datetimes found in npz file")
            return None
    except Exception as e:
        print(f"  ⚠️  Error loading from npz: {e}")
    return None


def highlight_tile_group(ax, tile_numbers, divisions=9, color='r', linewidth=3):
    """Draw a bounding rectangle that encapsulates the provided tile numbers."""
    if not tile_numbers:
        return
    
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    tile_width = (xlim[1] - xlim[0]) / divisions
    tile_height = (ylim[1] - ylim[0]) / divisions
    
    rows = [((tile - 1) // divisions) for tile in tile_numbers]
    cols = [((tile - 1) % divisions) for tile in tile_numbers]
    
    min_row, max_row = min(rows), max(rows)
    min_col, max_col = min(cols), max(cols)
    
    x = xlim[0] + min_col * tile_width
    width = (max_col - min_col + 1) * tile_width
    y = ylim[1] - (max_row + 1) * tile_height
    height = (max_row - min_row + 1) * tile_height
    
    from matplotlib.patches import Rectangle
    rect = Rectangle((x, y), width, height, linewidth=linewidth, edgecolor=color, facecolor='none')
    ax.add_patch(rect)


def load_and_preprocess_ar_eval_template(test_AR, data_path, rid_of_top, size):
    """Load and preprocess AR data with per-AR normalization"""
    # Try data_path location first, then fallback to playground root
    base1 = f'{data_path}/AR{test_AR}'
    base2 = f'{os.path.dirname(data_path) if os.path.dirname(data_path) else "."}/AR{test_AR}'
    
    # Check which location exists
    power_file = None
    mag_file = None
    cont_file = None
    
    for base in [base1, base2]:
        power_path = os.path.join(base, f'mean_pmdop{test_AR}_flat.npz')
        mag_path = os.path.join(base, f'mean_mag{test_AR}_flat.npz')
        cont_path = os.path.join(base, f'mean_int{test_AR}_flat.npz')
        
        if os.path.exists(power_path) and os.path.exists(mag_path) and os.path.exists(cont_path):
            power_file = power_path
            mag_file = mag_path
            cont_file = cont_path
            break
    
    if power_file is None:
        raise FileNotFoundError(f"Could not find AR{test_AR} data files. Checked: {base1} and {base2}")
    
    power = np.load(power_file, allow_pickle=True)
    mag   = np.load(mag_file, allow_pickle=True)
    cont  = np.load(cont_file, allow_pickle=True)

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


def load_model_config(config_path):
    """Load model configuration from an explicit JSON path."""
    config_path = Path(config_path).resolve()
    if not config_path.exists():
        print(f"    ⚠️  Config file not found: {config_path}")
        return None
        try:
        with open(config_path, "r") as f:
            data = json.load(f)
        cfg = data["best_config"] if "best_config" in data else data
        cfg = cfg.copy()
        cfg.setdefault("d_model", 256)
        cfg.setdefault("nhead", 8)
        cfg.setdefault("num_layers", 4)
        cfg.setdefault("dropout", 0.1)
        cfg.setdefault("output_len", 12)
        cfg.setdefault("use_temporal_conv", True)
        cfg.setdefault("timing_bias_weight", 0.1)
        cfg.setdefault("early_detection_weight", 0.2)
        print(f"    Loaded config from: {config_path}")
        return cfg
        except Exception as e:
        print(f"    ⚠️  Warning: could not load config from {config_path}: {e}")
        return None


def evaluate_all_models_on_ar(
    test_AR,
    model_configs,
    data_path,
    output_dir,
    device
):
    """Evaluate all models on a single AR with overlayed predictions
    
    Args:
        test_AR: AR number to evaluate
        model_configs: Dictionary of model configurations (from main())
        data_path: Path to data directory
        output_dir: Output directory for results
        device: PyTorch device
    """
    
    # Clear GPU cache at start
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    
    print(f"\n{'='*100}")
    print(f"Evaluating AR {test_AR} with all models")
    print(f"{'='*100}")
    
    rid_of_top = 1
    size = 9
    start_tile, before_plot, num_in, NOAA_first, NOAA_second = get_ar_settings_fixed(test_AR, rid_of_top)
    ar_graph_times = AR_GRAPH_TIME_SPANS.get(test_AR)
    graph_start_dt = ar_graph_times['start'] if ar_graph_times else NOAA_first
    graph_end_dt = ar_graph_times['end'] if ar_graph_times else NOAA_second
    # Map data tile indices (0-indexed in 63-tile array) to FITS tile numbers (1-indexed in 81-tile grid)
    # The starting_tile values (46, 28, 37) are in the ORIGINAL 81-tile grid (before removing top row)
    # When rid_of_top=1, we remove the first row (9 tiles), so:
    #   - Original tile 46 → data tile 46-9=37 → FITS tile 46+1=47 (1-indexed)
    #   - Original tile 28 → data tile 28-9=19 → FITS tile 28+1=29 (1-indexed)
    #   - Original tile 37 → data tile 37-9=28 → FITS tile 37+1=38 (1-indexed)
    # So the FITS tile number = original_tile + 1 (convert 0-indexed to 1-indexed)
    # But starting_tile is already the data tile index, so we need to add back the removed row
    original_starting_tile = start_tile + rid_of_top * 9  # Reconstruct original tile number
    tiles_to_highlight = [original_starting_tile + offset + 1 for offset in range(7)]  # +1 for 1-indexed
    
    # Calculate which row/col these tiles are in the 9x9 grid for verification
    def tile_to_row_col(tile_1indexed):
        """Convert 1-indexed tile number to (row, col) in 9x9 grid (0-indexed)"""
        tile_0indexed = tile_1indexed - 1
        row = tile_0indexed // 9
        col = tile_0indexed % 9
        return row, col
    
    print(f"  Data tile range: {start_tile} to {start_tile + 6} (0-indexed in 63-tile array)")
    print(f"  Original starting tile (81-tile grid, 0-indexed): {original_starting_tile}")
    print(f"  Original starting tile (81-tile grid, 1-indexed): {original_starting_tile + 1}")
    print(f"  Highlighting FITS tiles (1-indexed): {tiles_to_highlight}")
    print(f"  Tile positions in 9x9 grid (row, col are 0-indexed):")
    for i, tile in enumerate(tiles_to_highlight):
        row, col = tile_to_row_col(tile)
        print(f"    Tile {tile} (data index {start_tile + i}): row {row}, col {col}")
    
    # Verify: if the mapping is correct, these should match what's actually in the FITS images
    # AR13179/13183 work correctly, so their mapping is the reference
    # If AR11698/13165 are off by 2 rows, we need to investigate why
    
    NOAA1 = mdates.date2num(NOAA_first)
    NOAA2 = mdates.date2num(NOAA_second)
    
    # Load data
    inputs, ii, time_arr, norm_stats, ii_raw_full = load_and_preprocess_ar_eval_template(test_AR, data_path, rid_of_top, size)
    mp, Mp, mm, Mm, mi, Mi = norm_stats
    
    # Load all models
    models = {}
    lstm_fut = None
    
    # Load LSTM (Experiment A)
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
                'color': lstm_config.get('color', '#0066CC'),
                'linestyle': lstm_config.get('linestyle', '-'),
                'fut_idx': lstm_num_pred - 1,
                'config': {'output_len': lstm_num_pred},
                'show_in_derivatives': lstm_config.get('show_in_derivatives', True)
            }
            lstm_fut = lstm_num_pred - 1
            print(f"  Loaded LSTM (Experiment A)")
    
    # Load Transformer models
    for exp_name, exp_config in model_configs.items():
        if exp_name == 'lstm':
            continue
        
        try:
            model_path = exp_config['path']
            config_path = exp_config.get('config_path')
            config = load_model_config(config_path) if config_path else None
            
            # Use config overrides from model_configs if provided
            if 'config' in exp_config:
                if config is None:
                    config = exp_config['config'].copy()
                else:
                config.update(exp_config['config'])
            
            if config is None:
                raise ValueError(f"Config not found for {exp_name}. Provide config_path or config overrides.")
            
            # Print full config for debugging
            print(f"    Loading {exp_config['name']}")
            print(f"    Config from JSON: d_model={config.get('d_model')}, nhead={config.get('nhead')}, num_layers={config.get('num_layers')}, use_temporal_conv={config.get('use_temporal_conv')}, dropout={config.get('dropout')}")
            
            # Verify config values are valid
            for required_key in ["d_model", "nhead", "num_layers", "output_len"]:
                if config.get(required_key) is None:
                    raise ValueError(f"Config missing {required_key} for {exp_name}")
            
            if exp_config.get('type') == 'baseline':
                transformer = BaselineTransformer(
                    input_dim=inputs.shape[1],
                    d_model=config['d_model'],
                    nhead=config['nhead'],
                    num_layers=config['num_layers'],
                    dropout=config.get('dropout', 0.0),
                    output_len=config['output_len'],
                    max_seq_len=150,
                    use_temporal_conv=config.get('use_temporal_conv', True),
                ).to(device)
            else:
                transformer = EarlyDetectTransformer(
                    input_dim=inputs.shape[1],
                    d_model=config['d_model'],
                    nhead=config['nhead'],
                    num_layers=config['num_layers'],
                    dropout=config.get('dropout', 0.0),
                    output_len=config['output_len'],
                    max_seq_len=150,
                    use_temporal_conv=config.get('use_temporal_conv', True),
                    timing_bias_weight=config.get('timing_bias_weight', 0.1),
                    early_detection_weight=config.get('early_detection_weight', 0.2),
                ).to(device)
            
            # Load state dict - models are saved as SARTransformerLocalTile.state_dict()
            # so keys will have "transformer." prefix (for the internal EmergencePatternTransformer)
            state_dict = torch.load(model_path, map_location=device)
            
            # Try loading with strict matching first
            try:
                transformer.load_state_dict(state_dict, strict=True)
                print(f"    ✅ Loaded state dict successfully")
            except RuntimeError as e:
                error_msg = str(e)
                # If there are size mismatches, it means the architecture doesn't match
                # Check if it's just unexpected/missing keys or actual size mismatches
                if "size mismatch" in error_msg:
                    print(f"    ❌ Architecture mismatch! Config may be incorrect.")
                    print(f"    Error: {error_msg[:300]}...")
                    raise RuntimeError(f"Model architecture doesn't match checkpoint. Check config values.")
                else:
                    # Try with strict=False for unexpected/missing keys
                    print(f"    ⚠️  Strict loading failed (unexpected keys), trying lenient loading...")
                    try:
                        transformer.load_state_dict(state_dict, strict=False)
                        print(f"    ⚠️  Loaded with strict=False (some keys may be missing/mismatched)")
                    except Exception as e2:
                        print(f"    ❌ Failed to load state dict: {e2}")
                        raise
            
            transformer.eval()
            
            models[exp_name] = {
                'model': transformer,
                'name': exp_config['name'],
                'color': exp_config.get('color', 'red'),
                'linestyle': exp_config.get('linestyle', '-'),
                'fut_idx': config.get('output_len', 12) - 1,
                'config': config,
                'show_in_derivatives': exp_config.get('show_in_derivatives', False)
            }
            print(f"  ✅ Loaded {exp_config['name']}")
        except Exception as e:
            print(f"  ❌ Warning: Could not load {exp_name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Create evaluation plots with improved visuals (matching redo_experiment_f_eval.py)
    # Layout: 2 columns x 4 rows grid
    fig = plt.figure(figsize=(32, 46))  # Wider figure to accommodate 2 columns
    fig.subplots_adjust(left=0.08, right=0.95, top=0.97, bottom=0.1)
    gs0 = gridspec.GridSpec(4, 2, figure=fig, hspace=0.12, wspace=0.23)
    
    thr = -0.01
    st = 4
    all_tile_metrics = []
    
    # Store derivative axes and ranges for AR-wide range calculation
    derivative_axes_ranges = []
    
    for i in range(7):
        tile_idx = start_tile + i
        disp = tile_idx + 10
        print(f"    Processing Tile {disp}")
        
        # Calculate grid position: 2 columns x 4 rows
        # i=0: row=0, col=0; i=1: row=0, col=1
        # i=2: row=1, col=0; i=3: row=1, col=1
        # i=4: row=2, col=0; i=5: row=2, col=1
        # i=6: row=3, col=0
        row = i // 2
        col = i % 2
        grid_pos = (row, col)
        
        # Prepare test data
        X_test, y_test = lstm_ready_eval_template(tile_idx, size, inputs, ii, num_in, 12, model_seq_len=num_in)
        X_test = X_test.to(device)
        Xt = X_test.view(X_test.size(0), num_in, X_test.size(2))
        
        # Get predictions from all models
        predictions = {}
        
        with torch.no_grad():
            for model_name, model_info in models.items():
                if model_name == 'lstm':
                    pred = model_info['model'](X_test)[:, model_info['fut_idx']].cpu().numpy()
                else:
                    pred = model_info['model'](Xt)[:, model_info['fut_idx']].cpu().numpy()
                
                predictions[model_name] = pred
        
        true = y_test[:, lstm_fut if lstm_fut is not None else 11].numpy()
        
        # Denormalize and recalibrate
        last = ii.shape[1] - true.shape[0] - 1
        recal_point_raw = ii_raw_full[tile_idx, last]
        
        predictions_raw = {}
        for model_name, pred in predictions.items():
            # Denormalize: pred is normalized [0,1], convert back to raw
            if Mi != mi:
                pred_raw = pred * (Mi - mi) + mi
            else:
                pred_raw = pred * 0 + mi
            pred_raw = recalibrate(pred_raw, recal_point_raw)
            predictions_raw[model_name] = pred_raw
        
        # Denormalize true values
        if Mi != mi:
            true_raw = true * (Mi - mi) + mi
        else:
            true_raw = true * 0 + mi
        
        # Calculate metrics for all models
        all_model_metrics = {}
        obs_window_info = None
        
        # CRITICAL: Recalibrate normalized data for timing calculations (matching experiment_f_hyperparam_search.py)
        # This ensures predictions are anchored to the last observed value for fair timing comparison
        recal_point_norm = ii[tile_idx, last]
        true_calibrated = recalibrate(true, recal_point_norm)
        predictions_calibrated = {}
        for model_name, pred_norm in predictions.items():
            # Flatten if needed (pred_norm is already 1D from model output)
            pred_1d = pred_norm.flatten() if pred_norm.ndim > 1 else pred_norm
            predictions_calibrated[model_name] = recalibrate(pred_1d, recal_point_norm)
        
        # Calculate observed emergence window using recalibrated normalized data
        # Use np.gradient() to match experiment_f_hyperparam_search.py
        d_obs_norm_for_timing = np.gradient(smooth_with_numpy(true_calibrated))
        obs_start, obs_end, _ = select_onset_from_windows(find_emergence_windows(d_obs_norm_for_timing, thr, st))
        if obs_start is not None and obs_end is not None:
            obs_window_info = (obs_start, obs_end)
        
        # Calculate metrics for each model
        for model_name in models.keys():
            pred_norm = predictions[model_name]
            pred_calibrated = predictions_calibrated[model_name]
            pred_raw = predictions_raw[model_name]
            
            # Timing metrics using recalibrated normalized data (matching experiment_f_hyperparam_search.py)
            # Use np.gradient() to match experiment_f_hyperparam_search.py
            d_pred_norm = np.gradient(pred_calibrated)
            pred_start, pred_end, _ = select_onset_from_windows(find_emergence_windows(d_pred_norm, thr, st))
            
            lead_time = None
            if obs_start is not None and pred_start is not None:
                lead_time = (obs_start - pred_start) + 12
            
            # Accuracy metrics using denormalized data
            def calc_basic_metrics(y_true, y_pred):
                mae = np.mean(np.abs(y_true - y_pred))
                mse = np.mean((y_true - y_pred) ** 2)
                rmse = np.sqrt(mse)
                r2 = 1 - np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2)
                return mae, rmse, r2
            
            mae, rmse, r2 = calc_basic_metrics(true_raw, pred_raw)
            
            emerg_mae, emerg_rmse, emerg_r2 = None, None, None
            if obs_window_info:
                obs_start_win, obs_end_win = obs_window_info
                window_true = true_raw[obs_start_win:obs_end_win]
                window_pred = pred_raw[obs_start_win:obs_end_win]
                if len(window_true) > 0:
                    emerg_mae, emerg_rmse, emerg_r2 = calc_basic_metrics(window_true, window_pred)
            
            # Store emergence window for the model to detect ILAP cases
            pred_window = (pred_start, pred_end) if pred_start is not None else None
            
            all_model_metrics[model_name] = {
                'MAE': mae,
                'RMSE': rmse,
                'R2': r2,
                'emerg_MAE': emerg_mae,
                'emerg_RMSE': emerg_rmse,
                'emerg_R2': emerg_r2,
                'emergence_lead_time': lead_time,
                'emergence_window': pred_window  # Store model's predicted emergence window
            }
        
        all_model_metrics['observed'] = {'emergence_window': obs_window_info}
        all_tile_metrics.append(all_model_metrics)
        
        # Get raw "before" data
        before_raw = ii_raw_full[tile_idx, last - before_plot:last]
        tcut = time_arr[last - before_plot:last + true.shape[0]]
        tnum = mdates.date2num(tcut)
        nanarr = np.full(before_raw.shape, np.nan)
        
        # Calculate derivatives on RECALIBRATED NORMALIZED data for timing detection
        # (matching experiment_f_hyperparam_search.py methodology)
        before_norm = ii[tile_idx, last - before_plot:last]
        # Use recalibrated true for consistency with timing calculations
        d_obs_norm = np.gradient(smooth_with_numpy(np.concatenate((before_norm, true_calibrated))))
        d_predictions_norm = {}
        for model_name, pred_calibrated in predictions_calibrated.items():
            d_predictions_norm[model_name] = np.gradient(pred_calibrated)
        
        # Calculate derivatives on DENORMALIZED data for visualization
        d_obs_raw = safe_gradient(smooth_with_numpy(np.concatenate((before_raw, true_raw))))
        d_predictions_raw = {}
        for model_name, pred_raw in predictions_raw.items():
            d_predictions_raw[model_name] = safe_gradient(pred_raw)
        
        # Recalculate emergence window from FULL array (for correct visualization alignment)
        # This ensures the window indices match the full concatenated array used in plots
        obs_window_full_start, obs_window_full_end, _ = select_onset_from_windows(
            find_emergence_windows(d_obs_norm, thr, st), prediction_start_idx=before_plot
        )
        obs_window_full = (obs_window_full_start, obs_window_full_end) if obs_window_full_start is not None else None

        # Also get the window from metrics (relative to prediction portion only) for metrics calculation
        obs_window = all_model_metrics.get('observed', {}).get('emergence_window')

        t_start = None
        t_end = None
        if obs_window_full and obs_window_full[0] is not None and obs_window_full[1] is not None:
            # Use the window calculated from full array - indices are already correct for visualization
            start_idx = obs_window_full[0]
            end_idx = obs_window_full[1] - 1  # end_idx is exclusive
            if start_idx < len(tnum) and end_idx < len(tnum):
                t_start = tnum[start_idx]
                t_end = tnum[end_idx]
        
        # Calculate emergence indication using NORMALIZED data (for accurate timing)
        ind_o_norm = emergence_indication(d_obs_norm, thr, st)
        ind_predictions_norm = {}
        for model_name, d_pred_norm in d_predictions_norm.items():
            ind_predictions_norm[model_name] = emergence_indication(d_pred_norm, thr, st)
        
        # Create plots - 5 subplots: intensity, observed derivative, LSTM derivative, Experiment C derivative, error
        # Use grid position for 2x4 layout
        gs1 = gridspec.GridSpecFromSubplotSpec(5, 1, subplot_spec=gs0[grid_pos], height_ratios=[18, 3, 3, 3, 4], hspace=0.3)
        
        # Main intensity plot
        ax0 = fig.add_subplot(gs1[0])
        ax0.plot(tnum, np.concatenate((before_raw, true_raw)), 'k-', linewidth=2.5, label='Observed Intensity')
        
        # Plot all model predictions
        for model_name, model_info in models.items():
            if model_name in predictions_raw:
                pred_full = np.concatenate((nanarr, predictions_raw[model_name]))
                ax0.plot(tnum, pred_full, 
                        color=model_info['color'], 
                        linestyle=model_info['linestyle'],
                        linewidth=2.5,
                        label=model_info['name'])
        
        ax0.axvline(NOAA1, color='magenta', linestyle='--', linewidth=2, label='NOAA First Record')
        ax0.axvline(NOAA2, color='darkmagenta', linestyle='--', linewidth=2, label='NOAA Second Record')
        
        if obs_window_full and t_start is not None and t_end is not None:
            ax0.axvspan(t_start, t_end, color='yellow', alpha=0.3, label='Emergence Window')
        
        ax0.set_title(f'Tile {disp} - AR {test_AR}', fontsize=scaled_font(28))
        ax0.set_ylabel('Continuum Intensity', fontsize=scaled_font(24), labelpad=30)
        
        # Use consistent AR-wide range
        ar_min, ar_max = np.nanmin(ii_raw_full), np.nanmax(ii_raw_full)
        pad = max(0.05 * (ar_max - ar_min), 0.05)
        ax0.set_ylim([ar_min - pad, ar_max + pad])
        ax0.grid(True)
        ax0.yaxis.set_major_locator(MaxNLocator(nbins=8))
        ax0.tick_params(labelsize=scaled_font(18), labelbottom=False)
        
        # Create legend in bottom left (smaller)
        # Determine if this tile needs a smaller legend (legend hiding plot)
        # Display tile number = start_tile + i + 10
        display_tile = start_tile + i + 10
        needs_smaller_legend = False
        
        if test_AR == 13183 and display_tile in [41, 42]:
            needs_smaller_legend = True
        elif test_AR == 11726 and display_tile in [41, 42, 43]:
            needs_smaller_legend = True
        elif test_AR == 13165 and display_tile == 32:
            needs_smaller_legend = True
        elif test_AR == 13179 and display_tile in [41, 42]:
            needs_smaller_legend = True
        
        legend_kwargs = dict(
            bbox_to_anchor=(0.02, 0.02),
            loc='lower left',
            fontsize=scaled_font(16),
            framealpha=0.9,
            ncol=2
        )
        if i == 0:
            legend_kwargs.update(
                ncol=4,
                columnspacing=0.6,
                handlelength=1.4,
                handletextpad=0.4,
                borderpad=0.3,
                labelspacing=0.4
            )
        elif needs_smaller_legend:
            # Smaller legend for tiles where it hides the plot
            legend_kwargs.update(
                fontsize=scaled_font(12),  # Smaller font
                handlelength=1.0,  # Shorter handles
                handletextpad=0.3,  # Less padding
                borderpad=0.2,  # Less border padding
                labelspacing=0.3,  # Less spacing between labels
                columnspacing=0.5  # Less spacing between columns
            )
        legend = ax0.legend(**legend_kwargs)
        legend.get_frame().set_boxstyle('square', pad=0.5)
        
        # Derivative plots - display denormalized values but use normalized timing
        # Observed derivative
        ax1 = fig.add_subplot(gs1[1], sharex=ax0)
        ax1.plot(tnum, d_obs_raw, color='black', linewidth=2.5, label='_nolegend_')
        
        if obs_window_full and t_start is not None and t_end is not None:
            ax1.axvspan(t_start, t_end, color='yellow', alpha=0.3)
        
        # Use normalized timing indices for highlighting
        # d_obs_norm and d_obs_raw should have same length as tnum
        has_obs_emergence = False
        if len(d_obs_norm) == len(d_obs_raw) == len(tnum):
            for j in range(len(d_obs_norm) - 1):
                if ind_o_norm[j] != 0:
                    if j + 1 < len(tnum):
                        ax1.plot(tnum[j:j+2], d_obs_raw[j:j+2], color='#00CC66', linewidth=3, label='Predicted' if not has_obs_emergence else '')
                        has_obs_emergence = True
        ax1.set_ylabel(r'$\frac{dObs}{dt}$', fontsize=scaled_font(22), labelpad=30)
        ax1.tick_params(labelsize=scaled_font(18), labelbottom=False)
        
        # Calculate range for observed derivative (will be set after all three are calculated)
        d_obs_finite = d_obs_raw[np.isfinite(d_obs_raw)]
        if len(d_obs_finite) > 0:
            max_range_obs = max(abs(np.min(d_obs_finite)), abs(np.max(d_obs_finite)))
            y_lim_obs = max(max_range_obs * 1.1, 0.01)  # Add 10% padding, minimum 0.01
        else:
            y_lim_obs = 0.05
        
        ax1.grid(True)
        
        # LSTM derivative (separate subplot)
        ax2 = fig.add_subplot(gs1[2], sharex=ax0)
        has_lstm_emergence = False
        if 'lstm' in d_predictions_raw and 'lstm' in models:
            lstm_info = models['lstm']
            d_lstm_raw = d_predictions_raw['lstm']
            d_full = np.concatenate([np.full(before_plot, np.nan), d_lstm_raw])
            ax2.plot(tnum, d_full, 
                    color=lstm_info['color'], 
                    linestyle=lstm_info['linestyle'],
                    linewidth=2.5,
                    label='_nolegend_')
            
            # Use normalized timing indices for highlighting
            if 'lstm' in ind_predictions_norm:
                ind_pred_norm = ind_predictions_norm['lstm']
                for j in range(len(ind_pred_norm) - 1):
                    if ind_pred_norm[j] != 0:
                        denorm_idx = j + before_plot
                        if denorm_idx + 1 < len(d_full) and denorm_idx + 1 < len(tnum):
                            ax2.plot(tnum[denorm_idx:denorm_idx+2], d_full[denorm_idx:denorm_idx+2], color='#00CC66', linewidth=3, label='Predicted' if not has_lstm_emergence else '')
                            has_lstm_emergence = True
            
            # Calculate range for LSTM derivative (will be set after all three are calculated)
            d_lstm_finite = d_lstm_raw[np.isfinite(d_lstm_raw)]
            if len(d_lstm_finite) > 0:
                max_range_lstm = max(abs(np.min(d_lstm_finite)), abs(np.max(d_lstm_finite)))
                y_lim_lstm = max(max_range_lstm * 1.1, 0.01)  # Add 10% padding, minimum 0.01
            else:
                y_lim_lstm = 0.05
        else:
            y_lim_lstm = 0.05
        
        if obs_window_full and t_start is not None and t_end is not None:
            ax2.axvspan(t_start, t_end, color='yellow', alpha=0.3)
        ax2.set_ylabel(r'$\frac{dLSTM}{dt}$', fontsize=scaled_font(22), labelpad=30)
        ax2.tick_params(labelsize=scaled_font(18), labelbottom=False)
        ax2.grid(True)
        ax2.set_xlim(tnum[0], tnum[-1])
        if 'lstm' in d_predictions_raw and 'lstm' in models:
            ax2.legend(loc='upper left', fontsize=scaled_font(14), framealpha=0.9)
        
        # Experiment C derivative (separate subplot)
        ax3 = fig.add_subplot(gs1[3], sharex=ax0)
        has_emergence_expc = False
        if 'exp_d' in d_predictions_raw and 'exp_d' in models:
            exp_c_info = models['exp_d']
            d_expc_raw = d_predictions_raw['exp_d']
            d_full = np.concatenate([np.full(before_plot, np.nan), d_expc_raw])
            ax3.plot(tnum, d_full, 
                    color=exp_c_info['color'], 
                    linestyle=exp_c_info['linestyle'],
                    linewidth=2.5,
                    label='_nolegend_')
            
            # Use normalized timing indices for highlighting
            if 'exp_d' in ind_predictions_norm:
                ind_pred_norm = ind_predictions_norm['exp_d']
                for j in range(len(ind_pred_norm) - 1):
                    if ind_pred_norm[j] != 0:
                        denorm_idx = j + before_plot
                        if denorm_idx + 1 < len(d_full) and denorm_idx + 1 < len(tnum):
                            ax3.plot(tnum[denorm_idx:denorm_idx+2], d_full[denorm_idx:denorm_idx+2], color='#00CC66', linewidth=3, label='Predicted' if not has_emergence_expc else '')
                            has_emergence_expc = True
            
            # Calculate range for Experiment C derivative (will be set after all three are calculated)
            d_expc_finite = d_expc_raw[np.isfinite(d_expc_raw)]
            if len(d_expc_finite) > 0:
                max_range_expc = max(abs(np.min(d_expc_finite)), abs(np.max(d_expc_finite)))
                y_lim_expc = max(max_range_expc * 1.1, 0.01)  # Add 10% padding, minimum 0.01
            else:
                y_lim_expc = 0.05
        else:
            y_lim_expc = 0.05
        
        # Find the maximum range among all three derivatives for this tile
        y_lim_max_tile = max(y_lim_obs, y_lim_lstm, y_lim_expc)
        
        # Store axes and range for AR-wide update later
        derivative_axes_ranges.append((ax1, ax2, ax3, y_lim_max_tile))
        
        # Set temporary ranges (will be updated to AR-wide max after all tiles are processed)
        ax1.set_ylim([-y_lim_max_tile, y_lim_max_tile])
        ax1.set_yticks([-y_lim_max_tile, 0, y_lim_max_tile])
        ax2.set_ylim([-y_lim_max_tile, y_lim_max_tile])
        ax2.set_yticks([-y_lim_max_tile, 0, y_lim_max_tile])
        ax3.set_ylim([-y_lim_max_tile, y_lim_max_tile])
        ax3.set_yticks([-y_lim_max_tile, 0, y_lim_max_tile])
        
        if obs_window_full and t_start is not None and t_end is not None:
            ax3.axvspan(t_start, t_end, color='yellow', alpha=0.3)
        ax3.set_ylabel(r'$\frac{dEarlyDetect}{dt}$', fontsize=scaled_font(22), labelpad=30)
        ax3.tick_params(labelsize=scaled_font(18), labelbottom=False)
        ax3.grid(True)
        ax3.set_xlim(tnum[0], tnum[-1])

        # Emergence (green) may appear on any derivative; one legend on observed only.
        if has_obs_emergence or has_lstm_emergence or has_emergence_expc:
            ax1.plot([], [], color="#00CC66", linewidth=3, label="Emergence")
            ax1.legend(
                loc="upper left",
                fontsize=scaled_font(14),
                framealpha=0.9,
            )

        # Error analysis (with date labels)
        ax4 = fig.add_subplot(gs1[4], sharex=ax0)
        for model_name, model_info in models.items():
            if model_name in predictions_raw:
                errors = np.abs(true_raw - predictions_raw[model_name])
                ax4.plot(tnum[before_plot:before_plot+len(errors)], errors,
                        color=model_info['color'],
                        linestyle=model_info['linestyle'],
                        linewidth=2.5,
                        label=model_info['name'])
        ax4.axvline(NOAA1, color='magenta', linestyle='--', linewidth=2)
        if obs_window_full and t_start is not None and t_end is not None:
            ax4.axvspan(t_start, t_end, color='yellow', alpha=0.3)
        ax4.set_ylabel('|Error|', fontsize=scaled_font(24), labelpad=30)
        ax4.set_xlabel('Date', fontsize=scaled_font(24))
        ax4.set_xlim(tnum[0], tnum[-1])
        ax4.xaxis.set_major_locator(mdates.DayLocator())
        ax4.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax4.tick_params(labelsize=scaled_font(18), labelbottom=True)
        ax4.grid(True)
    
    # Apply AR-wide maximum range to all derivative plots
    if len(derivative_axes_ranges) > 0:
        # Find the maximum range across all tiles
        ar_wide_max = max(y_lim_max for _, _, _, y_lim_max in derivative_axes_ranges)
        
        # Apply AR-wide maximum to all derivative plots
        for ax1, ax2, ax3, _ in derivative_axes_ranges:
            ax1.set_ylim([-ar_wide_max, ar_wide_max])
            ax1.set_yticks([-ar_wide_max, 0, ar_wide_max])
            ax2.set_ylim([-ar_wide_max, ar_wide_max])
            ax2.set_yticks([-ar_wide_max, 0, ar_wide_max])
            ax3.set_ylim([-ar_wide_max, ar_wide_max])
            ax3.set_yticks([-ar_wide_max, 0, ar_wide_max])
    
    # Add two 9x9 grid images in the bottom right (position 3,1)
    # Create a nested grid for the two images - make them bigger
    gs_grid = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs0[3, 1], wspace=0.15, width_ratios=[1, 1])
    
    # Load the actual FITS frames at NOAA first and second record times
    # These correspond to the official NOAA detection times shown as vertical lines in the plots
    noaa_first_map = load_intensity_map_for_ar(test_AR, data_path, NOAA_first)
    noaa_second_map = load_intensity_map_for_ar(test_AR, data_path, NOAA_second)
    
    # Use placeholder if maps not available
    placeholder_size = 512
    if noaa_first_map is None:
        noaa_first_map = np.ones((placeholder_size, placeholder_size), dtype=np.float32) * 0.5
    if noaa_second_map is None:
        noaa_second_map = np.ones((placeholder_size, placeholder_size), dtype=np.float32) * 0.5
    
    def format_time_label(dt):
        return dt.strftime('%Y-%m-%d %H:%M') if dt is not None else 'N/A'
    
    def render_intensity_panel(ax, image, panel_title, timestamp_label, test_AR):
        # Flip image upside down for AR11698 and AR13165
        if test_AR in [11698, 13165]:
            image = np.flipud(image)
        
        finite = np.isfinite(image)
        if finite.any():
            finite_values = image[finite]
            # Use a wider percentile range to avoid black images
            vmin, vmax = np.percentile(finite_values, [1, 99.5])
            # If the range is too small, use min/max instead
            if vmax - vmin < 1e-6:
                vmin, vmax = np.nanmin(finite_values), np.nanmax(finite_values)
                if vmax - vmin < 1e-6:
                    # If still too small, use a symmetric range around the mean
                    mean_val = np.nanmean(finite_values)
                    vmin, vmax = mean_val - 0.1, mean_val + 0.1
        else:
            vmin = vmax = None
        ax.imshow(image, cmap='gray', origin='lower', vmin=vmin, vmax=vmax)
        add_grid_lines(ax, divisions=9, color='w', linewidth=1.5)
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        tile_width = (xlim[1] - xlim[0]) / 9
        tile_height = (ylim[1] - ylim[0]) / 9
        for tile_num in tiles_to_highlight:
            highlight_tile(ax, tile_num, divisions=9, color='r', linewidth=2.5)
            tile_zero_indexed = tile_num - 1
            tile_row = tile_zero_indexed // 9
            tile_col = tile_zero_indexed % 9
            tile_x_center = xlim[0] + (tile_col + 0.5) * tile_width
            tile_bottom = ylim[1] - (tile_row + 1) * tile_height
            ax.text(
                tile_x_center,
                tile_bottom - 0.12 * tile_height,
                f'{tile_num}',
                color='red',
                fontsize=scaled_font(14),
                ha='center',
                va='top',
                fontweight='bold',
                clip_on=False
            )
        highlight_tile_group(ax, tiles_to_highlight, divisions=9, color='r', linewidth=4)
        ax.set_title(panel_title, fontsize=scaled_font(22))
        ax.set_xlabel(timestamp_label, fontsize=scaled_font(20), labelpad=12)
        ax.xaxis.set_label_position('bottom')
        ax.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False,
                       labelleft=False, labelbottom=True)
        ax.set_xticks([])
        ax.set_yticks([])
    
    # First image (NOAA First Record)
    ax_image1 = fig.add_subplot(gs_grid[0, 0])
    render_intensity_panel(
        ax_image1,
        noaa_first_map,
        f'Continuum Intensity (NOAA First Record)\nAR {test_AR}',
        format_time_label(NOAA_first),
        test_AR
    )
    
    # Second image (NOAA Second Record)
    ax_image2 = fig.add_subplot(gs_grid[0, 1])
    render_intensity_panel(
        ax_image2,
        noaa_second_map,
        f'Continuum Intensity (NOAA Second Record)\nAR {test_AR}',
        format_time_label(NOAA_second),
        test_AR
    )
    
    plt.tight_layout(rect=[0, 0, 1.0, 0.96])
    plt.subplots_adjust(right=1.0)
    
    # Save as PDF
    plot_dir = Path(output_dir) / 'unified_evaluations' / 'pdf'
    plot_dir.mkdir(parents=True, exist_ok=True)
    out_path = plot_dir / f"AR{test_AR}_all_models_comparison.pdf"
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved plot to {out_path}")
    
    # Export metrics to CSV (per AR)
    export_metrics_to_csv(all_tile_metrics, test_AR, output_dir, model_configs)
    
    # Export emergence timing table (per AR)
    export_emergence_timing_table(all_tile_metrics, test_AR, start_tile, output_dir, model_configs)
    
    # Clean up models and GPU memory
    for model_info in models.values():
        if 'model' in model_info:
            del model_info['model']
    
    # Aggressive GPU cleanup
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        import gc
        gc.collect()
        torch.cuda.empty_cache()
    
    return all_tile_metrics


def main():
    parser = argparse.ArgumentParser(description='Unified Evaluation of All Experiments')
    parser.add_argument('--data_path', type=str,
                       default=str(ROOT / 'data'),
                       help='Path to data directory')
    parser.add_argument('--output_dir', type=str,
                       default=str(ROOT / 'results'),
                       help='Output directory')
    parser.add_argument('--test_ars', type=int, nargs='+',
                       default=[11698, 11726, 13165, 13179, 13183],
                       help='Test ARs to evaluate')
    
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Define model configurations
    # LSTM and Experiment C: strong standout colors with solid lines
    # Others: color-blind-friendly colors with dotted lines
    model_configs = {
        'lstm': {
            'path': str(ROOT / 'models' / 'checkpoints' / 'lstm_baseline.pth'),
            'name': 'LSTM Baseline',
            'color': '#6600CC',  # Strong purple
            'linestyle': '-',
            'show_in_derivatives': True  # Show in derivative plots
        },
        'exp_b_no_conv1d': {
            'path': str(ROOT / 'models' / 'checkpoints' / 'baseline_no_conv1d.pth'),
            'config_path': str(ROOT / 'models' / 'checkpoints' / 'configs' / 'best_config_baseline_no_conv1d.json'),
            'name': 'Baseline',
            'color': '#E69F00',  # Color-blind-friendly orange
            'linestyle': ':',  # Dotted
            'type': 'baseline',
            'config': {'use_temporal_conv': False},
            'show_in_derivatives': False
        },
        'exp_b_conv1d': {
            'path': str(ROOT / 'models' / 'checkpoints' / 'baseline_conv1d.pth'),
            'config_path': str(ROOT / 'models' / 'checkpoints' / 'configs' / 'best_config_baseline_conv1d.json'),
            'name': 'Baseline+Conv1D',
            'color': '#56B4E9',  # Color-blind-friendly light blue
            'linestyle': ':',  # Dotted
            'type': 'baseline',
            'config': {'use_temporal_conv': True},
            'show_in_derivatives': False
        },
        'exp_d': {
            # EarlyDetect: experiment_f (timing loss without Conv1D)
            # Best model: trial_039 (highest T_lead, early detection)
            'path': str(ROOT / 'models' / 'checkpoints' / 'earlydetect_no_conv1d.pth'),
            'config_path': str(ROOT / 'models' / 'checkpoints' / 'configs' / 'best_config_earlydetect_no_conv1d.json'),
            'name': 'EarlyDetect',
            'color': '#CC0000',  # Strong red
            'linestyle': '-',
            'type': 'earlydetect',
            'config': {'use_temporal_conv': False},
            'show_in_derivatives': True  # Show in derivative plots
        },
        'exp_f': {
            # EarlyDetect+Conv1D: experiment_d (timing loss with Conv1D)
            # Best model: trial_003
            'path': str(ROOT / 'models' / 'checkpoints' / 'earlydetect_conv1d.pth'),
            'config_path': str(ROOT / 'models' / 'checkpoints' / 'configs' / 'best_config_earlydetect_conv1d.json'),
            'name': 'EarlyDetect+Conv1D',
            'color': '#009E73',  # Color-blind-friendly teal
            'linestyle': ':',  # Dotted
            'type': 'earlydetect',
            'config': {'use_temporal_conv': True},
            'show_in_derivatives': False
        }
    }
    
    print(f"\n{'='*100}")
    print("UNIFIED EVALUATION OF ALL EXPERIMENTS")
    print(f"{'='*100}")
    print("Models to evaluate:")
    for key, config in model_configs.items():
        print(f"  - {config['name']}: {config['path']}")
    print(f"{'='*100}\n")
    
    # Evaluate on all ARs
    all_results = {}
    
    for test_AR in args.test_ars:
        max_retries = 3
        retry_delay = 5  # seconds
        
        for attempt in range(max_retries):
            try:
                # Clear GPU cache before each AR evaluation
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                    import gc
                    gc.collect()
                
                metrics = evaluate_all_models_on_ar(
                    test_AR, model_configs, args.data_path, output_dir, device
                )
                all_results[test_AR] = metrics
                break  # Success, exit retry loop
                
            except RuntimeError as e:
                error_msg = str(e)
                if "CUDA" in error_msg and "busy" in error_msg.lower():
                    if attempt < max_retries - 1:
                        print(f"  ⚠️  CUDA busy for AR {test_AR}, attempt {attempt + 1}/{max_retries}")
                        print(f"  Waiting {retry_delay} seconds before retry...")
                        import time
                        time.sleep(retry_delay)
                        # Increase delay for next retry
                        retry_delay *= 2
                        continue
                    else:
                        print(f"  ❌ Failed to evaluate AR {test_AR} after {max_retries} attempts")
                        print(f"  Error: {error_msg}")
                        # Try one more aggressive cleanup
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                            torch.cuda.synchronize()
                            import gc
                            gc.collect()
                            torch.cuda.empty_cache()
                else:
                    # Not a CUDA busy error, re-raise
                    raise
            except Exception as e:
                print(f"  ❌ Error evaluating AR {test_AR}: {e}")
                import traceback
                traceback.print_exc()
                break  # Don't retry for non-CUDA errors
    
    # Export combined CSV files for all ARs
    if all_results:
        print(f"\n{'='*100}")
        print("EXPORTING COMBINED CSV FILES")
        print(f"{'='*100}")
        export_combined_metrics_to_csv(all_results, output_dir, model_configs)
        export_combined_emergence_timing_table(all_results, output_dir, model_configs)
    
    print(f"\n{'='*100}")
    print("EVALUATION COMPLETED")
    print(f"{'='*100}")
    print(f"Results saved to: {output_dir}")
    print(f"Plots (PDFs) saved to: {output_dir}/unified_evaluations/pdf/")
    print(f"Metrics CSVs saved to: {output_dir}/unified_evaluations/csv/")
    print(f"Combined CSVs: all_ARs_metrics.csv, all_ARs_emergence_timing_table.csv")


if __name__ == '__main__':
    main()

