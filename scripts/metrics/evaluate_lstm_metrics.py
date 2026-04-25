#!/usr/bin/env python3
"""
Script to calculate Transformer RMSE, Lead Time, Emergence RMSE, and Model Efficiency
for the LSTM baseline model.

This script evaluates the LSTM model on all test ARs and calculates the same metrics
as used in the paper experiments.
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

from data_loader import load_ar_data_enhanced
from functions_spyros import (
    LSTM, lstm_ready, min_max_scaling, smooth_with_numpy, 
    emergence_indication, recalibrate, calculate_metrics
)


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


def calculate_emergence_metrics(true, pred_lstm, threshold=-0.01, min_duration=4):
    """Calculate emergence metrics for LSTM predictions"""
    # Calculate derivatives
    d_obs = np.gradient(smooth_with_numpy(true))
    d_lstm = np.gradient(pred_lstm)
    
    # Find emergence windows
    obs_start, obs_end = find_first_emergence_window(d_obs, threshold, min_duration)
    lstm_start, lstm_end = find_first_emergence_window(d_lstm, threshold, min_duration)
    
    # Calculate basic metrics
    def calc_basic_metrics(y_true, y_pred):
        mae = np.mean(np.abs(y_true - y_pred))
        mse = np.mean((y_true - y_pred) ** 2)
        rmse = np.sqrt(mse)
        r2 = 1 - np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2)
        return mae, rmse, r2
    
    lstm_mae, lstm_rmse, lstm_r2 = calc_basic_metrics(true, pred_lstm)
    
    # Calculate emergence window metrics
    lstm_emerg_mae, lstm_emerg_rmse, lstm_emerg_r2 = None, None, None
    
    if obs_start is not None and obs_end is not None:
        window_true = true[obs_start:obs_end]
        window_lstm = pred_lstm[obs_start:obs_end]
        
        if len(window_true) > 0:
            lstm_emerg_mae, lstm_emerg_rmse, lstm_emerg_r2 = calc_basic_metrics(window_true, window_lstm)
    
    # Calculate lead time
    lstm_lead_time = None
    if obs_start is not None and lstm_start is not None:
        lstm_lead_time = (obs_start - lstm_start) + 12
    
    return {
        'lstm_mae': lstm_mae,
        'lstm_rmse': lstm_rmse,  # This is the "Transformer RMSE" equivalent (LSTM RMSE)
        'lstm_r2': lstm_r2,
        'lstm_emerg_mae': lstm_emerg_mae,
        'lstm_emerg_rmse': lstm_emerg_rmse,  # Emergence RMSE
        'lstm_emerg_r2': lstm_emerg_r2,
        'lead_time': lstm_lead_time,
        'emergence_window_observed': (obs_start, obs_end) if obs_start is not None else None,
        'emergence_window_lstm': (lstm_start, lstm_end) if lstm_start is not None else None
    }


def calculate_model_efficiency(model, test_input, device):
    """Calculate model efficiency metrics (inference time, memory usage)"""
    model.eval()
    
    # Warm up
    with torch.no_grad():
        _ = model(test_input)
    
    # Measure inference time
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    start_time = time.time()
    
    with torch.no_grad():
        for _ in range(10):  # Run 10 times for average
            _ = model(test_input)
    
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    end_time = time.time()
    
    avg_inference_time = (end_time - start_time) / 10.0
    
    # Get model size
    param_count = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Memory usage (if CUDA available)
    if torch.cuda.is_available():
        memory_allocated = torch.cuda.memory_allocated() / 1024**3  # GB
        memory_reserved = torch.cuda.memory_reserved() / 1024**3  # GB
    else:
        memory_allocated = None
        memory_reserved = None
    
    return {
        'avg_inference_time_ms': avg_inference_time * 1000,  # Convert to milliseconds
        'total_parameters': param_count,
        'trainable_parameters': trainable_params,
        'model_size_mb': param_count * 4 / (1024**2),  # Assuming float32 (4 bytes per param)
        'memory_allocated_gb': memory_allocated,
        'memory_reserved_gb': memory_reserved
    }


def get_ar_settings_fixed(test_AR, rid_of_top):
    """Get AR-specific settings"""
    if test_AR == 11698:
        starting_tile = 46 - rid_of_top * 9
        before_plot = 50
        num_in = 96
    elif test_AR == 11726:
        starting_tile = 37 - rid_of_top * 9
        before_plot = 50
        num_in = 72
    elif test_AR == 13165:
        rid_of_top = 1
        starting_tile = 28 - rid_of_top * 9
        before_plot = 40
        num_in = 96
    elif test_AR == 13179:
        starting_tile = 37 - rid_of_top * 9
        before_plot = 40
        num_in = 96
    elif test_AR == 13183:
        starting_tile = 37 - rid_of_top * 9
        before_plot = 40
        num_in = 96
    else:
        starting_tile = 46 - rid_of_top * 9
        before_plot = 50
        num_in = 96
    
    return starting_tile, before_plot, num_in


def evaluate_lstm_on_ar(lstm_model, test_AR, data_path, lstm_path, device, rid_of_top=1, size=9):
    """Evaluate LSTM model on a specific AR"""
    print(f"  Evaluating AR {test_AR}...")
    
    lstm_model.eval()
    
    try:
        starting_tile, before_plot, num_in = get_ar_settings_fixed(test_AR, rid_of_top)
        
        # Load data using the same pattern as evaluate_all_experiments.py
        inputs, ii, time_arr, norm_stats, ii_raw_full = load_and_preprocess_ar_eval_template(
            test_AR, data_path, rid_of_top, size
        )
        
        # Parse LSTM model parameters from filename
        pat = r't(\d+)_r(\d+)_i(\d+)_n(\d+)_h(\d+)_e(\d+)_l([0-9.]+)\.pth'
        lstm_num_pred, _, _, lstm_num_layers, lstm_hidden_size, n_epochs, lr = (
            int(x) if i!=6 else float(x)
            for i,x in enumerate(re.findall(pat, lstm_path)[0])
        )
        
        all_tile_metrics = []
        efficiency_metrics = []
        
        # Evaluate on 63 tiles (7x9 grid)
        for tile_idx in range(63):
            disp = tile_idx + 1
            if disp % 15 == 0:
                print(f"    Processing Tile {disp}/63")
            
            # Prepare LSTM data
            X_test, y_test = lstm_ready_eval_template(
                tile_idx, size, inputs, ii, num_in, lstm_num_pred, model_seq_len=num_in
            )
            
            if X_test is None or y_test is None or len(X_test) == 0:
                continue
            
            X_test = X_test.to(device)
            y_true_tensor = y_test.to(device)
            
            # Process in batches
            batch_size = min(16, X_test.shape[0])
            tile_metrics_batch = []
            
            for batch_start in range(0, X_test.shape[0], batch_size):
                batch_end = min(batch_start + batch_size, X_test.shape[0])
                
                X_batch = X_test[batch_start:batch_end]
                y_batch_true = y_true_tensor[batch_start:batch_end]
                
                with torch.no_grad():
                    lstm_pred = lstm_model(X_batch)
                    
                    # Get predictions at the last time step (lstm_num_pred - 1)
                    lstm_fut = lstm_num_pred - 1
                    pred_values = lstm_pred[:, lstm_fut].cpu().numpy()
                    true_values = y_batch_true[:, lstm_fut].cpu().numpy()
                    
                    # Recalibrate predictions
                    last_idx = ii.shape[1] - true_values.shape[0] - 1
                    recal_point = ii[tile_idx, last_idx]
                    pred_calibrated = recalibrate(pred_values, recal_point)
                    true_calibrated = recalibrate(true_values, recal_point)
                    
                    # Calculate metrics for this batch
                    metrics = calculate_emergence_metrics(
                        true_calibrated.flatten(), 
                        pred_calibrated.flatten(),
                        threshold=-0.01,
                        min_duration=4
                    )
                    
                    tile_metrics_batch.append(metrics)
                    
                    # Calculate efficiency on first batch of first tile only
                    if tile_idx == 0 and batch_start == 0:
                        eff_metrics = calculate_model_efficiency(lstm_model, X_batch, device)
                        efficiency_metrics.append(eff_metrics)
                
                del X_batch, y_batch_true, lstm_pred
            
            # Average metrics for this tile across batches
            if tile_metrics_batch:
                avg_tile_metrics = {}
                for key in tile_metrics_batch[0].keys():
                    values = [m[key] for m in tile_metrics_batch if m[key] is not None]
                    if values:
                        avg_tile_metrics[key] = np.mean(values) if isinstance(values[0], (int, float)) else values[0]
                    else:
                        avg_tile_metrics[key] = None
                
                avg_tile_metrics['tile_idx'] = tile_idx
                avg_tile_metrics['ar'] = test_AR
                all_tile_metrics.append(avg_tile_metrics)
            
            del X_test, y_test, y_true_tensor
            
            if tile_idx % 10 == 0:
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        # Calculate average metrics for this AR
        if all_tile_metrics:
            avg_metrics = {}
            for key in all_tile_metrics[0].keys():
                if key not in ['tile_idx', 'ar', 'emergence_window_observed', 'emergence_window_lstm']:
                    values = [m[key] for m in all_tile_metrics if m[key] is not None]
                    if values:
                        avg_metrics[key] = np.mean(values)
                    else:
                        avg_metrics[key] = None
                else:
                    # For window metrics, keep as is or count non-None
                    avg_metrics[key] = sum(1 for m in all_tile_metrics if m[key] is not None)
            
            avg_metrics['ar'] = test_AR
            avg_metrics['num_tiles'] = len(all_tile_metrics)
            
            # Add efficiency metrics if available
            if efficiency_metrics:
                avg_metrics['model_efficiency'] = efficiency_metrics[0]
            
            print(f"  AR {test_AR} completed: {len(all_tile_metrics)} tiles evaluated")
            return avg_metrics, all_tile_metrics
        else:
            return None, []
            
    except Exception as e:
        print(f"    Error evaluating AR {test_AR}: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, []
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description='Calculate LSTM Model Metrics')
    parser.add_argument('--lstm_path', type=str, 
                       default=str(ROOT / 'models' / 'checkpoints' / 'lstm_baseline.pth'),
                       help='Path to LSTM model')
    parser.add_argument('--data_path', type=str,
                       default=str(ROOT / 'data'),
                       help='Path to data directory')
    parser.add_argument('--output_dir', type=str,
                       default=str(ROOT / 'results' / 'lstm_metrics'),
                       help='Output directory for results')
    parser.add_argument('--test_ars', type=int, nargs='+',
                       default=[11698, 11726, 13165, 13179, 13183],
                       help='Test ARs to evaluate')
    
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load LSTM model
    print(f"Loading LSTM model from: {args.lstm_path}")
    pat = r't(\d+)_r(\d+)_i(\d+)_n(\d+)_h(\d+)_e(\d+)_l([0-9.]+)\.pth'
    match = re.findall(pat, args.lstm_path)
    
    if not match:
        raise ValueError(f"Could not parse LSTM model path: {args.lstm_path}")
    
    lstm_num_pred, _, _, lstm_num_layers, lstm_hidden_size, n_epochs, lr = (
        int(x) if i!=6 else float(x)
        for i,x in enumerate(match[0])
    )
    
    print(f"LSTM Parameters: num_pred={lstm_num_pred}, layers={lstm_num_layers}, hidden={lstm_hidden_size}")
    
    # Load a sample AR to get input dimension
    sample_ar = args.test_ars[0]
    # Load sample data for efficiency testing
    inputs_sample, _, _, _, _ = load_and_preprocess_ar_eval_template(
        sample_ar, args.data_path, rid_of_top=1, size=9
    )
    input_dim = inputs_sample.shape[1]
    
    # Create and load LSTM model
    lstm_model = LSTM(input_dim, lstm_hidden_size, lstm_num_layers, lstm_num_pred).to(device)
    state_dict = torch.load(args.lstm_path, map_location=device)
    new_state_dict = OrderedDict((k[7:] if k.startswith('module.') else k, v) 
                                 for k, v in state_dict.items())
    lstm_model.load_state_dict(new_state_dict)
    lstm_model.eval()
    
    print(f"LSTM model loaded successfully")
    print(f"Model input dimension: {input_dim}")
    
    # Evaluate on all test ARs
    all_ar_results = []
    all_tile_results = []
    
    for test_AR in args.test_ars:
        print(f"\n{'='*80}")
        print(f"Evaluating AR {test_AR}")
        print(f"{'='*80}")
        
        ar_metrics, tile_metrics = evaluate_lstm_on_ar(
            lstm_model, test_AR, args.data_path, args.lstm_path, device
        )
        
        if ar_metrics:
            all_ar_results.append(ar_metrics)
            all_tile_results.extend(tile_metrics)
    
    # Calculate overall averages
    if all_ar_results:
        print(f"\n{'='*80}")
        print("OVERALL RESULTS")
        print(f"{'='*80}")
        
        overall_metrics = {}
        for key in all_ar_results[0].keys():
            if key not in ['ar', 'num_tiles', 'model_efficiency', 'emergence_window_observed', 'emergence_window_lstm']:
                values = [r[key] for r in all_ar_results if r[key] is not None]
                if values:
                    overall_metrics[key] = np.mean(values)
                else:
                    overall_metrics[key] = None
        
        overall_metrics['total_ars'] = len(all_ar_results)
        overall_metrics['total_tiles'] = sum(r['num_tiles'] for r in all_ar_results)
        
        # Print summary
        print("\nLSTM Model Metrics Summary:")
        print(f"  Overall RMSE: {overall_metrics.get('lstm_rmse', 'N/A'):.6f}")
        print(f"  Overall MAE: {overall_metrics.get('lstm_mae', 'N/A'):.6f}")
        print(f"  Overall R²: {overall_metrics.get('lstm_r2', 'N/A'):.6f}")
        print(f"  Emergence RMSE: {overall_metrics.get('lstm_emerg_rmse', 'N/A'):.6f}")
        print(f"  Lead Time (hrs): {overall_metrics.get('lead_time', 'N/A')}")
        
        if 'model_efficiency' in all_ar_results[0]:
            eff = all_ar_results[0]['model_efficiency']
            print(f"\nModel Efficiency:")
            print(f"  Average Inference Time: {eff.get('avg_inference_time_ms', 'N/A'):.2f} ms")
            print(f"  Total Parameters: {eff.get('total_parameters', 'N/A'):,}")
            print(f"  Model Size: {eff.get('model_size_mb', 'N/A'):.2f} MB")
        
        # Save results
        results_df = pd.DataFrame(all_ar_results)
        results_df.to_csv(output_dir / 'lstm_metrics_by_ar.csv', index=False)
        
        tile_results_df = pd.DataFrame(all_tile_results)
        tile_results_df.to_csv(output_dir / 'lstm_metrics_by_tile.csv', index=False)
        
        overall_df = pd.DataFrame([overall_metrics])
        overall_df.to_csv(output_dir / 'lstm_metrics_overall.csv', index=False)
        
        # Save as JSON for easy reading
        with open(output_dir / 'lstm_metrics_overall.json', 'w') as f:
            json.dump(overall_metrics, f, indent=2, default=str)
        
        with open(output_dir / 'lstm_metrics_by_ar.json', 'w') as f:
            json.dump(all_ar_results, f, indent=2, default=str)
        
        print(f"\nResults saved to: {output_dir}")
        print(f"  - lstm_metrics_overall.csv")
        print(f"  - lstm_metrics_by_ar.csv")
        print(f"  - lstm_metrics_by_tile.csv")
        print(f"  - lstm_metrics_overall.json")
        print(f"  - lstm_metrics_by_ar.json")
    else:
        print("No results to save!")


if __name__ == '__main__':
    main()

