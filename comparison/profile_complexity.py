#!/usr/bin/env python3
"""
Profile all models and generate complexity metrics table.

This script profiles all 5 models (LSTM, Baseline, Baseline+Conv1D, EarlyDetect, EarlyDetect+Conv1D)
and generates a CSV table for the paper (tab:complexity_results).
"""

import sys
import os
import time
import json
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
from pathlib import Path
import argparse

# Add paths for imports
ROOT = Path(__file__).resolve().parents[1]
sys.path.extend([
    str(ROOT / "scripts"),
    str(ROOT / "scripts/models"),
])

from functions_spyros import LSTM
from baseline_transformer import SARTransformerLocalTile as BaselineTransformer
from early_detect_transformer import SARTransformerLocalTile as EarlyDetectTransformer

# Install thop if not available
try:
    from thop import profile, clever_format
except ImportError:
    print("Installing thop...")
    os.system("pip install thop")
    from thop import profile, clever_format

def measure_peak_memory():
    """Get peak GPU memory usage in MB"""
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / 1024 / 1024
    return 0

def profile_model(model, sample_input, sample_target, device, model_name, loss_fn=None, optimizer=None):
    """Profile a single model and return metrics
    
    Matches original profiling methodology:
    - Measures peak memory AFTER forward+backward passes (includes gradients)
    - Uses same timing and memory measurement approach
    """
    print(f"\n{'='*80}")
    print(f"Profiling {model_name}")
    print(f"{'='*80}")
    
    model = model.to(device)
    
    # Clear memory and reset peak stats AFTER model is on device (matches original)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    
    # Measure MACs and FLOPs using thop (also returns parameter count)
    print("Measuring MACs and FLOPs...")
    try:
        macs, params = profile(model, inputs=(sample_input,), verbose=False)
        flops = macs * 2  # Approximate FLOPs from MACs
        print(f"MACs: {clever_format(macs, '%.3f')}")
        print(f"FLOPs: {clever_format(flops, '%.3f')}")
        print(f"Parameters: {clever_format(params, '%.3f')}")
    except Exception as e:
        print(f"Warning: Could not measure FLOPs: {e}")
        flops = 0
        params = sum(p.numel() for p in model.parameters())
        print(f"Parameters (manual count): {params:,}")
    
    # Forward-only timing
    print("Measuring forward-only runtime...")
    model.eval()
    forward_times = []
    
    # Warmup
    for _ in range(5):
        with torch.no_grad():
            _ = model(sample_input)
    
    # Measure forward pass
    for _ in range(10):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start_time = time.time()
        with torch.no_grad():
            _ = model(sample_input)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        forward_times.append(time.time() - start_time)
    
    forward_median = np.median(forward_times)
    print(f"Forward-only median time: {forward_median*1000:.2f} ms")
    
    # Forward+backward timing (to match original methodology)
    # This also ensures peak memory includes gradient computation
    if loss_fn is not None and optimizer is not None and sample_target is not None:
        print("Measuring forward+backward runtime...")
        model.train()
        fwd_bwd_times = []
        
        # Warmup
        for _ in range(5):
            optimizer.zero_grad()
            outputs = model(sample_input)
            loss = loss_fn(outputs, sample_target)
            loss.backward()
            optimizer.step()
        
        # Measure forward+backward pass
        for _ in range(10):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            start_time = time.time()
            optimizer.zero_grad()
            outputs = model(sample_input)
            loss = loss_fn(outputs, sample_target)
            loss.backward()
            optimizer.step()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            fwd_bwd_times.append(time.time() - start_time)
        
        fwd_bwd_median = np.median(fwd_bwd_times)
        print(f"Forward+backward median time: {fwd_bwd_median*1000:.2f} ms")
    
    # Memory usage - measure AFTER forward+backward passes (matches original)
    peak_memory = measure_peak_memory()
    print(f"Peak GPU memory: {peak_memory:.2f} MB")
    
    return {
        'parameters': int(params),
        'flops': int(flops),
        'peak_memory_mb': peak_memory,
        'forward_time_ms': forward_median * 1000
    }

def main():
    parser = argparse.ArgumentParser(description='Profile all models and generate complexity table')
    parser.add_argument('--output', type=str, default='complexity_table.csv',
                       help='Output CSV file path')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda or cpu)')
    
    args = parser.parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() and args.device == 'cuda' else 'cpu')
    print(f"Using device: {device}")
    
    # Create sample input data (batch_size=32, seq_len=128, input_dim=5)
    batch_size = 32
    seq_len = 128
    input_dim = 5
    sample_input = torch.randn(batch_size, seq_len, input_dim).to(device)
    sample_target = torch.randn(batch_size, 12).to(device)  # For loss computation
    
    all_results = {}
    
    # Profile LSTM
    print("\n" + "="*80)
    print("PROFILING LSTM")
    print("="*80)
    # Clear memory before each model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    
    lstm = LSTM(input_size=input_dim, hidden_size=64, num_layers=3, output_length=12)
    # LSTM forward takes (batch, seq_len, input_dim) and returns (batch, output_length)
    lstm_sample = torch.randn(batch_size, 110, input_dim).to(device)  # LSTM uses 110 timesteps
    lstm_target = torch.randn(batch_size, 12).to(device)
    lstm_loss_fn = torch.nn.MSELoss()
    lstm_optimizer = torch.optim.Adam(lstm.parameters(), lr=0.01)
    lstm_metrics = profile_model(lstm, lstm_sample, lstm_target, device, "LSTM", lstm_loss_fn, lstm_optimizer)
    all_results['LSTM'] = lstm_metrics
    
    # Clean up LSTM
    del lstm, lstm_sample, lstm_target, lstm_loss_fn, lstm_optimizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Profile Baseline (no Conv1D)
    print("\n" + "="*80)
    print("PROFILING BASELINE (no Conv1D)")
    print("="*80)
    # Clear memory before each model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    
    baseline = BaselineTransformer(
        input_dim=input_dim,
        d_model=256,
        nhead=4,
        num_layers=5,
        dropout=0.0,
        output_len=12,
        max_seq_len=150,
        use_temporal_conv=False
    )
    baseline_loss_fn = torch.nn.MSELoss()
    baseline_optimizer = torch.optim.AdamW(baseline.parameters(), lr=0.0001, weight_decay=1e-5)
    baseline_metrics = profile_model(baseline, sample_input, sample_target, device, "Baseline", baseline_loss_fn, baseline_optimizer)
    all_results['Baseline'] = baseline_metrics
    
    # Clean up
    del baseline, baseline_loss_fn, baseline_optimizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Profile Baseline+Conv1D
    print("\n" + "="*80)
    print("PROFILING BASELINE+Conv1D")
    print("="*80)
    # Clear memory before each model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    
    baseline_conv1d = BaselineTransformer(
        input_dim=input_dim,
        d_model=256,
        nhead=4,
        num_layers=5,
        dropout=0.0,
        output_len=12,
        max_seq_len=150,
        use_temporal_conv=True
    )
    baseline_conv1d_loss_fn = torch.nn.MSELoss()
    baseline_conv1d_optimizer = torch.optim.AdamW(baseline_conv1d.parameters(), lr=0.0001, weight_decay=1e-5)
    baseline_conv1d_metrics = profile_model(baseline_conv1d, sample_input, sample_target, device, "Baseline+Conv1D", baseline_conv1d_loss_fn, baseline_conv1d_optimizer)
    all_results['Baseline+Conv1D'] = baseline_conv1d_metrics
    
    # Clean up
    del baseline_conv1d, baseline_conv1d_loss_fn, baseline_conv1d_optimizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Profile EarlyDetect (no Conv1D)
    print("\n" + "="*80)
    print("PROFILING EARLYDETECT (no Conv1D)")
    print("="*80)
    # Clear memory before each model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    
    earlydetect = EarlyDetectTransformer(
        input_dim=6,
        d_model=256,
        nhead=8,
        num_layers=6,
        dropout=0.0,
        output_len=12,
        max_seq_len=150,
        use_temporal_conv=False,
        timing_bias_weight=0.1,
        early_detection_weight=0.3
    )
    earlydetect_loss_fn = torch.nn.MSELoss()
    earlydetect_optimizer = torch.optim.AdamW(earlydetect.parameters(), lr=0.0002, weight_decay=1e-5)
    earlydetect_metrics = profile_model(earlydetect, sample_input, sample_target, device, "EarlyDetect", earlydetect_loss_fn, earlydetect_optimizer)
    all_results['EarlyDetect'] = earlydetect_metrics
    
    # Clean up
    del earlydetect, earlydetect_loss_fn, earlydetect_optimizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Profile EarlyDetect+Conv1D
    print("\n" + "="*80)
    print("PROFILING EARLYDETECT+Conv1D")
    print("="*80)
    # Clear memory before each model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    
    earlydetect_conv1d = EarlyDetectTransformer(
        input_dim=input_dim,
        d_model=512,
        nhead=4,
        num_layers=4,
        dropout=0.0,
        output_len=12,
        max_seq_len=150,
        use_temporal_conv=True,
        timing_bias_weight=0.3,
        early_detection_weight=0.1
    )
    earlydetect_conv1d_loss_fn = torch.nn.MSELoss()
    earlydetect_conv1d_optimizer = torch.optim.AdamW(earlydetect_conv1d.parameters(), lr=5e-5, weight_decay=1e-5)
    earlydetect_conv1d_metrics = profile_model(earlydetect_conv1d, sample_input, sample_target, device, "EarlyDetect+Conv1D", earlydetect_conv1d_loss_fn, earlydetect_conv1d_optimizer)
    all_results['EarlyDetect+Conv1D'] = earlydetect_conv1d_metrics
    
    # Clean up
    del earlydetect_conv1d, earlydetect_conv1d_loss_fn, earlydetect_conv1d_optimizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Generate table
    print("\n" + "="*80)
    print("GENERATING COMPLEXITY TABLE")
    print("="*80)
    
    data = []
    model_order = ['LSTM', 'Baseline', 'Baseline+Conv1D', 'EarlyDetect', 'EarlyDetect+Conv1D']
    
    for model_name in model_order:
        if model_name not in all_results:
            continue
        
        metrics = all_results[model_name]
        
        # Format parameters
        params = metrics['parameters']
        if params >= 1e6:
            params_str = f"{params/1e6:.1f}M"
        elif params >= 1e3:
            params_str = f"{params/1e3:.1f}K"
        else:
            params_str = f"{params:.0f}"
        
        # Format FLOPs (in Giga)
        flops_g = metrics['flops'] / 1e9
        
        # Format memory (MB)
        memory_mb = metrics['peak_memory_mb']
        
        # Format forward time (ms)
        forward_time_ms = metrics['forward_time_ms']
        
        data.append({
            'Model': model_name,
            'Parameters': params_str,
            'FLOPs (G)': f"{flops_g:.1f}",
            'Peak Memory (MB)': f"{memory_mb:.1f}",
            'Fwd. Pass Time (ms)': f"{forward_time_ms:.2f}"
        })
    
    df = pd.DataFrame(data)
    
    # Save table
    output_path = Path(ROOT) / 'comparison' / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    
    print(f"\nComplexity table saved to: {output_path}")
    print("\nTable contents:")
    print(df.to_string(index=False))
    
    print("\n" + "="*80)
    print("PROFILING COMPLETED")
    print("="*80)
    
    return 0

if __name__ == '__main__':
    exit(main())

