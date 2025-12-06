import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import itertools
import json
import pickle
from pathlib import Path
import argparse
from datetime import datetime, timedelta
import time
import matplotlib.pyplot as plt
import os
import re
from collections import OrderedDict
import matplotlib.dates as mdates
from matplotlib import gridspec
from PIL import Image
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.extend(
    [
        str(ROOT / "scripts"),
        str(ROOT / "scripts/models"),
        str(ROOT / "scripts/data"),
    ]
)

from baseline_transformer import SARTransformerLocalTile
from data_loader import load_ar_data_enhanced, cross_ar_tile_data_preparation_attention
from functions_spyros import (
    LSTM,
    lstm_ready,
    min_max_scaling,
    smooth_with_numpy,
    emergence_indication,
    recalibrate,
    calculate_metrics,
)

def emergence_aware_loss(predictions, targets, lambda_emergence=0.1):
    """Loss function that encourages attention to emergence patterns"""
    mse_loss = nn.MSELoss()(predictions, targets)
    pred_derivatives = torch.gradient(predictions, dim=1)[0]
    target_derivatives = torch.gradient(targets, dim=1)[0]
    derivative_loss = nn.MSELoss()(pred_derivatives, target_derivatives)
    temporal_consistency = torch.mean(torch.abs(pred_derivatives[:, 1:] - pred_derivatives[:, :-1]))
    total_loss = mse_loss + lambda_emergence * derivative_loss + 0.01 * temporal_consistency
    return total_loss

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
    """Load and preprocess AR data"""
    base = f'{data_path}/AR{test_AR}'
    power = np.load(os.path.join(base, f'mean_pmdop{test_AR}_flat.npz'), allow_pickle=True)
    mag   = np.load(os.path.join(base, f'mean_mag{test_AR}_flat.npz'),   allow_pickle=True)
    cont  = np.load(os.path.join(base, f'mean_int{test_AR}_flat.npz'),   allow_pickle=True)

    pm23, pm34, pm45, pm56, time_arr = (
        power['arr_0'], power['arr_1'], power['arr_2'], power['arr_3'], power['arr_4']
    )
    mf = mag['arr_0']; ii = cont['arr_0']

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
    
    return inputs, ii, time_arr

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

def calculate_emergence_metrics_detailed(true, pred_lstm, pred_transformer, time_arr, threshold=-0.01, min_duration=4):
    """Calculate emergence metrics"""
    d_obs = np.gradient(smooth_with_numpy(true))
    d_lstm = np.gradient(pred_lstm)
    d_transformer = np.gradient(pred_transformer)
    
    obs_start, obs_end = find_first_emergence_window(d_obs, threshold, min_duration)
    lstm_start, lstm_end = find_first_emergence_window(d_lstm, threshold, min_duration)
    transformer_start, transformer_end = find_first_emergence_window(d_transformer, threshold, min_duration)
    
    def calc_basic_metrics(y_true, y_pred):
        mae = np.mean(np.abs(y_true - y_pred))
        mse = np.mean((y_true - y_pred) ** 2)
        rmse = np.sqrt(mse)
        r2 = 1 - np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2)
        return mae, rmse, r2
    
    lstm_mae, lstm_rmse, lstm_r2 = calc_basic_metrics(true, pred_lstm)
    transformer_mae, transformer_rmse, transformer_r2 = calc_basic_metrics(true, pred_transformer)
    
    lstm_emerg_mae, lstm_emerg_rmse, lstm_emerg_r2 = None, None, None
    transformer_emerg_mae, transformer_emerg_rmse, transformer_emerg_r2 = None, None, None
    
    if obs_start is not None and obs_end is not None:
        window_true = true[obs_start:obs_end]
        window_lstm = pred_lstm[obs_start:obs_end]
        window_transformer = pred_transformer[obs_start:obs_end]
        
        if len(window_true) > 0:
            lstm_emerg_mae, lstm_emerg_rmse, lstm_emerg_r2 = calc_basic_metrics(window_true, window_lstm)
            transformer_emerg_mae, transformer_emerg_rmse, transformer_emerg_r2 = calc_basic_metrics(window_true, window_transformer)
    
    lstm_timing_diff = None
    transformer_timing_diff = None
    
    if obs_start is not None:
        if lstm_start is not None:
            lstm_timing_diff = (lstm_start - obs_start)
        if transformer_start is not None:
            transformer_timing_diff = (transformer_start - obs_start)
    
    return {
        'lstm': {
            'MAE': lstm_mae,
            'RMSE': lstm_rmse,
            'R2': lstm_r2,
            'emerg_MAE': lstm_emerg_mae,
            'emerg_RMSE': lstm_emerg_rmse,
            'emerg_R2': lstm_emerg_r2,
            'emergence_timing_diff': lstm_timing_diff,
            'emergence_window': (lstm_start, lstm_end) if lstm_start is not None else None
        },
        'transformer': {
            'MAE': transformer_mae,
            'RMSE': transformer_rmse,
            'R2': transformer_r2,
            'emerg_MAE': transformer_emerg_mae,
            'emerg_RMSE': transformer_emerg_rmse,
            'emerg_R2': transformer_emerg_r2,
            'emergence_timing_diff': transformer_timing_diff,
            'emergence_window': (transformer_start, transformer_end) if transformer_start is not None else None
        },
        'observed': {
            'emergence_window': (obs_start, obs_end) if obs_start is not None else None
        }
    }

def create_emergence_metrics_table_eval_template(ax, metrics):
    """Create emergence metrics table"""
    lstm_metrics = metrics['lstm']
    transformer_metrics = metrics['transformer']
    obs_metrics = metrics['observed']
    
    has_emergence_window = obs_metrics['emergence_window'] is not None
    
    data = [['Metric', 'LSTM', 'Transformer']]
    
    data.extend([
        ['Overall MAE', f'{lstm_metrics["MAE"]:.4f}', f'{transformer_metrics["MAE"]:.4f}'],
        ['Overall RMSE', f'{lstm_metrics["RMSE"]:.4f}', f'{transformer_metrics["RMSE"]:.4f}'],
        ['Overall R2', f'{lstm_metrics["R2"]:.4f}', f'{transformer_metrics["R2"]:.4f}']
    ])
    
    if has_emergence_window:
        data.extend([
            ['Window MAE', 
             f'{lstm_metrics["emerg_MAE"]:.4f}' if lstm_metrics["emerg_MAE"] is not None else 'N/A',
             f'{transformer_metrics["emerg_MAE"]:.4f}' if transformer_metrics["emerg_MAE"] is not None else 'N/A'],
            ['Window RMSE', 
             f'{lstm_metrics["emerg_RMSE"]:.4f}' if lstm_metrics["emerg_RMSE"] is not None else 'N/A',
             f'{transformer_metrics["emerg_RMSE"]:.4f}' if transformer_metrics["emerg_RMSE"] is not None else 'N/A'],
            ['Window R2', 
             f'{lstm_metrics["emerg_R2"]:.4f}' if lstm_metrics["emerg_R2"] is not None else 'N/A',
             f'{transformer_metrics["emerg_R2"]:.4f}' if transformer_metrics["emerg_R2"] is not None else 'N/A']
        ])
    
    data.append(['? Emergence (hrs)', 
                f'{lstm_metrics["emergence_timing_diff"]:+.0f}' if lstm_metrics["emergence_timing_diff"] is not None else 'N/A',
                f'{transformer_metrics["emergence_timing_diff"]:+.0f}' if transformer_metrics["emergence_timing_diff"] is not None else 'N/A'])
    
    table_height = 0.8 if has_emergence_window else 0.6
    table_y_position = -0.6 if has_emergence_window else -0.4
    
    table = ax.table(
        cellText=data,
        loc='upper left',
        bbox=[1.02, table_y_position, 0.25, table_height],
        cellLoc='center',
        colLoc='center'
    )
    
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    
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
        
        if '? Emergence' in str(cell.get_text().get_text()) and col == 0:
            cell.set_text_props(fontsize=6)
        else:
            cell.set_text_props(fontsize=8)

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

def evaluate_all_ars_eval_template(
    model_path, 
    config, 
    data_path, 
    lstm_path, 
    output_dir,
    trial_idx
):
    """Evaluate ALL ARs with Conv1D support"""
    
    test_ars = [11698, 11726, 13165, 13179, 13183]
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    all_ar_results = {}
    successful_evaluations = 0
    
    for test_AR in test_ars:
        print(f"  Evaluating AR {test_AR} with Conv1D...")
        
        try:
            rid_of_top = 1
            size = 9
            start_tile, before_plot, num_in, NOAA_first, NOAA_second = get_ar_settings_fixed(test_AR, rid_of_top)
            
            NOAA1 = mdates.date2num(NOAA_first)
            NOAA2 = mdates.date2num(NOAA_second)
            
            inputs, ii, time_arr = load_and_preprocess_ar_eval_template(test_AR, data_path, rid_of_top, size)
            
            pat = r't(\d+)_r(\d+)_i(\d+)_n(\d+)_h(\d+)_e(\d+)_l([0-9.]+)\.pth'
            matches = re.findall(pat, lstm_path)
            if matches and len(matches) > 0:
                lstm_num_pred, _, _, lstm_num_layers, lstm_hidden_size, n_epochs, lr = (
                    int(x) if i!=6 else float(x)
                    for i,x in enumerate(matches[0])
                )
            else:
                # Default values for lstm_baseline.pth or other non-matching filenames
                # These match the typical LSTM configuration used in the paper
                lstm_num_pred = 12
                lstm_num_layers = 3
                lstm_hidden_size = 64
                print(f"    ⚠️  Could not parse LSTM path pattern, using defaults: num_pred={lstm_num_pred}, layers={lstm_num_layers}, hidden={lstm_hidden_size}")
            
            lstm = LSTM(inputs.shape[1], lstm_hidden_size, lstm_num_layers, lstm_num_pred).to(device)
            sd = torch.load(lstm_path,map_location=device)
            new_sd = OrderedDict((k[7:] if k.startswith('module.') else k, v) for k,v in sd.items())
            lstm.load_state_dict(new_sd); lstm.eval()
            
            # Load Enhanced Transformer with Conv1D
            transformer = SARTransformerLocalTile(
                input_dim=inputs.shape[1],
                d_model=config['d_model'],
                nhead=config['nhead'],
                num_layers=config['num_layers'],
                dropout=config['dropout'],
                output_len=config['output_len'],
                max_seq_len=150,
                use_temporal_conv=config.get('use_temporal_conv', True)  # NEW: Conv1D support
            ).to(device)
            
            transformer.load_state_dict(torch.load(model_path, map_location=device))
            transformer.eval()
            
            conv_status = "? ENABLED" if config.get('use_temporal_conv', True) else "? DISABLED"
            print(f"    Conv1D Temporal Features: {conv_status}")
            
            all_tile_metrics = []
            
            # Create evaluation plots with Conv1D information
            fig = plt.figure(figsize=(16,46))
            fig.subplots_adjust(left=0.15, right=0.85, top=0.97, bottom=0.1)
            gs0 = gridspec.GridSpec(7,1,figure=fig,hspace=.2)
            
            lstm_fut = lstm_num_pred-1
            transformer_fut = config['output_len']-1
            thr= -0.01; st=4
            
            for i in range(7):
                tile_idx = start_tile + i
                disp = tile_idx + 10
                print(f"    Processing Tile {disp}")
                
                X_test, y_test = lstm_ready_eval_template(tile_idx, size, inputs, ii, num_in, config['output_len'], model_seq_len=num_in)
                X_test = X_test.to(device)
                Xt = X_test.view(X_test.size(0), num_in, X_test.size(2))
                
                with torch.no_grad():
                    p_l = lstm(X_test)[:,lstm_fut].cpu().numpy()
                    p_t = transformer(Xt)[:,transformer_fut].cpu().numpy()
                true = y_test[:,lstm_fut].numpy()
                
                last = ii.shape[1]-true.shape[0]-1
                p_l = recalibrate(p_l, ii[tile_idx,last])
                p_t = recalibrate(p_t, ii[tile_idx,last])
                
                tile_metrics = calculate_emergence_metrics_detailed(true, p_l, p_t, time_arr, thr, st)
                all_tile_metrics.append(tile_metrics)
                
                before = ii[tile_idx,last-before_plot:last]
                tcut = time_arr[last-before_plot:last+true.shape[0]]
                tnum = mdates.date2num(tcut)
                nanarr = np.full(before.shape, np.nan)
                
                d_obs = np.gradient(smooth_with_numpy(np.concatenate((before, true))))
                d_l = np.gradient(p_l)
                d_t = np.gradient(p_t)
                
                nan_pad = np.full(before_plot, np.nan)
                d_l_full = np.concatenate([nan_pad, d_l])
                d_t_full = np.concatenate([nan_pad, d_t])
                
                ind_o = emergence_indication(d_obs, thr, st)
                
                obs_window = None
                for idx, val in enumerate(ind_o):
                    if val != 0:
                        start_idx = max(0, idx - 12)
                        end_idx = min(len(d_obs), idx + 12)
                        obs_window = (start_idx, end_idx)
                        break
                
                t_start = None
                t_end = None
                if obs_window:
                    t_start = tnum[obs_window[0]]
                    t_end = tnum[obs_window[1] - 1]
                
                gs1 = gridspec.GridSpecFromSubplotSpec(5,1,subplot_spec=gs0[i],height_ratios=[18,4,4,4,4],hspace=0.3)
                
                ax0 = fig.add_subplot(gs1[0])
                ax0.plot(tnum, np.concatenate((before,true)), 'k-', label='Observed Intensity')
                ax0.plot(tnum, np.concatenate((nanarr,p_l)), 'b-', label='LSTM Prediction')
                
                # Update transformer label to show Conv1D status
                transformer_label = f'Transformer + Conv1D ({conv_status})'
                ax0.plot(tnum, np.concatenate((nanarr,p_t)), 'r-', label=transformer_label)
                ax0.axvline(NOAA1, color='magenta', linestyle='--', label='NOAA First Record')
                ax0.axvline(NOAA2, color='darkmagenta', linestyle='--', label='NOAA Second Record')
                
                if obs_window:
                    ax0.axvspan(t_start, t_end, color='yellow', alpha=0.3, label='Emergence Window')
                
                ax0.set_title(f'Tile {disp} - Trial {trial_idx} - AR {test_AR} (Conv1D {conv_status})', fontsize=12)
                ax0.set_ylabel('Normalized Intensity', fontsize=9, labelpad=20)
                ax0.set_ylim([-0.1,1.1]); ax0.grid(True)
                ax0.set_yticks([0, 0.25, 0.5, 0.75, 1])
                legend = ax0.legend(bbox_to_anchor=(1.033, .83, 0.223, 0.11), loc='upper left', borderaxespad=0, fontsize=10, framealpha=1, mode='expand')
                legend.get_frame().set_boxstyle('square', pad=1)
                ax0.tick_params(labelbottom=False)
                
                create_emergence_metrics_table_eval_template(ax0, tile_metrics)
                
                # Derivative plots
                ax1 = fig.add_subplot(gs1[1], sharex=ax0)
                ax1.plot(tnum, d_obs, color='black', linewidth=1)
                
                if obs_window:
                    ax1.axvspan(t_start, t_end, color='yellow', alpha=0.3)
                
                for j in range(len(d_obs)-1):
                    if ind_o[j] != 0:
                        ax1.plot(tnum[j:j+2], d_obs[j:j+2], color='green', linewidth=1)
                ax1.set_ylabel('dObs/dt', fontsize=7, labelpad=10)
                ax1.set_ylim([-0.05,0.05]); ax1.set_yticks([0]); ax1.grid(True)
                ax1.tick_params(labelbottom=False)
                
                ax2 = fig.add_subplot(gs1[2], sharex=ax0)
                ax2.plot(tnum, d_t_full, color='red', linewidth=1)
                
                if obs_window:
                    ax2.axvspan(t_start, t_end, color='yellow', alpha=0.3)
                
                ind_t = emergence_indication(d_t_full, thr, st)
                for j in range(len(d_t_full)-1):
                    if ind_t[j] != 0:
                        ax2.plot(tnum[j:j+2], d_t_full[j:j+2], color='green', linewidth=1)
                ax2.set_ylabel('dTrans+Conv1D/dt', fontsize=7, labelpad=10)
                ax2.set_ylim([-0.05,0.05]); ax2.set_yticks([0]); ax2.grid(True)
                ax2.tick_params(labelbottom=False)
                ax2.set_xlim(tnum[0], tnum[-1])
                
                ax3 = fig.add_subplot(gs1[3], sharex=ax0)
                ax3.plot(tnum, d_l_full, color='blue', linewidth=1)
                
                if obs_window:
                    ax3.axvspan(t_start, t_end, color='yellow', alpha=0.3)
                
                ind_l = emergence_indication(d_l_full, thr, st)
                for j in range(len(d_l_full)-1):
                    if ind_l[j] != 0:
                        ax3.plot(tnum[j:j+2], d_l_full[j:j+2], color='green', linewidth=1)
                ax3.set_ylabel('dLSTM/dt', fontsize=7, labelpad=10)
                ax3.set_ylim([-0.05,0.05]); ax3.set_yticks([0]); ax3.grid(True)
                ax3.tick_params(labelbottom=False)
                ax3.set_xlim(tnum[0], tnum[-1])
                
                # Error analysis
                ax4 = fig.add_subplot(gs1[4], sharex=ax0)
                lstm_errors = np.abs(true - p_l)
                transformer_errors = np.abs(true - p_t)
                
                ax4.plot(tnum[before_plot:before_plot+len(true)], lstm_errors, 'b-', label='LSTM')
                ax4.plot(tnum[before_plot:before_plot+len(true)], transformer_errors, 'r-', label='Transformer+Conv1D')
                ax4.axvline(NOAA1, color='magenta', linestyle='--')
                
                if obs_window:
                    ax4.axvspan(t_start, t_end, color='yellow', alpha=0.3)
                
                x_vals = np.arange(len(lstm_errors))
                
                z_lstm = np.polyfit(x_vals, lstm_errors, 1)
                p_lstm = np.poly1d(z_lstm)
                ax4.plot(tnum[before_plot:before_plot+len(true)], p_lstm(x_vals), 'b--', alpha=0.7, linewidth=1)
                
                z_transformer = np.polyfit(x_vals, transformer_errors, 1)
                p_transformer = np.poly1d(z_transformer)
                ax4.plot(tnum[before_plot:before_plot+len(true)], p_transformer(x_vals), 'r--', alpha=0.7, linewidth=1)
                
                slope_lstm, intercept_lstm = z_lstm[0], z_lstm[1]
                slope_transformer, intercept_transformer = z_transformer[0], z_transformer[1]
                
                formula_lstm = f'LSTM: y = {slope_lstm:.4f}x + {intercept_lstm:.4f}'
                formula_transformer = f'Transformer+Conv1D: y = {slope_transformer:.4f}x + {intercept_transformer:.4f}'
                
                ax4.text(0.02, 0.98, formula_lstm, transform=ax4.transAxes, fontsize=7, 
                        verticalalignment='top', color='blue', bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))
                ax4.text(0.02, 0.85, formula_transformer, transform=ax4.transAxes, fontsize=7, 
                        verticalalignment='top', color='red', bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))
                
                ax4.set_ylabel('|Error|', fontsize=8)
                ax4.set_xlabel('Date', fontsize=10)
                ax4.set_xlim(tnum[0], tnum[-1]); ax4.grid(True)
                ax4.xaxis.set_major_locator(mdates.DayLocator())
                ax4.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
                ax4.tick_params(labelbottom=True)
            
            # Parameter and metrics tables
            def extract_params(path):
                fname = os.path.basename(path)
                pat = r't(\d+)_r(\d+)_i(\d+)_n(\d+)_h(\d+)_e(\d+)_l([0-9.]+)\.pth$'
                m = re.search(pat, fname)
                if not m:
                    return None
                return {
                    'Time Window': m.group(1),
                    'Rid of Top': m.group(2),
                    'Input Len': m.group(3),
                    'Layers': m.group(4),
                    'Hidden': m.group(5),
                    'Epochs': m.group(6),
                    'LR': m.group(7)
                }

            lstm_params = extract_params(lstm_path)
            # Handle case where LSTM params can't be extracted (e.g., lstm_baseline.pth)
            if lstm_params is None:
                lstm_params = {
                    'Time Window': '12',
                    'Rid of Top': '1',
                    'Input Len': '110',
                    'Layers': '3',
                    'Hidden': '64',
                    'Epochs': '1000',
                    'LR': '0.01'
                }
                print(f"    ⚠️  Using default LSTM parameters (could not parse from filename)")
            
            trfm_params = {
                'Time Window': config['output_len'],
                'Rid of Top': 1, 
                'Input Len': num_in,
                'Layers': config['num_layers'],
                'Hidden': config['d_model'],
                'Epochs': "1000",
                'LR': config['learning_rate']
            }
            
            def mean_metric(all_metrics, model_key, metric_key):
                values = []
                for tile_metrics in all_metrics:
                    val = tile_metrics[model_key][metric_key]
                    if val is not None and not np.isnan(val):
                        values.append(val)
                return np.mean(values) if values else None
            
            param_headers = ["Parameter", "LSTM", "Transformer+Conv1D"]
            param_rows = [
                ["Time Window", lstm_params['Time Window'], trfm_params['Time Window']],
                ["Rid of Top", lstm_params['Rid of Top'], trfm_params['Rid of Top']],
                ["Input Len", lstm_params['Input Len'], trfm_params['Input Len']],
                ["Layers", lstm_params['Layers'], trfm_params['Layers']],
                ["Hidden", lstm_params['Hidden'], trfm_params['Hidden']],
                ["Epochs", lstm_params['Epochs'], trfm_params['Epochs']],
                ["LR", lstm_params['LR'], trfm_params['LR']],
                ["Conv1D", "N/A", conv_status],
            ]
            
            metric_names = ['MAE', 'RMSE', 'R2', 'emerg_MAE', 'emerg_RMSE', 'emerg_R2', 'emergence_timing_diff']
            metric_labels = ['Overall MAE', 'Overall RMSE', 'Overall R2', 'Window MAE', 'Window RMSE', 'Window R2', '? Emergence(hrs)']
            
            metric_rows = []
            for name, label in zip(metric_names, metric_labels):
                lstm_val = mean_metric(all_tile_metrics, 'lstm', name)
                transformer_val = mean_metric(all_tile_metrics, 'transformer', name)
                
                if name == 'emergence_timing_diff':
                    lstm_str = f"{lstm_val:+.1f}" if lstm_val is not None else "N/A"
                    transformer_str = f"{transformer_val:+.1f}" if transformer_val is not None else "N/A"
                else:
                    lstm_str = f"{lstm_val:.4f}" if lstm_val is not None else "N/A"
                    transformer_str = f"{transformer_val:.4f}" if transformer_val is not None else "N/A"
                
                metric_rows.append([label, lstm_str, transformer_str])
            
            # Create tables
            metrics_ax = fig.add_axes([0.15, -0.045, 0.3, 0.12])
            metrics_ax.axis('off')
            
            metrics_ax.text(0.5, 1, 'Overall Performance Metrics', 
                           ha='center', va='center', fontsize=12)
            
            metrics_data = [['Metric', 'LSTM', 'Transformer+Conv1D']] + metric_rows
            
            metrics_table = metrics_ax.table(
                cellText=metrics_data,
                colLabels=['Metric', 'LSTM', 'Transformer+Conv1D'],
                colColours=['#e0e0e0'] * 3,
                cellLoc='center',
                loc='upper center'
            )
            
            metrics_table.auto_set_font_size(False)
            metrics_table.set_fontsize(10)
            metrics_table.scale(1, 1.3)
            
            for (row, col), cell in metrics_table.get_celld().items():
                cell.set_edgecolor('gray')
                cell.set_linewidth(0.5)
                if row == 0:
                    cell.set_text_props(weight='bold')
                elif row % 2 == 1:
                    cell.set_facecolor('#f9f9f9')
                else:
                    cell.set_facecolor('white')
            
            params_ax = fig.add_axes([0.5, -0.045, 0.3, 0.12])
            params_ax.axis('off')
            
            params_ax.text(0.5, 1, 'Model Parameters', 
                          ha='center', va='center', fontsize=12)
            
            params_table = params_ax.table(
                cellText=param_rows,
                colLabels=param_headers,
                colColours=['#e0e0e0'] * 3,
                cellLoc='center',
                loc='upper center'
            )
            
            params_table.auto_set_font_size(False)
            params_table.set_fontsize(10)
            params_table.scale(1, 1.3)
            
            for (row, col), cell in params_table.get_celld().items():
                cell.set_edgecolor('gray')
                cell.set_linewidth(0.5)
                if row == 0:
                    cell.set_text_props(weight='bold')
                elif row % 2 == 1:
                    cell.set_facecolor('#f9f9f9')
                else:
                    cell.set_facecolor('white')
            
            plt.tight_layout(rect=[0,0,0.8,0.96]); plt.subplots_adjust(right=0.8)
            plt.suptitle(f'Trial {trial_idx} Model Comparison - AR {test_AR} (Conv1D {conv_status})', y=0.99)
            
            plot_dir = Path(output_dir) / 'detailed_ar_evaluations'
            plot_dir.mkdir(parents=True, exist_ok=True)
            conv_suffix = "conv1d" if config.get('use_temporal_conv', False) else "no_conv1d"
            out = plot_dir / f"Trial_{trial_idx:03d}_AR{test_AR}_{conv_suffix}_eval.png"
            plt.savefig(out, dpi=300, bbox_inches='tight')
            plt.close()
            
            img = Image.open(out)
            w, h = img.size
            cropped = img.crop((0, 0, w, h - 500))
            cropped.save(out)
            
            all_ar_results[test_AR] = {
                'plot_path': str(out),
                'metrics': all_tile_metrics,
                'use_temporal_conv': config.get('use_temporal_conv', True),
                'avg_metrics': {
                    'lstm_mae': mean_metric(all_tile_metrics, 'lstm', 'MAE'),
                    'transformer_mae': mean_metric(all_tile_metrics, 'transformer', 'MAE'),
                    'lstm_rmse': mean_metric(all_tile_metrics, 'lstm', 'RMSE'),
                    'transformer_rmse': mean_metric(all_tile_metrics, 'transformer', 'RMSE'),
                    'lstm_r2': mean_metric(all_tile_metrics, 'lstm', 'R2'),
                    'transformer_r2': mean_metric(all_tile_metrics, 'transformer', 'R2'),
                    'lstm_emergence_rmse': mean_metric(all_tile_metrics, 'lstm', 'emerg_RMSE'),        # ? ADD THIS
                    'transformer_emergence_rmse': mean_metric(all_tile_metrics, 'transformer', 'emerg_RMSE'),  # ? ADD THIS
                    'lstm_timing': mean_metric(all_tile_metrics, 'lstm', 'emergence_timing_diff'),
                    'transformer_timing': mean_metric(all_tile_metrics, 'transformer', 'emergence_timing_diff')
                }
            }
            successful_evaluations += 1
            print(f"    ? AR {test_AR} evaluation completed with Conv1D")
            
        except Exception as e:
            print(f"    ? AR {test_AR} evaluation failed: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"  Completed {successful_evaluations}/{len(test_ars)} AR evaluations with Conv1D")
    
    return all_ar_results

def run_attention_hyperparameter_search_eval_template(
    data_path,
    output_dir,
    lstm_path,
    max_trials=32,
    epochs=1000,
    use_temporal_conv=None
):
    """Hyperparameter search with Conv1D support"""
    
    # Define search space with Conv1D options
    # If use_temporal_conv is specified, use that value; otherwise search both
    if use_temporal_conv is not None:
        conv_values = [use_temporal_conv]
    else:
        conv_values = [False, True]  # Search both variants
    
    search_space = {
        'd_model': [128, 256],
        'nhead': [4],
        'num_layers': [3, 5],
        'dropout': [0.0, 0.3],
        'learning_rate': [1e-3, 1e-4],
        'output_len': [12],
        'use_temporal_conv': conv_values
    }
    
    keys = list(search_space.keys())
    values = list(search_space.values())
    
    all_combinations = []
    for combination in itertools.product(*values):
        config = dict(zip(keys, combination))
        if config['d_model'] % config['nhead'] != 0:
            continue
        all_combinations.append(config)
    
    if max_trials and len(all_combinations) > max_trials:
        np.random.seed(42)
        selected_indices = np.random.choice(len(all_combinations), max_trials, replace=False)
        all_combinations = [all_combinations[i] for i in selected_indices]
    
    # Determine Conv1D status message
    if use_temporal_conv is None:
        conv_msg = "Searching both variants (with and without Conv1D)"
    elif use_temporal_conv:
        conv_msg = "✅ ENABLED (with Conv1D)"
    else:
        conv_msg = "❌ DISABLED (pure transformer architecture)"
    
    print(f"Starting hyperparameter search for Baseline transformer")
    print(f"Will evaluate ALL 5 ARs [11698, 11726, 13165, 13179, 13183] for each trial")
    print(f"Training epochs: {epochs}")
    print(f"Conv1D testing: {conv_msg}")
    print(f"Total configurations: {len(all_combinations)}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output_dir = Path(output_dir).resolve()
    
    # Add suffix to output directory based on Conv1D configuration
    if use_temporal_conv is None:
        # Searching both variants - use base directory
        output_dir = output_dir
    elif use_temporal_conv:
        output_dir = output_dir / 'with_conv1d'
    else:
        output_dir = output_dir / 'no_conv1d'
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"✓ Output directory: {output_dir}")
    
    # Training data
    original_ars = [11130,11149,11158,11162,11199,11327,11344,11387,11393,11416,11422,11455,11619,11640,11660,11678,11682,11765,11768,11776,11916,11928,12036,12051,12085,12089,12144,12175,12203,12257,12331,12494,12659,12778,12864,12877,12900,12929,13004,13085,13098]
    train_ars = original_ars
    
    results = []
    best_score = float('inf') 
    best_config = None
    best_model_path = None
    best_trial_idx = None
    
    total_ar_evaluations = 0
    successful_ar_evaluations = 0
    
    for trial_idx, config in enumerate(all_combinations):
        print(f"\n{'='*100}")
        print(f"TRIAL {trial_idx + 1}/{len(all_combinations)} - Conv1D Testing")
        print(f"Config: {config}")
        print(f"Conv1D Enabled: {'?' if config['use_temporal_conv'] else '?'}")
        print(f"{'='*100}")
        
        try:
            start_time = time.time()
            
            all_inputs, all_intensities = load_ar_data_enhanced(
                train_ars, rid_of_top=1, size=9, num_in=128, num_pred=12, data_path=data_path
            )
            
            # Initialize model with Conv1D support
            model = SARTransformerLocalTile(
                input_dim=5,
                d_model=config['d_model'],
                nhead=config['nhead'],
                num_layers=config['num_layers'],
                dropout=config['dropout'],
                output_len=config['output_len'],
                max_seq_len=150,
                use_temporal_conv=config['use_temporal_conv']  # NEW: Conv1D control
            ).to(device)
            
            total_params = sum(p.numel() for p in model.parameters())
            conv_params = 0
            if config['use_temporal_conv']:
                conv_params = sum(p.numel() for name, p in model.named_parameters() 
                                if 'temporal_conv' in name or 'conv' in name.lower())
            
            print(f"Model parameters: {total_params:,} total")
            print(f"Conv1D parameters: {conv_params:,}")
            
            optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=1e-5)
            tiles = 63
            
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer, max_lr=config['learning_rate'], epochs=epochs, 
                steps_per_epoch=tiles, pct_start=0.1
            )
            
            best_loss = float('inf')
            current_model_path = output_dir / f'models/trial_{trial_idx:03d}/model.pth'
            current_model_path.parent.mkdir(parents=True, exist_ok=True)
            
            conv_status = "? ENABLED" if config['use_temporal_conv'] else "? DISABLED"
            print(f"Training Enhanced Transformer with Conv1D {conv_status}...")
            
            # Training loop
            for epoch in range(epochs):
                model.train()
                epoch_losses = []
                
                for tile in range(0, tiles, 8):
                    X_tile, y_tile = cross_ar_tile_data_preparation_attention(
                        tile, 9, all_inputs, all_intensities, 128, 12
                    )
                    
                    if len(X_tile) == 0:
                        continue
                    
                    X_tile = X_tile.to(device)
                    y_tile = y_tile.to(device)
                    
                    optimizer.zero_grad()
                    predictions = model(X_tile)
                    loss = emergence_aware_loss(predictions, y_tile, lambda_emergence=0.1)
                    
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    scheduler.step()
                    
                    epoch_losses.append(loss.item())
                
                avg_loss = np.mean(epoch_losses) if epoch_losses else float('inf')
                
                if epoch % 50 == 0:
                    print(f"Epoch {epoch}: loss={avg_loss:.6f}, lr={scheduler.get_last_lr()[0]:.6f}, Conv1D={conv_status}")
                
                if avg_loss < best_loss:
                    best_loss = avg_loss
                    torch.save(model.state_dict(), current_model_path)
            
            score = best_loss
            
            trial_result = {
                'trial_idx': trial_idx,
                'config': config,
                'score': score,
                'total_params': total_params,
                'conv1d_params': conv_params,
                'training_time': time.time() - start_time,
                'model_path': str(current_model_path),
                'training_method': 'enhanced_attention_with_conv1d'
            }
            
            results.append(trial_result)
            
            if score < best_score:
                best_score = score
                best_config = config
                best_model_path = str(current_model_path)
                best_trial_idx = trial_idx
                print(f"NEW BEST CONFIG! Score: {score:.6f}, Conv1D: {conv_status}")
            
            print(f"Trial {trial_idx + 1} completed. Score: {score:.6f}")
            
            # Evaluate ALL ARs
            print(f"Creating detailed evaluations for ALL ARs with Conv1D...")
            ar_results = evaluate_all_ars_eval_template(
                str(current_model_path), 
                config, 
                data_path, 
                lstm_path, 
                output_dir,
                trial_idx
            )
            
            trial_ar_evaluations = 5  # 5 ARs
            trial_successful_evaluations = len(ar_results)
            
            total_ar_evaluations += trial_ar_evaluations
            successful_ar_evaluations += trial_successful_evaluations
            
            trial_result['ar_evaluations'] = ar_results
            trial_result['ar_evaluation_stats'] = {
                'total_ars': trial_ar_evaluations,
                'successful_ars': trial_successful_evaluations,
                'success_rate': trial_successful_evaluations / trial_ar_evaluations if trial_ar_evaluations > 0 else 0
            }
            
            # Calculate average emergence metrics across all ARs
            def calculate_trial_avg_metrics(ar_results):
                all_lstm_rmse = []
                all_transformer_rmse = []
                all_lstm_emergence_rmse = []
                all_transformer_emergence_rmse = []
                all_lstm_timing = []
                all_transformer_timing = []
                
                for ar_data in ar_results.values():
                    if 'avg_metrics' in ar_data:
                        metrics = ar_data['avg_metrics']
                        if metrics.get('lstm_rmse') is not None:
                            all_lstm_rmse.append(metrics['lstm_rmse'])
                        if metrics.get('transformer_rmse') is not None:
                            all_transformer_rmse.append(metrics['transformer_rmse'])
                        if metrics.get('lstm_emergence_rmse') is not None:
                            all_lstm_emergence_rmse.append(metrics['lstm_emergence_rmse'])
                        if metrics.get('transformer_emergence_rmse') is not None:
                            all_transformer_emergence_rmse.append(metrics['transformer_emergence_rmse'])
                        if metrics.get('lstm_timing') is not None:
                            all_lstm_timing.append(metrics['lstm_timing'])
                        if metrics.get('transformer_timing') is not None:
                            all_transformer_timing.append(metrics['transformer_timing'])
                
                return {
                    'lstm_rmse': np.mean(all_lstm_rmse) if all_lstm_rmse else None,
                    'transformer_rmse': np.mean(all_transformer_rmse) if all_transformer_rmse else None,
                    'lstm_emergence_rmse': np.mean(all_lstm_emergence_rmse) if all_lstm_emergence_rmse else None,
                    'transformer_emergence_rmse': np.mean(all_transformer_emergence_rmse) if all_transformer_emergence_rmse else None,
                    'lstm_timing': np.mean(all_lstm_timing) if all_lstm_timing else None,
                    'transformer_timing': np.mean(all_transformer_timing) if all_transformer_timing else None
                }
            
            trial_result['avg_metrics'] = calculate_trial_avg_metrics(ar_results)
            
            print(f"? Trial {trial_idx} AR evaluations: {trial_successful_evaluations}/{trial_ar_evaluations} successful")
            
        except Exception as e:
            print(f"? Trial {trial_idx + 1} failed: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Save comprehensive results
    if results:
        results_df = pd.DataFrame([
            {
                'trial_idx': r['trial_idx'],
                'score': r['score'],
                'total_params': r['total_params'],
                'conv1d_params': r.get('conv1d_params', 0),
                'training_time': r['training_time'],
                'training_method': r['training_method'],
                'ar_success_rate': r.get('ar_evaluation_stats', {}).get('success_rate', 0),
                'total_ar_plots': len(r.get('ar_evaluations', {})),
                # Add emergence RMSE metrics
                'avg_lstm_rmse': r.get('avg_metrics', {}).get('lstm_rmse'),
                'avg_transformer_rmse': r.get('avg_metrics', {}).get('transformer_rmse'),
                'avg_lstm_emergence_rmse': r.get('avg_metrics', {}).get('lstm_emergence_rmse'),
                'avg_transformer_emergence_rmse': r.get('avg_metrics', {}).get('transformer_emergence_rmse'),
                'avg_lstm_timing_diff': r.get('avg_metrics', {}).get('lstm_timing'),
                'avg_transformer_timing_diff': r.get('avg_metrics', {}).get('transformer_timing'),
                **r['config']
            }
            for r in results
        ])
        
        results_df = results_df.sort_values('score')
        
        # Use appropriate filename based on Conv1D configuration
        if use_temporal_conv is None:
            csv_filename = 'baseline_search_results_ALL_ARS.csv'
            pkl_filename = 'baseline_detailed_results_ALL_ARS.pkl'
            json_filename = 'best_config_baseline_ALL_ARS.json'
        elif use_temporal_conv:
            csv_filename = 'baseline_conv1d_search_results_ALL_ARS.csv'
            pkl_filename = 'baseline_conv1d_detailed_results_ALL_ARS.pkl'
            json_filename = 'best_config_baseline_conv1d_ALL_ARS.json'
        else:
            csv_filename = 'baseline_no_conv1d_search_results_ALL_ARS.csv'
            pkl_filename = 'baseline_no_conv1d_detailed_results_ALL_ARS.pkl'
            json_filename = 'best_config_baseline_no_conv1d_ALL_ARS.json'
        
        csv_path = output_dir / csv_filename
        results_df.to_csv(csv_path, index=False)
        
        with open(output_dir / pkl_filename, 'wb') as f:
            pickle.dump({
                'results': results,
                'best_config': best_config,
                'best_score': best_score,
                'best_model_path': best_model_path,
                'best_trial_idx': best_trial_idx,
                'total_ar_evaluations': total_ar_evaluations,
                'successful_ar_evaluations': successful_ar_evaluations,
                'overall_ar_success_rate': successful_ar_evaluations / total_ar_evaluations if total_ar_evaluations > 0 else 0,
                'method': 'baseline_transformer',
                'evaluated_ars': [11698, 11726, 13165, 13179, 13183],
                'search_space': search_space
            }, f)
        
        with open(output_dir / json_filename, 'w') as f:
            json.dump({
                'best_config': best_config,
                'best_score': best_score,
                'best_trial_idx': best_trial_idx,
                'best_model_path': best_model_path,
                'total_trials': len(results),
                'total_ar_evaluations': total_ar_evaluations,
                'successful_ar_evaluations': successful_ar_evaluations,
                'overall_ar_success_rate': successful_ar_evaluations / total_ar_evaluations if total_ar_evaluations > 0 else 0,
                'method': 'baseline_transformer',
                'evaluated_ars': [11698, 11726, 13165, 13179, 13183],
                'conv1d_enabled': best_config.get('use_temporal_conv', True) if best_config else None,
                'search_completed': datetime.now().isoformat()
            }, f, indent=2)
    
    print(f"\n{'='*100}")
    print("BASELINE TRANSFORMER SEARCH COMPLETED!")
    print(f"{'='*100}")
    print(f"Best score: {best_score:.6f}")
    print(f"Best config: {best_config}")
    if best_config:
        print(f"Best Conv1D setting: {'? ENABLED' if best_config.get('use_temporal_conv', True) else '? DISABLED'}")
    print(f"Best trial: {best_trial_idx}")
    print(f"Total AR evaluations: {successful_ar_evaluations}/{total_ar_evaluations}")
    print(f"AR evaluation success rate: {successful_ar_evaluations/total_ar_evaluations*100:.1f}%")
    print(f"Results saved to: {output_dir}")
    print(f"Detailed AR plots: Available in {output_dir}/detailed_ar_evaluations/")
    
    return results, best_config, best_score

def set_deterministic_seeds(seed=42):
    """Set all random seeds for full reproducibility"""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # Enable deterministic algorithms
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Set environment variables for additional determinism
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    
    print(f"✅ Set deterministic seeds: {seed}")
    print(f"✅ Enabled deterministic algorithms")
    print(f"✅ Set cuDNN deterministic mode")

def main():
    parser = argparse.ArgumentParser(description='Baseline Transformer Training (supports both Conv1D variants)')
    parser.add_argument('--data_path', type=str, default=str(ROOT / 'data'), help='Data directory')
    parser.add_argument('--output_dir', type=str, default=str(ROOT / 'results' / 'baseline'), help='Output directory')
    parser.add_argument('--lstm_path', type=str, default=str(ROOT / 'models' / 'checkpoints' / 'lstm_baseline.pth'), help='LSTM model path for comparison')
    parser.add_argument('--max_trials', type=int, default=32, help='Maximum trials')
    parser.add_argument('--epochs', type=int, default=1000, help='Training epochs')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--use_temporal_conv', type=lambda x: (str(x).lower() == 'true'), default=None, 
                       help='Enable Conv1D front-end. If not specified, searches both variants. Use --use_temporal_conv=True or --use_temporal_conv=False')
    
    args = parser.parse_args()
    
    # Set deterministic seeds FIRST (as per README.md)
    set_deterministic_seeds(args.seed)
    
    # Determine Conv1D status
    if args.use_temporal_conv is None:
        conv_status = "Searching both variants"
    elif args.use_temporal_conv:
        conv_status = "With Conv1D"
    else:
        conv_status = "Without Conv1D"
    
    print("="*100)
    print(f"BASELINE TRANSFORMER HYPERPARAMETER SEARCH - {conv_status}")
    print("FOR ALL 5 ARs: [11698, 11726, 13165, 13179, 13183]")
    print("="*100)
    print("Features:")
    print("  • Enhanced transformer with relative positional encoding")
    print("  • Emergence-aware loss function")
    print("  • ALL 5 ARs evaluated for each trial")
    print(f"  • Conv1D: {conv_status}")
    print("  • Detailed emergence metrics and timing analysis")
    print(f"  • Output directory: {args.output_dir}")
    print("="*100)
    
    start_time = time.time()
    
    results, best_config, best_score = run_attention_hyperparameter_search_eval_template(
        args.data_path,
        args.output_dir,
        args.lstm_path,
        args.max_trials,
        args.epochs,
        args.use_temporal_conv
    )
    
    end_time = time.time()
    total_time = (end_time - start_time) / 3600
    
    print(f"\n{'='*100}")
    print("BASELINE TRANSFORMER SEARCH COMPLETED!")
    print(f"{'='*100}")
    print(f"Total time: {total_time:.2f} hours")
    print(f"Total trials: {len(results)}")
    print(f"Best score: {best_score:.6f}")
    if best_config:
        print(f"Best configuration:")
        for key, value in best_config.items():
            print(f"  {key}: {value}")
        conv_status = "✅ ENABLED" if best_config.get('use_temporal_conv', False) else "❌ DISABLED"
        print(f"  Conv1D Status: {conv_status}")
    print(f"Results directory: {args.output_dir}")

if __name__ == '__main__':
    main()