import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path

def min_max_scaling(arr, min_val, max_val):
    """Min-max scale array safely."""
    if max_val == min_val:
        return np.zeros_like(arr)
    return (arr - min_val) / (max_val - min_val)

def attention_ready_transformer(tile, size, power_maps, intensities, num_in, num_pred):
    """Prepare overlapping sequences for transformer attention."""
    final_maps = np.transpose(power_maps, axes=(2, 1, 0))  # (time, features, tiles)
    final_ints = np.transpose(intensities, axes=(1, 0))     # (time, tiles)
    
    X_trans = final_maps[:, :, tile]  # (time, features)
    y_trans = final_ints[:, tile]     # (time,)
    
    # For attention transformer: create multiple overlapping sequences
    total_len = len(X_trans)
    
    if total_len < num_in + num_pred + 20:
        print(f"    Skipping tile {tile}: sequence too short ({total_len} < {num_in + num_pred + 20})")
        return torch.tensor([]), torch.tensor([])
    
    # Create overlapping sequences for attention learning
    step_size = max(1, (total_len - num_in - num_pred) // 8)  # 8 overlapping sequences
    
    X_sequences = []
    y_sequences = []
    
    for start_idx in range(0, total_len - num_in - num_pred + 1, step_size):
        end_input = start_idx + num_in
        end_target = end_input + num_pred
        
        input_seq = X_trans[start_idx:end_input]  # (num_in, features)
        target_seq = y_trans[end_input:end_target]  # (num_pred,)
        
        X_sequences.append(input_seq)
        y_sequences.append(target_seq)
    
    if len(X_sequences) > 0:
        X_stacked = torch.FloatTensor(np.stack(X_sequences, axis=0))  # (num_sequences, num_in, features)
        y_stacked = torch.FloatTensor(np.stack(y_sequences, axis=0))  # (num_sequences, num_pred)
        print(f"    Created {len(X_sequences)} attention sequences for tile {tile}")
        return X_stacked, y_stacked
    else:
        return torch.tensor([]), torch.tensor([])

def lstm_ready_transformer(tile, size, power_maps, intensities, num_in, num_pred):
    """
    LEGACY: Keep for backwards compatibility with basic transformer
    """
    # Same preprocessing as LSTM
    final_maps = np.transpose(power_maps, axes=(2, 1, 0))  # (time, features, tiles)
    final_ints = np.transpose(intensities, axes=(1, 0))     # (time, tiles)
    
    X_trans = final_maps[:, :, tile]  # (time, features)
    y_trans = final_ints[:, tile]     # (time,)
    
    # For basic transformer: use FULL sequence as input, predict future
    total_len = len(X_trans)
    print(f"    Sequence length for tile {tile}: {total_len}")
    
    if total_len < num_in + num_pred:
        print(f"    Skipping tile {tile}: sequence too short ({total_len} < {num_in + num_pred})")
        return torch.tensor([]), torch.tensor([])
    
    # Split: use first part as input, predict last part
    input_seq = X_trans[:total_len - num_pred]  # All but last num_pred
    target_seq = y_trans[total_len - num_pred:]  # Last num_pred steps
    
    print(f"    Input sequence shape: {input_seq.shape}, Target shape: {target_seq.shape}")
    
    # Return full sequences (not sliding windows)
    return torch.FloatTensor(input_seq), torch.FloatTensor(target_seq)

def load_ar_data_enhanced(ARs, rid_of_top, size, num_in, num_pred, data_path):
    """
    ENHANCED: Load all AR data optimized for attention transformer
    """
    all_inputs = []
    all_intensities = []
    max_seq_len = 0
    
    for AR in ARs:
        try:
            # Load data (same paths as LSTM)
            power_maps = np.load(f'{data_path}/AR{AR}/mean_pmdop{AR}_flat.npz', allow_pickle=True)
            mag_flux = np.load(f'{data_path}/AR{AR}/mean_mag{AR}_flat.npz', allow_pickle=True)
            intensities = np.load(f'{data_path}/AR{AR}/mean_int{AR}_flat.npz', allow_pickle=True)
            
            power_maps23 = power_maps['arr_0']
            power_maps34 = power_maps['arr_1']
            power_maps45 = power_maps['arr_2']
            power_maps56 = power_maps['arr_3']
            mag_flux_data = mag_flux['arr_0']
            intensities_data = intensities['arr_0']
            
            # Track sequence length before trimming
            original_seq_len = power_maps23.shape[1]
            max_seq_len = max(max_seq_len, original_seq_len)
            
            # Trim (same as LSTM)
            slice_idx = slice(rid_of_top*size, -rid_of_top*size)
            power_maps23 = power_maps23[slice_idx, :]
            power_maps34 = power_maps34[slice_idx, :]
            power_maps45 = power_maps45[slice_idx, :]
            power_maps56 = power_maps56[slice_idx, :]
            mag_flux_data = mag_flux_data[slice_idx, :]
            intensities_data = intensities_data[slice_idx, :]
            
            # Handle NaN
            mag_flux_data[np.isnan(mag_flux_data)] = 0
            intensities_data[np.isnan(intensities_data)] = 0
            
            # Stack and normalize PER-AR (same as LSTM)
            stacked_maps = np.stack([power_maps23, power_maps34, power_maps45, power_maps56], axis=1)
            stacked_maps[np.isnan(stacked_maps)] = 0
            
            # Per-AR normalization (IDENTICAL to LSTM)
            min_p, max_p = np.min(stacked_maps), np.max(stacked_maps)
            min_m, max_m = np.min(mag_flux_data), np.max(mag_flux_data)
            min_i, max_i = np.min(intensities_data), np.max(intensities_data)
            
            stacked_maps = min_max_scaling(stacked_maps, min_p, max_p)
            mag_flux_data = min_max_scaling(mag_flux_data, min_m, max_m)
            intensities_data = min_max_scaling(intensities_data, min_i, max_i)
            
            # Combine features (5 total: 4 power + 1 flux)
            mag_flux_reshaped = np.expand_dims(mag_flux_data, axis=1)
            inputs = np.concatenate([stacked_maps, mag_flux_reshaped], axis=1)
            
            all_inputs.append(inputs)
            all_intensities.append(intensities_data)
            
            print(f"Loaded AR {AR}: {inputs.shape}")
            
        except Exception as e:
            print(f"Failed to load AR {AR}: {e}")
            continue
    
    if len(all_inputs) == 0:
        raise ValueError("No ARs could be loaded!")
    
    # Handle variable sequence lengths by padding/truncating to max_seq_len
    # Find the actual maximum sequence length from loaded data
    actual_max_seq_len = max(inp.shape[2] for inp in all_inputs)
    
    # Pad or truncate all arrays to the same length
    padded_inputs = []
    padded_intensities = []
    
    for inp, ints in zip(all_inputs, all_intensities):
        current_seq_len = inp.shape[2]
        if current_seq_len < actual_max_seq_len:
            # Pad with zeros
            pad_width = actual_max_seq_len - current_seq_len
            inp_padded = np.pad(inp, ((0, 0), (0, 0), (0, pad_width)), mode='constant', constant_values=0)
            ints_padded = np.pad(ints, ((0, 0), (0, pad_width)), mode='constant', constant_values=0)
        elif current_seq_len > actual_max_seq_len:
            # Truncate
            inp_padded = inp[:, :, :actual_max_seq_len]
            ints_padded = ints[:, :actual_max_seq_len]
        else:
            inp_padded = inp
            ints_padded = ints
        
        padded_inputs.append(inp_padded)
        padded_intensities.append(ints_padded)
    
    all_inputs = np.stack(padded_inputs, axis=-1)
    all_intensities = np.stack(padded_intensities, axis=-1)
    
    print(f"Total data shape: inputs {all_inputs.shape}, intensities {all_intensities.shape}")
    print(f"Maximum sequence length found: {actual_max_seq_len}")
    
    return all_inputs, all_intensities

def cross_ar_tile_data_preparation_attention(tile, size, all_power_maps, all_intensities, num_in, num_pred):
    """
    ENHANCED: Prepare attention transformer data - overlapping sequences across all ARs
    """
    print(f"  Preparing attention data for Tile {tile} across all ARs...")
    
    X_list, y_list = [], []
    
    for ar_idx in range(all_power_maps.shape[-1]):
        power_maps = all_power_maps[:, :, :, ar_idx]
        intensities = all_intensities[:, :, ar_idx]
        
        try:
            X_ar, y_ar = attention_ready_transformer(tile, size, power_maps, intensities, num_in, num_pred)
            if len(X_ar) > 0:
                # X_ar is already (num_sequences, num_in, features)
                # y_ar is already (num_sequences, num_pred)
                X_list.append(X_ar)
                y_list.append(y_ar)
        except Exception as e:
            print(f"    Skipping AR {ar_idx}: {e}")
            continue
    
    if len(X_list) > 0:
        # Stack all sequences from all ARs
        X_tile = torch.cat(X_list, dim=0)  # (total_sequences, num_in, features)
        y_tile = torch.cat(y_list, dim=0)  # (total_sequences, num_pred)
        print(f"    Tile {tile}: {len(X_tile)} attention sequences from {len(X_list)} ARs")
    else:
        X_tile = torch.tensor([])
        y_tile = torch.tensor([])
    
    return X_tile, y_tile

def cross_ar_tile_data_preparation_fixed(tile, size, all_power_maps, all_intensities, num_in, num_pred):
    """
    LEGACY: Keep for backwards compatibility with basic transformer
    """
    print(f"  Preparing full sequence data for Tile {tile} across all ARs...")
    
    X_list, y_list = [], []
    
    for ar_idx in range(all_power_maps.shape[-1]):
        power_maps = all_power_maps[:, :, :, ar_idx]
        intensities = all_intensities[:, :, ar_idx]
        
        try:
            X_ar, y_ar = lstm_ready_transformer(tile, size, power_maps, intensities, num_in, num_pred)
            if len(X_ar) > 0:
                X_list.append(X_ar.unsqueeze(0))  # Add batch dimension
                y_list.append(y_ar.unsqueeze(0))
        except Exception as e:
            print(f"    Skipping AR {ar_idx}: {e}")
            continue
    
    if len(X_list) > 0:
        # Stack all sequences (each AR contributes one full sequence)
        X_tile = torch.cat(X_list, dim=0)  # (num_ars, seq_len, features)
        y_tile = torch.cat(y_list, dim=0)  # (num_ars, output_len)
        print(f"    Tile {tile}: {len(X_tile)} full sequences from {len(X_list)} ARs")
        print(f"    Max input sequence length: {max(x.size(1) for x in X_list) if X_list else 0}")
    else:
        X_tile = torch.tensor([])
        y_tile = torch.tensor([])
    
    return X_tile, y_tile
