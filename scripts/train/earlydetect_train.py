#!/usr/bin/env python3

import argparse
import itertools
import json
import os
import pickle
import random
import re
import time
from collections import OrderedDict
from datetime import datetime, timedelta
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from matplotlib import gridspec
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
SCRIPT_ROOT = ROOT / "scripts"

import sys

sys.path.extend(
    [
        str(SCRIPT_ROOT),
        str(SCRIPT_ROOT / "models"),
        str(SCRIPT_ROOT / "data"),
    ]
)

from early_detect_transformer import SARTransformerLocalTile
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


def early_detection_loss(
    predictions,
    targets,
    lambda_early=0.2,
    lambda_timing=0.3,
    lambda_emergence=0.1,
):
    """Enhanced loss that heavily penalizes late predictions and rewards early detection"""
    mse_loss = nn.MSELoss()(predictions, targets)
    
    # Early detection reward: penalize predictions that come after targets
    timing_penalty = torch.mean(torch.relu(predictions - targets))  # Only penalize over-prediction
    
    # Derivative-based early warning
    pred_derivatives = torch.gradient(predictions, dim=1)[0]
    target_derivatives = torch.gradient(targets, dim=1)[0]
    derivative_loss = nn.MSELoss()(pred_derivatives, target_derivatives)
    
    # Early signal amplification - reward when predictions lead targets
    early_signal_loss = torch.mean(torch.relu(target_derivatives - pred_derivatives))
    
    # Temporal consistency
    temporal_consistency = torch.mean(torch.abs(pred_derivatives[:, 1:] - pred_derivatives[:, :-1]))
    
    total_loss = mse_loss + lambda_early * early_signal_loss + lambda_timing * timing_penalty + lambda_emergence * derivative_loss + 0.01 * temporal_consistency
    return total_loss


def set_deterministic_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


def main():
    parser = argparse.ArgumentParser(
        description="EarlyDetect training (toggle Conv1D with --use_temporal_conv)"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default=str(ROOT / "data"),
        help="Data directory",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(ROOT / "results" / "earlydetect"),
        help="Output directory",
    )
    parser.add_argument(
        "--lstm_path",
        type=str,
        default=str(ROOT / "models" / "checkpoints" / "lstm_baseline.pth"),
        help="LSTM model path for comparison",
    )
    parser.add_argument("--max_trials", type=int, default=16, help="Maximum trials")
    parser.add_argument("--epochs", type=int, default=1000, help="Training epochs")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--use_temporal_conv",
        action="store_true",
        help="Enable Conv1D front-end (defaults to disabled)",
    )
    args = parser.parse_args()

    use_conv = args.use_temporal_conv
    set_deterministic_seeds(args.seed)

    print("=" * 100)
    print(f"EARLYDETECT TRAINING ({'with' if use_conv else 'without'} Conv1D)")
    print("ARs: [11698, 11726, 13165, 13179, 13183]")
    print(f"Output directory: {args.output_dir}")
    print("=" * 100)

    # ---------------------------
    # Load data (all ARs, central tiles)
    ARS = [11698, 11726, 13165, 13179, 13183]
    rid_of_top = 1
    size = 9
    num_in = 110
    num_pred = 12
    all_power_maps, all_intensities = load_ar_data_enhanced(
        ARS, rid_of_top, size, num_in, num_pred, args.data_path
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Simple config (can be tuned or replaced by best_config JSON)
    config = {
        "d_model": 256,
        "nhead": 8,
        "num_layers": 6,
        "dropout": 0.0,
        "output_len": num_pred,
        "use_temporal_conv": args.use_temporal_conv,
        "timing_bias_weight": 0.1,
        "early_detection_weight": 0.2,
    }

    model = SARTransformerLocalTile(
        input_dim=all_power_maps.shape[1],
        d_model=config["d_model"],
        nhead=config["nhead"],
        num_layers=config["num_layers"],
        dropout=config["dropout"],
        output_len=config["output_len"],
        max_seq_len=150,
        use_temporal_conv=config["use_temporal_conv"],
        timing_bias_weight=config["timing_bias_weight"],
        early_detection_weight=config["early_detection_weight"],
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=2e-4,
        steps_per_epoch=1,
        epochs=args.epochs,
        pct_start=0.3,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model.train()
    start_time = time.time()
    for epoch in range(args.epochs):
        epoch_loss = 0.0
        batches = 0
        for tile in range(all_power_maps.shape[2]):  # central row tiles already trimmed
            X_tile, y_tile = cross_ar_tile_data_preparation_attention(
                tile, size, all_power_maps, all_intensities, num_in, num_pred
            )
            if len(X_tile) == 0:
                continue
            X_tile = X_tile.to(device)
            y_tile = y_tile.to(device)

            optimizer.zero_grad()
            preds = model(X_tile)
            loss = early_detection_loss(preds, y_tile)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            batches += 1

        if batches > 0:
            epoch_loss /= batches
        else:
            epoch_loss = float("nan")

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"[Epoch {epoch+1}/{args.epochs}] loss={epoch_loss:.6f}")

    ckpt_path = output_dir / f"earlydetect_{'conv' if use_conv else 'no_conv'}.pth"
    torch.save(model.state_dict(), ckpt_path)
    print(f"Saved checkpoint to {ckpt_path}")
    print(f"Total time: {(time.time() - start_time)/60:.1f} min")


if __name__ == "__main__":
    main()

