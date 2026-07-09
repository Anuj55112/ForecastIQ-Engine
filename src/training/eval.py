import os
import argparse
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

from src.config import load_config
from src.data.processor import PyTorchForecastDataset
from src.models.patchtst import PatchTST
from src.models.nbeats import NBeats
from src.models.tft import TemporalFusionTransformer
from src.utils.logging import setup_logger

logger = setup_logger("eval", "eval.log")

def calculate_smape(actual: np.ndarray, forecast: np.ndarray) -> float:
    """Calculates symmetric Mean Absolute Percentage Error."""
    denom = (np.abs(actual) + np.abs(forecast)) / 2.0
    # Avoid division by zero
    mask = denom > 1e-5
    smape = np.mean(np.abs(actual[mask] - forecast[mask]) / denom[mask]) * 100
    return float(smape)

def evaluate_model(model_type: str, csv_path: str, checkpoint_path: str = ""):
    config = load_config()
    device = torch.device('cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'))
    
    # Initialize model
    if model_type == "patchtst":
        model = PatchTST(
            seq_len=config.ptst_seq_len,
            pred_len=config.ptst_pred_len,
            patch_len=config.ptst_patch_len,
            stride=config.ptst_stride,
            d_model=config.ptst_d_model,
            n_heads=config.ptst_n_heads,
            d_ff=config.ptst_d_ff,
            n_layers=config.ptst_n_layers
        )
    elif model_type == "nbeats":
        model = NBeats(
            seq_len=config.nb_seq_len,
            pred_len=config.nb_pred_len,
            num_stacks=config.nb_num_stacks,
            num_blocks=config.nb_num_blocks,
            width=config.nb_width
        )
    elif model_type == "tft":
        model = TemporalFusionTransformer(
            seq_len=config.tft_seq_len,
            pred_len=config.tft_pred_len,
            hidden_size=config.tft_hidden_size,
            num_heads=config.tft_num_heads
        )
    else:
        raise ValueError(f"Invalid model type: {model_type}")
        
    model.to(device)
    
    # Try loading checkpoint if provided
    if checkpoint_path and os.path.exists(checkpoint_path):
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint["model_state_dict"])
            logger.info(f"Loaded checkpoint from {checkpoint_path}")
        except Exception as e:
            logger.warning(f"Could not load checkpoint: {e}. Evaluating with randomized weights.")
    else:
        logger.info("No checkpoint provided or found. Evaluating with randomized weights.")
        
    model.eval()
    
    # Dataset
    val_dataset = PyTorchForecastDataset(
        csv_path=csv_path,
        seq_len=model.seq_len,
        pred_len=model.pred_len,
        is_train=False
    )
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)
    
    all_actuals = []
    all_forecasts = []
    
    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            batch_x = batch_x.to(device)
            pred_y = model(batch_x)
            
            all_actuals.append(batch_y.cpu().numpy())
            all_forecasts.append(pred_y.cpu().numpy())
            
    if not all_actuals:
        logger.error("No validation samples available.")
        return
        
    actuals = np.concatenate(all_actuals, axis=0)
    forecasts = np.concatenate(all_forecasts, axis=0)
    
    # Inverse scaling to report real-world metrics
    processor = val_dataset.processor
    actuals_unscaled = processor.inverse_transform(actuals)
    forecasts_unscaled = processor.inverse_transform(forecasts)
    
    # Compute metrics
    mae = float(np.mean(np.abs(actuals_unscaled - forecasts_unscaled)))
    rmse = float(np.sqrt(np.mean((actuals_unscaled - forecasts_unscaled) ** 2)))
    smape = calculate_smape(actuals_unscaled, forecasts_unscaled)
    
    # Print results
    print("\n" + "="*50)
    print(f"ForecastIQ Model Evaluation Report: {model_type.upper()}")
    print("="*50)
    print(f"Validation Windows: {len(val_dataset)}")
    print(f"MAE:                {mae:.4f}")
    print(f"RMSE:               {rmse:.4f}")
    print(f"sMAPE:              {smape:.2f}%")
    print("="*50 + "\n")
    
    return {
        "mae": mae,
        "rmse": rmse,
        "smape": smape
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate ForecastIQ Time Series Models")
    parser.add_argument("--model", type=str, default="patchtst", choices=["patchtst", "nbeats", "tft"], help="Model type to evaluate")
    parser.add_argument("--checkpoint", type=str, default="", help="Path to checkpoint file")
    args = parser.parse_args()
    
    csv_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data", "sample", "time_series.csv"))
    evaluate_model(args.model, csv_path, args.checkpoint)
