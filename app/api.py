import os
import torch
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query, Body
from pydantic import BaseModel
from typing import List, Dict, Any, Optional

# Package imports
from src.config import load_config
from src.data.processor import TimeSeriesProcessor
from src.models.patchtst import PatchTST
from src.models.nbeats import NBeats
from src.models.tft import TemporalFusionTransformer

app = FastAPI(
    title="ForecastIQ - Time Series Forecasting API",
    description="REST API for SOTA time-series forecasting comparing PatchTST, N-BEATS, and Temporal Fusion Transformers (TFT).",
    version="1.0.0"
)

# Load config
config = load_config()
device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))

# Global states
MODELS: Dict[str, Any] = {
    "patchtst": None,
    "nbeats": None,
    "tft": None
}
PROCESSOR = TimeSeriesProcessor(seq_len=96, pred_len=24)

# Input schemas
class PredictRequest(BaseModel):
    history: List[float] # List of 96 historical numbers
    model_type: str = "patchtst" # patchtst, nbeats, tft

def get_model(model_type: str):
    """Loads and caches forecasting models on demand."""
    if MODELS[model_type] is not None:
        return MODELS[model_type]
        
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
        checkpoint_path = os.path.join(config.checkpoint_dir, "checkpoint_patchtst.pth")
        if os.path.exists(checkpoint_path):
            try:
                checkpoint = torch.load(checkpoint_path, map_location=device)
                model.load_state_dict(checkpoint["model_state_dict"])
                print("Loaded PatchTST weights from checkpoint.")
            except Exception as e:
                print(f"Could not load PatchTST checkpoint: {e}")
        model.to(device)
        model.eval()
        MODELS["patchtst"] = model
        
    elif model_type == "nbeats":
        model = NBeats(
            seq_len=config.nb_seq_len,
            pred_len=config.nb_pred_len,
            num_stacks=config.nb_num_stacks,
            num_blocks=config.nb_num_blocks,
            width=config.nb_width
        )
        checkpoint_path = os.path.join(config.checkpoint_dir, "checkpoint_nbeats.pth")
        if os.path.exists(checkpoint_path):
            try:
                checkpoint = torch.load(checkpoint_path, map_location=device)
                model.load_state_dict(checkpoint["model_state_dict"])
                print("Loaded N-BEATS weights from checkpoint.")
            except Exception as e:
                print(f"Could not load N-BEATS checkpoint: {e}")
        model.to(device)
        model.eval()
        MODELS["nbeats"] = model
        
    elif model_type == "tft":
        model = TemporalFusionTransformer(
            seq_len=config.tft_seq_len,
            pred_len=config.tft_pred_len,
            hidden_size=config.tft_hidden_size,
            num_heads=config.tft_num_heads
        )
        checkpoint_path = os.path.join(config.checkpoint_dir, "checkpoint_tft.pth")
        if os.path.exists(checkpoint_path):
            try:
                checkpoint = torch.load(checkpoint_path, map_location=device)
                model.load_state_dict(checkpoint["model_state_dict"])
                print("Loaded TFT weights from checkpoint.")
            except Exception as e:
                print(f"Could not load TFT checkpoint: {e}")
        model.to(device)
        model.eval()
        MODELS["tft"] = model
        
    return MODELS[model_type]

@app.on_event("startup")
def startup_event():
    # Pre-warm default model
    get_model("patchtst")
    
    # Initialize processor with sample values to set default mean/std scale factors
    csv_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "sample", "time_series.csv"))
    if os.path.exists(csv_path):
        try:
            df = pd.read_csv(csv_path)
            vals = df["value"].values.astype(np.float32)
            PROCESSOR.fit_transform(vals)
        except Exception as e:
            print(f"Failed to fit scaler: {e}")

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "device": str(device),
        "patchtst_ready": MODELS["patchtst"] is not None,
        "nbeats_ready": MODELS["nbeats"] is not None,
        "tft_ready": MODELS["tft"] is not None
    }

@app.post("/predict")
def run_prediction(request: PredictRequest):
    if len(request.history) != 96:
        raise HTTPException(
            status_code=400, 
            detail=f"History length must be exactly 96. Provided length: {len(request.history)}"
        )
        
    if request.model_type not in ["patchtst", "nbeats", "tft"]:
        raise HTTPException(status_code=400, detail="Invalid model type. Choose: patchtst, nbeats, tft")
        
    try:
        model = get_model(request.model_type)
        
        # Scale inputs
        raw_arr = np.array(request.history, dtype=np.float32).reshape(-1, 1)
        scaled_arr = PROCESSOR.transform(raw_arr)
        
        # Format input tensor [Batch=1, Seq_len=96, Features=1]
        input_tensor = torch.from_numpy(scaled_arr).unsqueeze(0).to(device)
        
        with torch.no_grad():
            pred_tensor = model(input_tensor) # [1, Pred_len=24, 1]
            
        scaled_preds = pred_tensor.squeeze(0).cpu().numpy()
        
        # Inverse scale
        preds = PROCESSOR.inverse_transform(scaled_preds).squeeze(-1).tolist()
        return {
            "model_type": request.model_type,
            "forecast": preds
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference execution failed: {e}")

@app.get("/backtest")
def run_backtest(
    model_type: str = Query("patchtst", description="Model choice: patchtst, nbeats, tft")
):
    """Returns actual values vs predicted values for a validation segment of sample data."""
    csv_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "sample", "time_series.csv"))
    if not os.path.exists(csv_path):
        raise HTTPException(status_code=404, detail="Sample time series data file not found.")
        
    try:
        df = pd.read_csv(csv_path)
        vals = df["value"].values.astype(np.float32)
        
        # Use last 120 points for backtesting (96 input steps + 24 predicted steps)
        segment = vals[-120:]
        history = segment[:96]
        actuals = segment[96:]
        
        # Predict
        model = get_model(model_type)
        scaled_hist = PROCESSOR.transform(history.reshape(-1, 1))
        input_tensor = torch.from_numpy(scaled_hist).unsqueeze(0).to(device)
        
        with torch.no_grad():
            pred_tensor = model(input_tensor)
            
        scaled_preds = pred_tensor.squeeze(0).cpu().numpy()
        forecast = PROCESSOR.inverse_transform(scaled_preds).squeeze(-1).tolist()
        
        # Compute metrics
        actuals_np = np.array(actuals)
        forecast_np = np.array(forecast)
        mae = float(np.mean(np.abs(actuals_np - forecast_np)))
        rmse = float(np.sqrt(np.mean((actuals_np - forecast_np) ** 2)))
        
        return {
            "model_type": model_type,
            "history": history.tolist(),
            "actuals": actuals.tolist(),
            "forecast": forecast,
            "metrics": {
                "MAE": mae,
                "RMSE": rmse
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Backtest computation failed: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.api:app", host=config.api_host, port=config.api_port, reload=True)
