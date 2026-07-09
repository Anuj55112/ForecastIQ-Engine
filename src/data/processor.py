import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Tuple, Dict, Optional

class TimeSeriesProcessor:
    """Handles scaling and rolling-window sequence formatting for time-series data."""
    def __init__(self, seq_len: int = 96, pred_len: int = 24):
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.mean = 0.0
        self.std = 1.0

    def fit_transform(self, data: np.ndarray) -> np.ndarray:
        self.mean = data.mean()
        self.std = data.std() if data.std() > 1e-5 else 1.0
        return (data - self.mean) / self.std

    def transform(self, data: np.ndarray) -> np.ndarray:
        return (data - self.mean) / self.std

    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        return (data * self.std) + self.mean

class PyTorchForecastDataset(Dataset):
    """
    Formulates a sequence forecasting dataset of shape:
    x: [seq_len, num_features] -> past window
    y: [pred_len, num_features] -> future target window
    """
    def __init__(
        self,
        csv_path: str,
        seq_len: int = 96,
        pred_len: int = 24,
        is_train: bool = True,
        val_split: float = 0.2,
        processor: Optional[TimeSeriesProcessor] = None
    ):
        self.seq_len = seq_len
        self.pred_len = pred_len
        
        # Load csv
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            # Take float columns (excluding timestamp)
            numeric_cols = [col for col in df.columns if col != "timestamp"]
            data_arr = df[numeric_cols].values.astype(np.float32)
        else:
            # Generate synthetic series if CSV is missing
            time = np.arange(500)
            series = 100 + 20 * np.sin(time / 12) + 0.1 * time
            data_arr = series[:, np.newaxis].astype(np.float32)

        # Scale data
        if processor is None:
            self.processor = TimeSeriesProcessor(seq_len, pred_len)
            self.data = self.processor.fit_transform(data_arr)
        else:
            self.processor = processor
            self.data = self.processor.transform(data_arr)

        # Train/Val split
        split_idx = int(len(self.data) * (1 - val_split))
        if is_train:
            self.data_slice = self.data[:split_idx]
        else:
            self.data_slice = self.data[split_idx:]
            
        # Ensure we have enough data
        self.num_samples = len(self.data_slice) - self.seq_len - self.pred_len + 1
        if self.num_samples <= 0:
            # Fallback if too short
            self.data_slice = self.data
            self.num_samples = len(self.data_slice) - self.seq_len - self.pred_len + 1

    def __len__(self) -> int:
        return max(self.num_samples, 0)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x_start = idx
        x_end = x_start + self.seq_len
        y_start = x_end
        y_end = y_start + self.pred_len
        
        x = self.data_slice[x_start:x_end]
        y = self.data_slice[y_start:y_end]
        
        return torch.from_numpy(x).float(), torch.from_numpy(y).float()
