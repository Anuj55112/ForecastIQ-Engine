import pytest
import os
import numpy as np
from src.data.processor import TimeSeriesProcessor, PyTorchForecastDataset

def test_time_series_processor_scaling():
    processor = TimeSeriesProcessor(seq_len=10, pred_len=2)
    dummy_data = np.array([[10.0], [20.0], [30.0]], dtype=np.float32)
    
    scaled = processor.fit_transform(dummy_data)
    # Check mean normalization
    assert np.allclose(scaled.mean(), 0.0, atol=1e-5)
    
    # Check inverse transform recovers original values
    recovered = processor.inverse_transform(scaled)
    assert np.allclose(recovered, dummy_data)

def test_forecast_dataset_windowing():
    # Pass missing path to force synthetic generator fallback
    dataset = PyTorchForecastDataset(
        csv_path="nonexistent.csv",
        seq_len=96,
        pred_len=24,
        is_train=True
    )
    
    # Check that length is calculated correctly
    assert len(dataset) > 0
    
    # Retrieve item
    x, y = dataset[0]
    assert x.shape == (96, 1)
    assert y.shape == (24, 1)
