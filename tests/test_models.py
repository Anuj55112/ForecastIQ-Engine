import pytest
import torch
from src.models.patchtst import PatchTST
from src.models.nbeats import NBeats
from src.models.tft import TemporalFusionTransformer

def test_patchtst_forward():
    model = PatchTST(
        seq_len=96,
        pred_len=24,
        patch_len=16,
        stride=8,
        d_model=32,
        n_heads=2,
        d_ff=64,
        n_layers=1
    )
    # [Batch, Seq_len, Features]
    dummy_x = torch.randn(2, 96, 1)
    dummy_y = model(dummy_x)
    assert dummy_y.shape == (2, 24, 1)

def test_nbeats_forward():
    model = NBeats(
        seq_len=96,
        pred_len=24,
        num_stacks=2,
        num_blocks=2,
        width=32
    )
    dummy_x = torch.randn(2, 96, 1)
    dummy_y = model(dummy_x)
    assert dummy_y.shape == (2, 24, 1)

def test_tft_forward():
    model = TemporalFusionTransformer(
        seq_len=96,
        pred_len=24,
        hidden_size=16,
        num_heads=2
    )
    dummy_x = torch.randn(2, 96, 1)
    dummy_y = model(dummy_x)
    assert dummy_y.shape == (2, 24, 1)
