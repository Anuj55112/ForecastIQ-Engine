import torch
import torch.nn as nn
from typing import Tuple

class NBeatsBlock(nn.Module):
    """A single N-BEATS block computing backcast and forecast projections."""
    def __init__(self, seq_len: int, pred_len: int, width: int = 128):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(seq_len, width),
            nn.ReLU(),
            nn.Linear(width, width),
            nn.ReLU(),
            nn.Linear(width, width),
            nn.ReLU()
        )
        self.backcast_head = nn.Linear(width, seq_len)
        self.forecast_head = nn.Linear(width, pred_len)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.fc(x)
        backcast = self.backcast_head(h)
        forecast = self.forecast_head(h)
        return backcast, forecast

class NBeats(nn.Module):
    """
    N-BEATS: Neural basis expansion analysis for interpretable time series forecasting.
    
    Ref: https://arxiv.org/abs/1905.10437
    """
    def __init__(
        self,
        seq_len: int = 96,
        pred_len: int = 24,
        num_stacks: int = 4,
        num_blocks: int = 3,
        width: int = 128
    ):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        
        # Build blocks list
        self.blocks = nn.ModuleList()
        for _ in range(num_stacks * num_blocks):
            self.blocks.append(NBeatsBlock(seq_len, pred_len, width))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor [Batch, Seq_Len, Features]
        Returns:
            Output tensor [Batch, Pred_Len, Features]
        """
        batch_size, seq_len, num_features = x.shape
        
        # Process channels independently
        x = x.transpose(1, 2).reshape(batch_size * num_features, seq_len)
        
        # Residual tracking
        backcast_residual = x
        forecast_accumulator = torch.zeros((batch_size * num_features, self.pred_len), device=x.device)
        
        for block in self.blocks:
            backcast, forecast = block(backcast_residual)
            backcast_residual = backcast_residual - backcast
            forecast_accumulator = forecast_accumulator + forecast
            
        # Reshape back to [Batch, Pred_Len, Features]
        output = forecast_accumulator.reshape(batch_size, num_features, self.pred_len).transpose(1, 2)
        return output
