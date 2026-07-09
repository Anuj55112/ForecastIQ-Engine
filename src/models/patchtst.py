import torch
import torch.nn as nn
import math
from typing import Optional

class PatchTST(nn.Module):
    """
    Patch Time Series Transformer.
    Splits time series sequences into patches to preserve local correlation
    and reduce self-attention computational complexity.
    
    Ref: https://arxiv.org/abs/2211.14731
    """
    def __init__(
        self,
        seq_len: int = 96,
        pred_len: int = 24,
        patch_len: int = 16,
        stride: int = 8,
        d_model: int = 128,
        n_heads: int = 4,
        d_ff: int = 256,
        n_layers: int = 3,
        dropout: float = 0.05
    ):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.patch_len = patch_len
        self.stride = stride
        
        # Calculate number of patches
        # num_patches = ceil((seq_len - patch_len) / stride) + 1
        self.num_patches = int(math.ceil((seq_len - patch_len) / stride)) + 1
        
        # Linear projection of patches
        self.patch_projection = nn.Linear(patch_len, d_model)
        
        # Positional Encoding
        self.pos_encoder = nn.Parameter(torch.randn(1, self.num_patches, d_model))
        
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # Output prediction head
        self.head = nn.Linear(self.num_patches * d_model, pred_len)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape [Batch, Seq_Len, Features]
        Returns:
            Output tensor of shape [Batch, Pred_Len, Features]
        """
        # Assume channel independence (process each feature channel independently)
        batch_size, seq_len, num_features = x.shape
        
        # Reshape to [Batch * Features, Seq_Len] to process channels in parallel
        x = x.transpose(1, 2).reshape(batch_size * num_features, seq_len)
        
        # 1. Patching
        patches = []
        for i in range(self.num_patches):
            start = i * self.stride
            end = start + self.patch_len
            if end <= seq_len:
                patch = x[:, start:end]
            else:
                # Zero padding if patch goes beyond sequence length
                patch = torch.zeros((x.shape[0], self.patch_len), device=x.device)
                valid_len = seq_len - start
                patch[:, :valid_len] = x[:, start:]
            patches.append(patch)
            
        # Stack patches: [Batch * Features, Num_Patches, Patch_Len]
        patches = torch.stack(patches, dim=1)
        
        # 2. Linear projection & Positional Encoding
        enc_out = self.patch_projection(patches) # [B*F, Num_Patches, d_model]
        enc_out = enc_out + self.pos_encoder
        enc_out = self.dropout(enc_out)
        
        # 3. Transformer Encoder pass
        enc_out = self.transformer_encoder(enc_out) # [B*F, Num_Patches, d_model]
        
        # 4. Flatten and Linear Prediction Head
        enc_out = enc_out.reshape(enc_out.shape[0], -1) # [B*F, Num_Patches * d_model]
        output = self.head(enc_out) # [B*F, Pred_Len]
        
        # Reshape back to [Batch, Pred_Len, Features]
        output = output.reshape(batch_size, num_features, self.pred_len).transpose(1, 2)
        
        return output
