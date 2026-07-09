import torch
import torch.nn as nn
import torch.nn.functional as F

class GatedLinearUnit(nn.Module):
    """GLU component for controlling information flow via gating mechanisms."""
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.fc = nn.Linear(input_dim, output_dim * 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_fc = self.fc(x)
        val, gate = torch.chunk(x_fc, 2, dim=-1)
        return val * torch.sigmoid(gate)

class GatedResidualNetwork(nn.Module):
    """GRN component for adaptive model capacity and feature mapping."""
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, dropout: float = 0.05):
        super().__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, output_dim)
        self.glu = GatedLinearUnit(output_dim, output_dim)
        self.gate_norm = nn.LayerNorm(output_dim)
        
        # Skip connection projection if input and output dimensions differ
        self.skip_project = nn.Linear(input_dim, output_dim) if input_dim != output_dim else nn.Identity()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.elu(self.linear1(x))
        h = self.linear2(h)
        h = self.dropout(h)
        # GLU gating
        gated = self.glu(h)
        # Skip connection + layer norm
        out = self.gate_norm(self.skip_project(x) + gated)
        return out

class TemporalFusionTransformer(nn.Module):
    """
    Lightweight PyTorch implementation of Temporal Fusion Transformer (TFT).
    Utilizes GRNs and Multi-Head Attention to manage temporal dependencies.
    
    Ref: https://arxiv.org/abs/1912.09363
    """
    def __init__(
        self,
        seq_len: int = 96,
        pred_len: int = 24,
        hidden_size: int = 64,
        num_heads: int = 4
    ):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.hidden_size = hidden_size
        
        # 1. Processing of feature dimensions
        self.input_grn = GatedResidualNetwork(1, hidden_size, hidden_size)
        
        # 2. Positional Encoder
        self.pos_encoder = nn.Parameter(torch.randn(1, seq_len, hidden_size))
        
        # 3. Multi-head self-attention over history
        self.attn = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=num_heads, batch_first=True)
        self.post_attn_grn = GatedResidualNetwork(hidden_size, hidden_size, hidden_size)
        
        # 4. Decoder mapping to forecast horizon
        self.decoder_grn = GatedResidualNetwork(seq_len * hidden_size, hidden_size * 2, hidden_size)
        self.output_head = nn.Linear(hidden_size, pred_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape [Batch, Seq_Len, Features]
        Returns:
            Output forecast tensor of shape [Batch, Pred_Len, Features]
        """
        batch_size, seq_len, num_features = x.shape
        
        # Process each feature channel in parallel
        # Reshape to [Batch * Features, Seq_Len, 1]
        x_flat = x.transpose(1, 2).reshape(batch_size * num_features, seq_len, 1)
        
        # 1. Input gate residual mapping
        h = self.input_grn(x_flat) # [B*F, Seq_Len, hidden_size]
        h = h + self.pos_encoder
        
        # 2. Temporal Self-Attention
        attn_out, _ = self.attn(h, h, h) # [B*F, Seq_Len, hidden_size]
        attn_out = self.post_attn_grn(attn_out)
        
        # 3. Flatten and Decode
        flat_attn = attn_out.reshape(batch_size * num_features, -1) # [B*F, Seq_Len * hidden_size]
        decoded = self.decoder_grn(flat_attn) # [B*F, hidden_size]
        
        # 4. Linear project output
        output_flat = self.output_head(decoded) # [B*F, Pred_Len]
        
        # Reshape back to [Batch, Pred_Len, Features]
        output = output_flat.reshape(batch_size, num_features, self.pred_len).transpose(1, 2)
        return output
