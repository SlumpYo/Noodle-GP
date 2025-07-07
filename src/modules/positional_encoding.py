import math
import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    """
    Implements the sinusoidal positional encoding from
    'Attention Is All You Need' (Vaswani et al., 2017).
    Adds a (max_len, d_model) matrix to embeddings.
    """

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        # Create a long enough PEs matrix once, then slice per input length.
        pe = torch.zeros(max_len, d_model)              
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float)
            * -(math.log(10000.0) / d_model)
        )
        # Even dims defined by sin, odd dims by cos
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        input is tensor of shape (S, B, D)
        output is tensor with positional encodings, same shape as input.
        """
        # input is (S, B, D)
        seq_len = x.size(0)
        # (S, 1, D) so it broadcasts over the batch dimension
        pos_emb = self.pe[:seq_len].unsqueeze(1)
        return x + pos_emb