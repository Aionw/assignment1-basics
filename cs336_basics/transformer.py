import torch

from torch import nn
from cs336_basics.rms_norm import RMSNorm
from cs336_basics.attention import MultiHeadeSelfAttentionWithRoPE
from cs336_basics.linear import SwiGLUFFN


class Transformer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, max_seq_len: int, theta: float, device=None) -> None:
        super().__init__()
        self.pre_atten_norm = RMSNorm(d_model=d_model, device=device)
        self.attn = MultiHeadeSelfAttentionWithRoPE(d_model, num_heads, theta, max_seq_len, device)
        self.pre_ffn_norm = RMSNorm(d_model=d_model, device=device)
        self.ffn = SwiGLUFFN(d_model, d_ff, device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: Batch, seq_len, d_model
        h = self.attn(self.pre_atten_norm(x)) + x
        return self.ffn(self.pre_ffn_norm(h)) + h
