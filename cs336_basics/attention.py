from math import sqrt

import torch
from torch import nn

from cs336_basics.func import softmax
from cs336_basics.rope import RoPE
from cs336_basics.linear import Linear


def scaled_dot_product_attention(
    queries: torch.Tensor, keys: torch.Tensor, values: torch.Tensor, mask: torch.Tensor | None
) -> torch.Tensor:
    # keys: [batch_size, ..., seq_len, d_k]
    # queries: [batch_size, ..., seq_len, d_k]
    # values: [batch_size, ..., seq_len, d_v]
    # mask: [seq_len, seq_len]
    # output: [batch_size, ..., seq_len, d_k]

    d_k = keys.shape[-1]
    scores = queries @ keys.transpose(-1, -2)
    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))
    scores = scores / sqrt(d_k)
    return softmax(scores, -1) @ values


class MultiHeadeSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, device=None) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.wq = Linear(d_model, d_model, device=device)
        self.wk = Linear(d_model, d_model, device=device)
        self.wv = Linear(d_model, d_model, device=device)
        self.wo = Linear(self.head_dim * num_heads, d_model, device=device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.shape[-2]
        # x: [..., seq_len, num_head, head_dim ]
        Q = self.wq(x).unflatten(-1, (self.num_heads, self.head_dim))
        K = self.wk(x).unflatten(-1, (self.num_heads, self.head_dim))
        V = self.wv(x).unflatten(-1, (self.num_heads, self.head_dim))
        # [..., num_head, seq_len, head_dim]
        Q = Q.transpose(-2, -3)
        K = K.transpose(-2, -3)
        V = V.transpose(-2, -3)
        # No masking
        mask = torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool).tril()
        atten = scaled_dot_product_attention(Q, K, V, mask)
        atten = atten.transpose(-2, -3).contiguous().view(x.shape)
        return self.wo(atten)


class MultiHeadeSelfAttentionWithRoPE(nn.Module):
    def __init__(self, d_model: int, num_heads: int, theta: float, max_seq_len: int, device=None) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.wq = Linear(d_model, d_model, device=device)
        self.wk = Linear(d_model, d_model, device=device)
        self.wv = Linear(d_model, d_model, device=device)
        self.wo = Linear(self.head_dim * num_heads, d_model, device=device)
        self.rope = RoPE(theta, self.head_dim, max_seq_len, device)

    def forward(self, x: torch.Tensor, positions: torch.Tensor | None = None) -> torch.Tensor:
        seq_len = x.shape[-2]
        Q = self.wq(x).unflatten(-1, (self.num_heads, self.head_dim)).transpose(-2, -3)
        K = self.wk(x).unflatten(-1, (self.num_heads, self.head_dim)).transpose(-2, -3)
        V = self.wv(x).unflatten(-1, (self.num_heads, self.head_dim)).transpose(-2, -3)
        # [..., num_head, seq_len, head_dim]
        pos = positions if positions is not None else torch.arange(seq_len, device=x.device)
        Q = self.rope(Q, pos)
        K = self.rope(K, pos)
        mask = torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool).tril()
        # No masking
        atten = scaled_dot_product_attention(Q, K, V, mask)
        atten = atten.transpose(-2, -3).contiguous().view(x.shape)
        return self.wo(atten)
