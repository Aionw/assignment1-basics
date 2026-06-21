import torch
from torch import nn
from torch import Tensor


class RoPE(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None) -> None:
        super().__init__()
        thetas = 1 / (theta ** (torch.arange(0, d_k, 2) / d_k))
        poss = torch.arange(0, max_seq_len)
        freqs = torch.outer(poss, thetas)

        cos = torch.cos(freqs)
        sin = torch.sin(freqs)

        self.register_buffer("cos", cos, False)
        self.register_buffer("sin", sin, False)

    def forward(self, x: Tensor, token_positions: Tensor) -> Tensor:
        # x.shape [.., seq_len, d_k]
        # cos.shape [max_seq_len, d_k/2]
        cos = self.cos[token_positions]  # [token_pos.len, d_k/2]
        sin = self.sin[token_positions]

        x1 = x[..., 0::2]
        x2 = x[..., 1::2]

        new_x = cos * x1 - sin * x2
        new_y = sin * x1 + cos * x2
        out = torch.empty_like(x)
        out[..., 0::2] = new_x
        out[..., 1::2] = new_y
        return out

