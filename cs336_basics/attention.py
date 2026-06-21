from math import sqrt

import torch
from torch import nn, rsqrt

from cs336_basics.func import softmax


def scaled_dot_product_attention(
    queries: torch.Tensor, keys: torch.Tensor, values: torch.Tensor, mask: torch.Tensor
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
