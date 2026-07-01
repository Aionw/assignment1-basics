import torch
from torch import nn
from torch import Tensor
from torch.nn.functional import silu


class Linear(nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None) -> None:
        super().__init__()
        self.input_dimension = in_features
        self.output_dimension = out_features
        all_dim = self.input_dimension + self.output_dimension
        std = (2 / all_dim) ** 0.5
        self.weight = nn.Parameter(
            nn.init.trunc_normal_(
                torch.empty(self.output_dimension, self.input_dimension, device=device, dtype=dtype),
                mean=0.0,
                std=std,
                a=-3 * std,
                b=3 * std,
            )
        )

    def forward(self, x: Tensor) -> Tensor:
        return x @ self.weight.T


def sliu(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(x)


class SwiGLUFFN(nn.Module):
    def __init__(self, d_model: int, d_ff: int, device=None) -> None:
        super().__init__()
        self.l1 = Linear(d_model, d_ff, device=device)
        self.gate = Linear(d_model, d_ff, device=device)
        self.l2 = Linear(d_ff, d_model, device=device)

    def forward(self, x: Tensor) -> Tensor:
        return self.l2(silu(self.l1(x)) * self.gate(x))
