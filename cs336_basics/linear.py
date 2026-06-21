import torch
from torch import nn
from torch import Tensor


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
