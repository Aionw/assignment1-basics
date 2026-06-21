import torch
from torch import nn


class Embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None) -> None:
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight = nn.Parameter(
            nn.init.trunc_normal_(
                torch.empty(self.num_embeddings, self.embedding_dim, device=device, dtype=dtype),
                mean=0.0,
                std=1,
                a=-3,
                b=3,
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.weight[x]
