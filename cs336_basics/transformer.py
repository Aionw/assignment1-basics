import torch

from torch import nn
from cs336_basics.rms_norm import RMSNorm
from cs336_basics.attention import MultiHeadeSelfAttentionWithRoPE
from cs336_basics.linear import SwiGLUFFN
from cs336_basics.embedding import Embedding
from cs336_basics.linear import Linear


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


class TransformerLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float,
        device=None,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.token_embeddings = Embedding(vocab_size, d_model, device=device)
        self.layers = nn.ModuleList(
            [
                Transformer(d_model, num_heads, d_ff, context_length, rope_theta, device=device)
                for _ in range(num_layers)
            ]
        )
        self.ln_final = RMSNorm(d_model, device=device)
        self.lm_head = Linear(d_model, vocab_size, device=device)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        if token_ids.shape[-1] > self.context_length:
            raise ValueError(
                f"input sequence length {token_ids.shape[-1]} exceeds context length {self.context_length}"
            )
        x = self.token_embeddings(token_ids)
        for layer in self.layers:
            x = layer(x)
        return self.lm_head(self.ln_final(x))
