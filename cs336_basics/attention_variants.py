import torch
from torch import Tensor, nn


__all__ = [
    "MultiHeadLatentAttention",
    "PAPER_URLS",
    "causal_mask",
    "compressed_sparse_attention",
    "deepseek_sparse_attention",
    "grouped_query_attention",
    "hybrid_cache_attention",
    "repeat_kv_heads",
    "selective_state_space_scan",
    "sliding_window_attention",
    "sliding_window_causal_mask",
]


PAPER_URLS = {
    "gqa": "https://arxiv.org/abs/2305.13245",
    "mla": "https://arxiv.org/abs/2405.04434",
    "swa": "https://arxiv.org/abs/2310.06825",
    "dsa": "https://arxiv.org/abs/2512.02556",
    "csa": "https://huggingface.co/deepseek-ai/DeepSeek-V4",
    "hca": "https://huggingface.co/deepseek-ai/DeepSeek-V4",
    "mamba": "https://arxiv.org/abs/2312.00752",
}


def repeat_kv_heads(x: Tensor, num_query_heads: int) -> Tensor:
    """GQA/MQA helper. Paper: https://arxiv.org/abs/2305.13245"""
    raise NotImplementedError


def grouped_query_attention(Q: Tensor, K: Tensor, V: Tensor, mask: Tensor | None = None) -> Tensor:
    """Grouped-Query Attention. Paper: https://arxiv.org/abs/2305.13245"""
    raise NotImplementedError


def causal_mask(seq_len: int, device: torch.device | None = None) -> Tensor:
    """Causal attention mask used by decoder-only attention."""
    raise NotImplementedError


def sliding_window_causal_mask(seq_len: int, window_size: int, device: torch.device | None = None) -> Tensor:
    """Sliding Window Attention mask. Paper: https://arxiv.org/abs/2310.06825"""
    raise NotImplementedError


def sliding_window_attention(Q: Tensor, K: Tensor, V: Tensor, window_size: int) -> Tensor:
    """Sliding Window Attention. Paper: https://arxiv.org/abs/2310.06825"""
    raise NotImplementedError


def deepseek_sparse_attention(Q: Tensor, K: Tensor, V: Tensor, top_k: int, mask: Tensor | None = None) -> Tensor:
    """DeepSeek Sparse Attention-style interface. Paper: https://arxiv.org/abs/2512.02556"""
    raise NotImplementedError


def compressed_sparse_attention(
    Q: Tensor,
    K: Tensor,
    V: Tensor,
    compressed_keys: Tensor,
    top_k: int,
    mask: Tensor | None = None,
) -> Tensor:
    """Compressed Sparse Attention. Paper: https://huggingface.co/deepseek-ai/DeepSeek-V4"""
    raise NotImplementedError


def hybrid_cache_attention(
    Q: Tensor,
    K: Tensor,
    V: Tensor,
    full_kv_cache: Tensor,
    compressed_kv_cache: Tensor,
    cache_budget: int,
    mask: Tensor | None = None,
) -> Tensor:
    """Hybrid Cache Attention. Paper: https://huggingface.co/deepseek-ai/DeepSeek-V4"""
    raise NotImplementedError


class MultiHeadLatentAttention(nn.Module):
    """Multi-head Latent Attention. Paper: https://arxiv.org/abs/2405.04434"""

    def __init__(self, d_model: int, num_heads: int, kv_lora_rank: int, device=None) -> None:
        super().__init__()
        raise NotImplementedError

    def forward(self, x: Tensor, mask: Tensor | None = None, return_latent_cache: bool = False) -> Tensor | tuple[Tensor, Tensor]:
        raise NotImplementedError


def selective_state_space_scan(x: Tensor, delta: Tensor, A: Tensor, B: Tensor, C: Tensor, D: Tensor) -> Tensor:
    """Mamba selective SSM scan. Paper: https://arxiv.org/abs/2312.00752"""
    raise NotImplementedError
