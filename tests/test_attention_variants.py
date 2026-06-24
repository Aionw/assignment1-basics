import inspect

import pytest
import torch

from cs336_basics import attention_variants
from cs336_basics.attention_variants import MultiHeadLatentAttention


xfail_not_implemented = pytest.mark.xfail(raises=NotImplementedError, strict=False)


def test_attention_variant_paper_urls_are_recorded():
    assert attention_variants.PAPER_URLS == {
        "gqa": "https://arxiv.org/abs/2305.13245",
        "mla": "https://arxiv.org/abs/2405.04434",
        "swa": "https://arxiv.org/abs/2310.06825",
        "dsa": "https://arxiv.org/abs/2512.02556",
        "csa": "https://huggingface.co/deepseek-ai/DeepSeek-V4",
        "hca": "https://huggingface.co/deepseek-ai/DeepSeek-V4",
        "mamba": "https://arxiv.org/abs/2312.00752",
    }


def test_public_attention_variant_api_is_explicit():
    assert attention_variants.__all__ == [
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


def test_causal_mask_function_signature():
    assert list(inspect.signature(attention_variants.causal_mask).parameters) == [
        "seq_len",
        "device",
    ]


def test_gqa_function_signatures():
    assert list(inspect.signature(attention_variants.repeat_kv_heads).parameters) == [
        "x",
        "num_query_heads",
    ]
    assert list(inspect.signature(attention_variants.grouped_query_attention).parameters) == [
        "Q",
        "K",
        "V",
        "mask",
    ]


def test_swa_function_signatures():
    assert list(inspect.signature(attention_variants.sliding_window_causal_mask).parameters) == [
        "seq_len",
        "window_size",
        "device",
    ]
    assert list(inspect.signature(attention_variants.sliding_window_attention).parameters) == [
        "Q",
        "K",
        "V",
        "window_size",
    ]


def test_dsa_function_signature():
    assert list(inspect.signature(attention_variants.deepseek_sparse_attention).parameters) == [
        "Q",
        "K",
        "V",
        "top_k",
        "mask",
    ]


def test_csa_function_signature():
    assert list(inspect.signature(attention_variants.compressed_sparse_attention).parameters) == [
        "Q",
        "K",
        "V",
        "compressed_keys",
        "top_k",
        "mask",
    ]


def test_hca_function_signature():
    assert list(inspect.signature(attention_variants.hybrid_cache_attention).parameters) == [
        "Q",
        "K",
        "V",
        "full_kv_cache",
        "compressed_kv_cache",
        "cache_budget",
        "mask",
    ]


def test_mla_class_signatures():
    assert list(inspect.signature(MultiHeadLatentAttention).parameters) == [
        "d_model",
        "num_heads",
        "kv_lora_rank",
        "device",
    ]
    assert list(inspect.signature(MultiHeadLatentAttention.forward).parameters) == [
        "self",
        "x",
        "mask",
        "return_latent_cache",
    ]


def test_mamba_function_signature():
    assert list(inspect.signature(attention_variants.selective_state_space_scan).parameters) == [
        "x",
        "delta",
        "A",
        "B",
        "C",
        "D",
    ]


@xfail_not_implemented
def test_gqa_rejects_query_heads_not_divisible_by_kv_heads():
    q = torch.randn(1, 4, 2, 8)
    k = torch.randn(1, 3, 2, 8)
    v = torch.randn(1, 3, 2, 8)

    with pytest.raises(ValueError):
        attention_variants.grouped_query_attention(q, k, v)


@xfail_not_implemented
def test_gqa_handles_empty_sequence():
    q = torch.empty(1, 4, 0, 8)
    k = torch.empty(1, 2, 0, 8)
    v = torch.empty(1, 2, 0, 8)
    mask = torch.empty(0, 0, dtype=torch.bool)

    output = attention_variants.grouped_query_attention(q, k, v, mask)

    assert output.shape == q.shape


@xfail_not_implemented
def test_swa_rejects_non_positive_window_size():
    with pytest.raises(ValueError):
        attention_variants.sliding_window_causal_mask(seq_len=4, window_size=0)


@xfail_not_implemented
def test_swa_window_size_one_only_attends_to_current_token():
    q = torch.randn(1, 2, 3, 4)
    k = torch.randn(1, 2, 3, 4)
    v = torch.randn(1, 2, 3, 4)

    output = attention_variants.sliding_window_attention(q, k, v, window_size=1)

    torch.testing.assert_close(output, v)


@xfail_not_implemented
def test_dsa_rejects_non_positive_top_k():
    q = torch.randn(1, 2, 3, 4)
    k = torch.randn(1, 2, 3, 4)
    v = torch.randn(1, 2, 3, 4)

    with pytest.raises(ValueError):
        attention_variants.deepseek_sparse_attention(q, k, v, top_k=0)


@xfail_not_implemented
def test_dsa_allows_top_k_larger_than_available_keys():
    q = torch.randn(1, 2, 3, 4)
    k = torch.randn(1, 2, 3, 4)
    v = torch.randn(1, 2, 3, 4)

    output = attention_variants.deepseek_sparse_attention(q, k, v, top_k=99)

    assert output.shape == q.shape
    assert torch.isfinite(output).all()


@xfail_not_implemented
def test_csa_rejects_non_positive_top_k():
    q = torch.randn(1, 2, 3, 4)
    k = torch.randn(1, 2, 3, 4)
    v = torch.randn(1, 2, 3, 4)
    compressed_keys = torch.randn(1, 2, 3, 4)

    with pytest.raises(ValueError):
        attention_variants.compressed_sparse_attention(q, k, v, compressed_keys, top_k=0)


@xfail_not_implemented
def test_csa_allows_empty_compressed_keys():
    q = torch.randn(1, 2, 3, 4)
    k = torch.randn(1, 2, 3, 4)
    v = torch.randn(1, 2, 3, 4)
    compressed_keys = torch.empty(1, 2, 0, 4)

    output = attention_variants.compressed_sparse_attention(
        q,
        k,
        v,
        compressed_keys,
        top_k=1,
    )

    assert output.shape == q.shape


@xfail_not_implemented
def test_hca_rejects_negative_cache_budget():
    q = torch.randn(1, 2, 3, 4)
    k = torch.randn(1, 2, 3, 4)
    v = torch.randn(1, 2, 3, 4)
    full_kv_cache = torch.randn(1, 2, 3, 8)
    compressed_kv_cache = torch.randn(1, 2, 2, 8)

    with pytest.raises(ValueError):
        attention_variants.hybrid_cache_attention(
            q,
            k,
            v,
            full_kv_cache,
            compressed_kv_cache,
            cache_budget=-1,
        )


@xfail_not_implemented
def test_hca_allows_zero_full_cache_budget_with_compressed_cache():
    q = torch.randn(1, 2, 3, 4)
    k = torch.randn(1, 2, 3, 4)
    v = torch.randn(1, 2, 3, 4)
    full_kv_cache = torch.empty(1, 2, 0, 8)
    compressed_kv_cache = torch.randn(1, 2, 2, 8)

    output = attention_variants.hybrid_cache_attention(
        q,
        k,
        v,
        full_kv_cache,
        compressed_kv_cache,
        cache_budget=0,
    )

    assert output.shape == q.shape


@xfail_not_implemented
def test_mla_rejects_d_model_not_divisible_by_num_heads():
    with pytest.raises(ValueError):
        MultiHeadLatentAttention(d_model=10, num_heads=3, kv_lora_rank=4)


@xfail_not_implemented
def test_mla_can_return_latent_cache_for_empty_sequence():
    mla = MultiHeadLatentAttention(d_model=8, num_heads=2, kv_lora_rank=3)
    x = torch.empty(1, 0, 8)

    output, latent_cache = mla(x, return_latent_cache=True)

    assert output.shape == x.shape
    assert latent_cache.shape == (1, 0, 3)


@xfail_not_implemented
def test_mamba_scan_rejects_inconsistent_state_shapes():
    x = torch.randn(1, 3, 2)
    delta = torch.randn(1, 3, 2)
    a = torch.randn(3, 4)
    b = torch.randn(1, 3, 4)
    c = torch.randn(1, 3, 4)
    d = torch.randn(2)

    with pytest.raises(ValueError):
        attention_variants.selective_state_space_scan(x, delta, a, b, c, d)


@xfail_not_implemented
def test_mamba_scan_handles_empty_sequence():
    x = torch.empty(1, 0, 2)
    delta = torch.empty(1, 0, 2)
    a = torch.randn(2, 4)
    b = torch.empty(1, 0, 4)
    c = torch.empty(1, 0, 4)
    d = torch.randn(2)

    output = attention_variants.selective_state_space_scan(x, delta, a, b, c, d)

    assert output.shape == x.shape
