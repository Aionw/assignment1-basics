from __future__ import annotations

import io

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from .adapters import (
    get_adamw_cls,
    run_cross_entropy,
    run_embedding,
    run_get_batch,
    run_get_lr_cosine_schedule,
    run_gradient_clipping,
    run_linear,
    run_load_checkpoint,
    run_multihead_self_attention,
    run_rope,
    run_save_checkpoint,
    run_scaled_dot_product_attention,
    run_softmax,
)


def test_linear_preserves_leading_dimensions_and_has_no_bias():
    weights = torch.tensor([[1.0, -2.0, 0.5], [0.0, 3.0, -1.0]])
    x = torch.arange(24, dtype=torch.float32).reshape(2, 4, 3)

    actual = run_linear(d_in=3, d_out=2, weights=weights, in_features=x)
    expected = x @ weights.T

    assert actual.shape == (2, 4, 2)
    torch.testing.assert_close(actual, expected)


def test_embedding_supports_multidimensional_token_ids():
    weights = torch.arange(30, dtype=torch.float32).reshape(10, 3)
    token_ids = torch.tensor([[0, 2, 4], [9, 1, 3]])

    actual = run_embedding(vocab_size=10, d_model=3, weights=weights, token_ids=token_ids)

    assert actual.shape == (2, 3, 3)
    torch.testing.assert_close(actual, weights[token_ids])


def test_softmax_handles_non_last_dimension_and_large_values():
    x = torch.tensor([[10000.0, 10001.0, 9999.0], [-10000.0, -9999.0, -10001.0]])

    actual = run_softmax(x, dim=0)
    expected = F.softmax(x, dim=0)

    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(actual.sum(dim=0), torch.ones(x.shape[1]))


def test_cross_entropy_is_invariant_to_logit_translation():
    logits = torch.tensor([[1.0, 2.0, 3.0], [5.0, -1.0, 0.25]])
    targets = torch.tensor([2, 0])

    base = run_cross_entropy(logits, targets)
    shifted = run_cross_entropy(logits + torch.tensor([[1000.0], [-500.0]]), targets)

    torch.testing.assert_close(base, shifted)


def test_gradient_clipping_leaves_small_gradients_unchanged_and_ignores_none():
    p1 = torch.nn.Parameter(torch.tensor([1.0, -2.0]))
    p2 = torch.nn.Parameter(torch.tensor([3.0]))
    p3 = torch.nn.Parameter(torch.tensor([4.0]))
    p1.grad = torch.tensor([0.03, 0.04])
    p2.grad = torch.tensor([0.01])
    p3.grad = None

    expected_p1_grad = p1.grad.clone()
    expected_p2_grad = p2.grad.clone()

    run_gradient_clipping([p1, p2, p3], max_l2_norm=1.0)

    torch.testing.assert_close(p1.grad, expected_p1_grad)
    torch.testing.assert_close(p2.grad, expected_p2_grad)
    assert p3.grad is None


def test_gradient_clipping_bounds_global_norm():
    params = [torch.nn.Parameter(torch.zeros(3)), torch.nn.Parameter(torch.zeros(2))]
    params[0].grad = torch.tensor([3.0, 4.0, 0.0])
    params[1].grad = torch.tensor([0.0, 12.0])

    run_gradient_clipping(params, max_l2_norm=5.0)

    total_norm = torch.linalg.vector_norm(torch.cat([p.grad.flatten() for p in params]))
    assert total_norm <= 5.0 + 1e-6


def test_get_batch_with_minimal_valid_dataset():
    dataset = np.arange(5)
    x, y = run_get_batch(dataset=dataset, batch_size=8, context_length=4, device="cpu")

    assert x.shape == (8, 4)
    assert y.shape == (8, 4)
    assert x.dtype == torch.long
    assert y.dtype == torch.long
    torch.testing.assert_close(x, torch.tensor([[0, 1, 2, 3]]).expand(8, 4))
    torch.testing.assert_close(y, torch.tensor([[1, 2, 3, 4]]).expand(8, 4))


@pytest.mark.parametrize(
    ("it", "expected"),
    [
        (0, 0.0),
        (2, 0.5),
        (4, 1.0),
        (8, 0.55),
        (12, 0.1),
        (20, 0.1),
    ],
)
def test_cosine_schedule_boundaries(it, expected):
    actual = run_get_lr_cosine_schedule(
        it=it,
        max_learning_rate=1.0,
        min_learning_rate=0.1,
        warmup_iters=4,
        cosine_cycle_iters=12,
    )

    assert actual == pytest.approx(expected)


def test_scaled_dot_product_attention_matches_pytorch_without_mask():
    torch.manual_seed(0)
    q = torch.randn(2, 3, 4)
    k = torch.randn(2, 5, 4)
    v = torch.randn(2, 5, 6)

    actual = run_scaled_dot_product_attention(q, k, v, mask=None)
    expected = F.scaled_dot_product_attention(q, k, v)

    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)


def test_scaled_dot_product_attention_respects_boolean_mask():
    q = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
    k = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
    v = torch.tensor([[[10.0, 0.0], [0.0, 20.0]]])
    mask = torch.tensor([[[True, False], [False, True]]])

    actual = run_scaled_dot_product_attention(q, k, v, mask=mask)

    torch.testing.assert_close(actual, v)


def test_rope_position_zero_is_identity_and_preserves_pair_norms():
    x = torch.randn(2, 3, 8)
    zero_positions = torch.zeros(2, 3, dtype=torch.long)

    identity = run_rope(d_k=8, theta=10000.0, max_seq_len=16, in_query_or_key=x, token_positions=zero_positions)
    rotated = run_rope(
        d_k=8,
        theta=10000.0,
        max_seq_len=16,
        in_query_or_key=x,
        token_positions=torch.tensor([[0, 1, 2], [3, 4, 5]]),
    )

    torch.testing.assert_close(identity, x)
    torch.testing.assert_close(
        torch.linalg.vector_norm(rotated.reshape(*rotated.shape[:-1], 4, 2), dim=-1),
        torch.linalg.vector_norm(x.reshape(*x.shape[:-1], 4, 2), dim=-1),
        atol=1e-5,
        rtol=1e-5,
    )


def test_multihead_self_attention_single_head_identity_weights_is_causal_sdpa():
    x = torch.tensor([[[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]])
    eye = torch.eye(2)

    actual = run_multihead_self_attention(
        d_model=2,
        num_heads=1,
        q_proj_weight=eye,
        k_proj_weight=eye,
        v_proj_weight=eye,
        o_proj_weight=eye,
        in_features=x,
    )
    expected = F.scaled_dot_product_attention(
        x,
        x,
        x,
        attn_mask=torch.ones(3, 3, dtype=torch.bool).tril(),
    )

    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)


def test_checkpoint_supports_file_like_objects():
    torch.manual_seed(0)
    model = torch.nn.Linear(3, 2)
    optimizer = get_adamw_cls()(model.parameters(), lr=1e-3)

    loss = model(torch.ones(3)).sum()
    loss.backward()
    optimizer.step()

    buffer = io.BytesIO()
    run_save_checkpoint(model, optimizer, iteration=17, out=buffer)
    buffer.seek(0)

    restored_model = torch.nn.Linear(3, 2)
    restored_optimizer = get_adamw_cls()(restored_model.parameters(), lr=1e-3)
    loaded_iteration = run_load_checkpoint(buffer, restored_model, restored_optimizer)

    assert loaded_iteration == 17
    for key, value in model.state_dict().items():
        torch.testing.assert_close(restored_model.state_dict()[key], value)
    assert restored_optimizer.state_dict()["param_groups"] == optimizer.state_dict()["param_groups"]
