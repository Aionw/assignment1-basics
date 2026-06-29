from jaxtyping import Float, Int
from collections.abc import Iterable
import torch
import math


def softmax(i: int) -> torch.Tensor:
    exp_x = torch.exp(x - torch.max(x, dim=i, keepdim=True).values)
    softmaxed = exp_x / torch.sum(exp_x, dim=i, keepdim=True)
    return softmaxed


def cross_entropy(
    inputs: Float[torch.Tensor, " batch_size vocab_size"], targets: Int[torch.Tensor, " batch_size"]
) -> Float[torch.Tensor, ""]:
    normalized = inputs - torch.max(inputs, dim=-1, keepdim=True).values
    target_values = normalized[torch.arange(normalized.shape[0], device=normalized.device), targets]
    return (torch.logsumexp(normalized, -1) - target_values).mean()


def lr_cosine_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
):
    if it < warmup_iters:
        return (it / warmup_iters) * max_learning_rate
    elif it <= cosine_cycle_iters:
        return (
            min_learning_rate
            + (1 + math.cos((it - warmup_iters) * math.pi / (cosine_cycle_iters - warmup_iters)))
            * (max_learning_rate - min_learning_rate)
            / 2
        )
    else:
        return min_learning_rate


def gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float, eps: float = 1e-6) -> None:
    grads = [p.grad for p in parameters if p.grad is not None]
    if not grads:
        return

    l2_norm = torch.linalg.vector_norm(torch.stack([torch.linalg.vector_norm(g.detach()) for g in grads]))
    if l2_norm >= max_l2_norm:
        scale = max_l2_norm / (l2_norm + eps)
        for g in grads:
            g.mul_(scale)
