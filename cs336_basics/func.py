from jaxtyping import Float, Int
import torch


def softmax(x: torch.Tensor, i: int) -> torch.Tensor:
    exp_x = torch.exp(x - torch.max(x, dim=i, keepdim=True).values)
    softmaxed = exp_x / torch.sum(exp_x, dim=i, keepdim=True)
    return softmaxed


def cross_entropy(
    inputs: Float[torch.Tensor, " batch_size vocab_size"], targets: Int[torch.Tensor, " batch_size"]
) -> Float[torch.Tensor, ""]:
    normalized = inputs - torch.max(inputs, dim=-1, keepdim=True).values
    target_values = normalized[torch.arange(normalized.shape[0], device=normalized.device), targets]
    return (torch.logsumexp(normalized, -1) - target_values).mean()
