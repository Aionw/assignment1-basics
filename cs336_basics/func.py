import torch


def softmax(x: torch.Tensor, i: int) -> torch.Tensor:
    exp_x = torch.exp(x - torch.max(x, dim=i, keepdim=True).values)
    softmaxed = exp_x / torch.sum(exp_x, dim=i, keepdim=True)
    return softmaxed
