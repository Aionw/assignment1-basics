import torch
from collections.abc import Callable
from typing import Optional
import math


class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, weight_decay=1, betas=(0.0, 0.0), eps=0.0):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr, "weight_decay": weight_decay, "betas": betas, "eps": eps}
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]  # Get the learning rate.
            weight_decay = group["weight_decay"]
            betas = group["betas"]
            eps = group["eps"]
            beta1, beta2 = betas[0], betas[1]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]  # Get state associated with p.
                g = p.grad.data  # NOTE: g is a tensor
                lr = state.get("lr", lr)
                t = state.get("t", 1)  # Get iteration number from the state, or 1.
                m = state.get("m", torch.zeros_like(p))
                v = state.get("v", torch.zeros_like(p))
                lr_t = lr * math.sqrt(1 - beta2**t) / (1 - beta1**t)
                m = beta1 * m + (1 - beta1) * g
                v = beta2 * v + (1 - beta2) * (g**2)
                p.data *= 1 - lr * weight_decay
                p.data -= lr_t * m / (torch.sqrt(v) + eps)
                state["lr"] = lr
                state["t"] = t + 1
                state["m"] = m
                state["v"] = v
        return loss
