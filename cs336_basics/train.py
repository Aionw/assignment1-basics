import numpy.typing as npt
import torch
import os
from typing import IO, Any, BinaryIO


def get_batch(
    dataset: npt.NDArray, batch_size: int, context_length: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    starts = torch.randint(0, len(dataset) - context_length, (batch_size,))
    sample = torch.stack([torch.tensor(dataset[i : i + context_length]) for i in starts])
    target = torch.stack([torch.tensor(dataset[i + 1 : i + 1 + context_length]) for i in starts])
    return sample.to(device), target.to(device)


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | BinaryIO | IO[bytes],
):
    checkpoint = {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "iteration": iteration}
    torch.save(checkpoint, out)


def load_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
):
    checkpoint = torch.load(src)
    iteration = checkpoint["iteration"]
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    return iteration
