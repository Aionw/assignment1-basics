from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import torch

from cs336_basics.adamw import AdamW
from cs336_basics.func import cross_entropy, gradient_clipping, lr_cosine_schedule
from cs336_basics.train import get_batch, load_checkpoint, save_checkpoint
from cs336_basics.transformer import TransformerLM


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a Transformer language model on token-id datasets.")

    parser.add_argument("--train-data", type=Path, required=True, help="Path to a 1D token-id dataset.")
    parser.add_argument("--valid-data", type=Path, help="Optional path to a 1D validation token-id dataset.")
    parser.add_argument("--data-dtype", default="uint16", help="dtype for raw memmap files. Ignored for .npy files.")

    parser.add_argument("--vocab-size", type=int, required=True)
    parser.add_argument("--context-length", type=int, default=256)
    parser.add_argument("--d-model", type=int, default=512)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--num-heads", type=int, default=16)
    parser.add_argument("--d-ff", type=int, default=1344)
    parser.add_argument("--rope-theta", type=float, default=10_000.0)

    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-iters", type=int, default=10_000)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--min-lr", type=float, default=3e-5)
    parser.add_argument("--warmup-iters", type=int, default=500)
    parser.add_argument("--cosine-cycle-iters", type=int)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.95)
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--grad-clip", type=float, default=1.0)

    parser.add_argument("--eval-iters", type=int, default=20)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument("--checkpoint-every", type=int, default=1000)
    parser.add_argument("--checkpoint-path", type=Path, help="Where checkpoints should be written.")
    parser.add_argument("--resume-from", type=Path, help="Checkpoint path to resume from.")

    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--tensorboard-log-dir", type=Path, help="If set, write metrics for TensorBoard.")

    return parser.parse_args()


def load_token_array(path: Path, dtype: str) -> np.ndarray:
    if path.suffix == ".npy":
        data = np.load(path, mmap_mode="r")
    else:
        data = np.memmap(path, dtype=np.dtype(dtype), mode="r")
    if data.ndim != 1:
        raise ValueError(f"{path} must contain a 1D token-id array, got shape {data.shape}")
    return data


def set_optimizer_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = lr
        for param in group["params"]:
            if param in optimizer.state:
                optimizer.state[param]["lr"] = lr


def batch_loss(model: torch.nn.Module, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    logits = model(x)
    return cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1))


@torch.no_grad()
def estimate_loss(
    model: torch.nn.Module,
    dataset: np.ndarray,
    batch_size: int,
    context_length: int,
    device: str,
    eval_iters: int,
) -> float:
    model.eval()
    losses = []
    for _ in range(eval_iters):
        x, y = get_batch(dataset, batch_size, context_length, device)
        losses.append(batch_loss(model, x.long(), y.long()).item())
    model.train()
    return float(sum(losses) / len(losses))


def maybe_init_tensorboard(log_dir: Path | None):
    if log_dir is None:
        return None
    from torch.utils.tensorboard import SummaryWriter

    log_dir.mkdir(parents=True, exist_ok=True)
    return SummaryWriter(log_dir)


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    train_data = load_token_array(args.train_data, args.data_dtype)
    valid_data = load_token_array(args.valid_data, args.data_dtype) if args.valid_data else None
    if len(train_data) <= args.context_length:
        raise ValueError("training data must be longer than --context-length")
    if valid_data is not None and len(valid_data) <= args.context_length:
        raise ValueError("validation data must be longer than --context-length")

    model = TransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        rope_theta=args.rope_theta,
        device=args.device,
    )
    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(args.beta1, args.beta2),
        eps=args.eps,
    )

    start_iter = 0
    if args.resume_from:
        start_iter = load_checkpoint(args.resume_from, model, optimizer)
        print(f"resumed checkpoint from {args.resume_from} at iteration {start_iter}")

    writer = maybe_init_tensorboard(args.tensorboard_log_dir)
    cosine_cycle_iters = args.cosine_cycle_iters or args.num_iters

    model.train()
    for iteration in range(start_iter, args.num_iters):
        lr = lr_cosine_schedule(iteration, args.lr, args.min_lr, args.warmup_iters, cosine_cycle_iters)
        set_optimizer_lr(optimizer, lr)

        x, y = get_batch(train_data, args.batch_size, args.context_length, args.device)
        loss = batch_loss(model, x.long(), y.long())

        optimizer.zero_grad()
        loss.backward()
        if args.grad_clip > 0:
            gradient_clipping(model.parameters(), args.grad_clip)
        optimizer.step()

        step = iteration + 1
        metrics = {"train/loss": loss.item(), "train/perplexity": math.exp(min(loss.item(), 20)), "lr": lr}

        if step % args.eval_every == 0 and valid_data is not None:
            valid_loss = estimate_loss(
                model, valid_data, args.batch_size, args.context_length, args.device, args.eval_iters
            )
            metrics["valid/loss"] = valid_loss
            metrics["valid/perplexity"] = math.exp(min(valid_loss, 20))

        if step % args.log_every == 0 or step == 1 or "valid/loss" in metrics:
            fields = " ".join(f"{name}={value:.6g}" for name, value in metrics.items())
            print(f"iter={step} {fields}", flush=True)
        if writer is not None:
            for name, value in metrics.items():
                writer.add_scalar(name, value, step)

        if args.checkpoint_path and (step % args.checkpoint_every == 0 or step == args.num_iters):
            args.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            save_checkpoint(model, optimizer, step, args.checkpoint_path)
            print(f"saved checkpoint to {args.checkpoint_path} at iteration {step}", flush=True)

    if writer is not None:
        writer.close()


if __name__ == "__main__":
    main()
