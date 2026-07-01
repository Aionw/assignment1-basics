from __future__ import annotations

import argparse
import json
import math
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import torch

from cs336_basics.adamw import AdamW
from cs336_basics.func import cross_entropy, gradient_clipping, lr_cosine_schedule
from cs336_basics.train import get_batch, load_checkpoint, save_checkpoint
from cs336_basics.transformer import TransformerLM


def parse_args() -> argparse.Namespace:
    defaults = {
        "data_dtype": "uint16",
        "context_length": 256,
        "d_model": 512,
        "num_layers": 4,
        "num_heads": 16,
        "d_ff": 1344,
        "rope_theta": 10_000.0,
        "batch_size": 32,
        "num_iters": 10_000,
        "lr": 3e-4,
        "min_lr": 3e-5,
        "warmup_iters": 500,
        "cosine_cycle_iters": None,
        "weight_decay": 0.1,
        "beta1": 0.9,
        "beta2": 0.95,
        "eps": 1e-8,
        "grad_clip": 1.0,
        "eval_iters": 20,
        "log_every": 100,
        "eval_every": 500,
        "checkpoint_every": 1000,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "seed": 1337,
        "compile": True,
        "precision": "bf16",
    }

    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--model-config", type=Path, help="JSON file with model hyperparameters.")
    config_parser.add_argument("--optimizer-config", type=Path, help="JSON file with optimizer hyperparameters.")
    config_parser.add_argument("--training-config", type=Path, help="JSON file with training loop settings.")
    config_args, remaining_args = config_parser.parse_known_args()
    config_defaults = load_config_defaults(config_args)

    parser = argparse.ArgumentParser(
        description="Train a Transformer language model on token-id datasets.",
        parents=[config_parser],
    )
    parser.set_defaults(**defaults)
    parser.set_defaults(**config_defaults)

    parser.add_argument("--train-data", type=Path, required=True, help="Path to a 1D token-id dataset.")
    parser.add_argument("--valid-data", type=Path, help="Optional path to a 1D validation token-id dataset.")
    parser.add_argument("--data-dtype", help="dtype for raw memmap files. Ignored for .npy files.")

    parser.add_argument("--vocab-size", type=int)
    parser.add_argument("--context-length", type=int)
    parser.add_argument("--d-model", type=int)
    parser.add_argument("--num-layers", type=int)
    parser.add_argument("--num-heads", type=int)
    parser.add_argument("--d-ff", type=int)
    parser.add_argument("--rope-theta", type=float)

    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--num-iters", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--min-lr", type=float)
    parser.add_argument("--warmup-iters", type=int)
    parser.add_argument("--cosine-cycle-iters", type=int)
    parser.add_argument("--weight-decay", type=float)
    parser.add_argument("--beta1", type=float)
    parser.add_argument("--beta2", type=float)
    parser.add_argument("--eps", type=float)
    parser.add_argument("--grad-clip", type=float)

    parser.add_argument("--eval-iters", type=int)
    parser.add_argument("--log-every", type=int)
    parser.add_argument("--eval-every", type=int)
    parser.add_argument("--checkpoint-every", type=int)
    parser.add_argument("--checkpoint-path", type=Path, help="Where checkpoints should be written.")
    parser.add_argument("--resume-from", type=Path, help="Checkpoint path to resume from.")

    parser.add_argument("--device")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--precision", choices=("fp32", "bf16", "fp16"))
    parser.add_argument("--compile", dest="compile", action="store_true", help="Compile the model with torch.compile.")
    parser.add_argument("--no-compile", dest="compile", action="store_false", help="Disable torch.compile.")
    parser.add_argument("--tensorboard-log-dir", type=Path, help="If set, write metrics for TensorBoard.")

    args = parser.parse_args(remaining_args)
    if args.vocab_size is None:
        parser.error("--vocab-size is required unless provided in --model-config")
    return args


def load_json_config(path: Path) -> dict[str, Any]:
    with path.open() as f:
        config = json.load(f)
    if not isinstance(config, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return config


def load_config_defaults(config_args: argparse.Namespace) -> dict[str, Any]:
    config_fields = {
        "model_config": {
            "vocab_size",
            "context_length",
            "d_model",
            "num_layers",
            "num_heads",
            "d_ff",
            "rope_theta",
        },
        "optimizer_config": {
            "lr",
            "min_lr",
            "warmup_iters",
            "cosine_cycle_iters",
            "weight_decay",
            "beta1",
            "beta2",
            "eps",
            "grad_clip",
        },
        "training_config": {
            "batch_size",
            "num_iters",
            "eval_iters",
            "log_every",
            "eval_every",
            "checkpoint_every",
            "data_dtype",
            "device",
            "seed",
            "compile",
            "precision",
        },
    }
    defaults = {}
    for attr, allowed_fields in config_fields.items():
        path = getattr(config_args, attr)
        if path is None:
            continue
        config = load_json_config(path)
        unknown_fields = set(config) - allowed_fields
        if unknown_fields:
            fields = ", ".join(sorted(unknown_fields))
            raise ValueError(f"{path} contains unsupported fields: {fields}")
        defaults.update(config)
    return defaults


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


def get_amp_dtype(precision: str) -> torch.dtype | None:
    if precision == "fp32":
        return None
    if precision == "bf16":
        return torch.bfloat16
    if precision == "fp16":
        return torch.float16
    raise ValueError(f"unsupported precision: {precision}")


def should_use_amp(device: str, precision: str) -> bool:
    return device.startswith("cuda") and precision != "fp32"


def batch_loss(
    model: torch.nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    amp_enabled: bool,
    amp_dtype: torch.dtype | None,
) -> torch.Tensor:
    context = torch.autocast(device_type="cuda", dtype=amp_dtype) if amp_enabled else nullcontext()
    with context:
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
    precision: str,
) -> float:
    model.eval()
    losses = []
    amp_enabled = should_use_amp(device, precision)
    amp_dtype = get_amp_dtype(precision)
    for _ in range(eval_iters):
        x, y = get_batch(dataset, batch_size, context_length, device)
        losses.append(batch_loss(model, x.long(), y.long(), amp_enabled, amp_dtype).item())
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

    raw_model = TransformerLM(
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
        raw_model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(args.beta1, args.beta2),
        eps=args.eps,
    )

    start_iter = 0
    if args.resume_from:
        start_iter = load_checkpoint(args.resume_from, raw_model, optimizer)
        print(f"resumed checkpoint from {args.resume_from} at iteration {start_iter}")

    model = torch.compile(raw_model) if args.compile else raw_model
    if args.compile:
        print("compiled model with torch.compile", flush=True)

    writer = maybe_init_tensorboard(args.tensorboard_log_dir)
    cosine_cycle_iters = args.cosine_cycle_iters or args.num_iters
    amp_enabled = should_use_amp(args.device, args.precision)
    amp_dtype = get_amp_dtype(args.precision)
    scaler = torch.amp.GradScaler("cuda", enabled=args.device.startswith("cuda") and args.precision == "fp16")
    print(f"training precision={args.precision}", flush=True)

    model.train()
    for iteration in range(start_iter, args.num_iters):
        lr = lr_cosine_schedule(iteration, args.lr, args.min_lr, args.warmup_iters, cosine_cycle_iters)
        set_optimizer_lr(optimizer, lr)

        x, y = get_batch(train_data, args.batch_size, args.context_length, args.device)
        loss = batch_loss(model, x.long(), y.long(), amp_enabled, amp_dtype)

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        if scaler.is_enabled():
            scaler.unscale_(optimizer)
        if args.grad_clip > 0:
            gradient_clipping(model.parameters(), args.grad_clip)
        scaler.step(optimizer)
        scaler.update()

        step = iteration + 1
        metrics = {"train/loss": loss.item(), "train/perplexity": math.exp(min(loss.item(), 20)), "lr": lr}

        if step % args.eval_every == 0 and valid_data is not None:
            valid_loss = estimate_loss(
                model, valid_data, args.batch_size, args.context_length, args.device, args.eval_iters, args.precision
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
            save_checkpoint(raw_model, optimizer, step, args.checkpoint_path)
            print(f"saved checkpoint to {args.checkpoint_path} at iteration {step}", flush=True)

    if writer is not None:
        writer.close()


if __name__ == "__main__":
    main()
