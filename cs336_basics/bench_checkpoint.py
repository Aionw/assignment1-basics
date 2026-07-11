"""Load a checkpoint, start a temporary server, and benchmark it in one command."""

from __future__ import annotations

import argparse
import socket
import threading
import time
from collections.abc import Sequence
from pathlib import Path

import torch
import uvicorn

from cs336_basics.bench_lm import main as benchmark_main
from cs336_basics.run_lm import LocalInference, create_app, load_model, load_tokenizer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Directly benchmark a local language-model checkpoint")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--model-config", type=Path, required=True)
    parser.add_argument("--vocab", type=Path, default=Path("data/vocab-16k.json"))
    parser.add_argument("--merges", type=Path, default=Path("data/merges-16k.txt"))
    parser.add_argument("--model-name", default="tinystories-46m")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--prompt", default="Once upon a time")
    parser.add_argument("--input-file", type=Path)
    parser.add_argument("--num-prompts", type=int, default=100)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--request-rate", type=float)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-json", type=Path)
    return parser


def _check_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise SystemExit(f"{label} not found: {path}")


def _benchmark_arguments(args: argparse.Namespace, port: int) -> list[str]:
    arguments = [
        "--base-url",
        f"http://127.0.0.1:{port}",
        "--model",
        args.model_name,
        "--vocab",
        str(args.vocab),
        "--merges",
        str(args.merges),
        "--prompt",
        args.prompt,
        "--num-prompts",
        str(args.num_prompts),
        "--max-tokens",
        str(args.max_tokens),
        "--concurrency",
        str(args.concurrency),
        "--temperature",
        str(args.temperature),
        "--top-p",
        str(args.top_p),
        "--warmup",
        str(args.warmup),
        "--timeout",
        str(args.timeout),
        "--seed",
        str(args.seed),
    ]
    for option, value in (
        ("--input-file", args.input_file),
        ("--request-rate", args.request_rate),
        ("--output-json", args.output_json),
    ):
        if value is not None:
            arguments.extend((option, str(value)))
    return arguments


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    for path, label in (
        (args.checkpoint, "checkpoint"),
        (args.model_config, "model config"),
        (args.vocab, "vocabulary"),
        (args.merges, "merges"),
    ):
        _check_file(path, label)

    print(f"Loading checkpoint {args.checkpoint} on {args.device} ...", flush=True)
    tokenizer = load_tokenizer(args.vocab, args.merges)
    model = load_model(args.model_config, args.checkpoint, args.device)
    engine = LocalInference(model, tokenizer, args.device, args.model_name)

    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind(("127.0.0.1", 0))
    listener.listen()
    port = listener.getsockname()[1]
    server = uvicorn.Server(
        uvicorn.Config(create_app(engine), log_level="warning", access_log=False)
    )
    thread = threading.Thread(target=server.run, kwargs={"sockets": [listener]}, daemon=True)
    thread.start()
    deadline = time.monotonic() + 30
    while not server.started and thread.is_alive() and time.monotonic() < deadline:
        time.sleep(0.01)
    if not server.started:
        server.should_exit = True
        thread.join(timeout=5)
        listener.close()
        raise SystemExit("temporary inference server failed to start")

    try:
        return benchmark_main(_benchmark_arguments(args, port))
    finally:
        server.should_exit = True
        thread.join(timeout=10)
        listener.close()


if __name__ == "__main__":
    raise SystemExit(main())
