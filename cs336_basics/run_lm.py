from __future__ import annotations

import argparse
import json
import threading
from collections.abc import Generator, Sequence
from pathlib import Path
from typing import Any

import torch
import uvicorn
from fastapi import FastAPI
from fastapi_openai_compat import CompletionResult, MessageParam, create_chat_completion_router

from cs336_basics.bpe_cli import gpt2_bytes_to_unicode
from cs336_basics.tokenizer import Tokenizer
from cs336_basics.transformer import TransformerLM


def load_tokenizer(vocab_path: Path, merges_path: Path) -> Tokenizer:
    byte_decoder = {char: byte for byte, char in gpt2_bytes_to_unicode().items()}
    with vocab_path.open(encoding="utf-8") as f:
        encoded_vocab = json.load(f)

    vocab = {
        token_id: bytes(byte_decoder[char] for char in token)
        for token, token_id in encoded_vocab.items()
    }
    merges: list[tuple[bytes, bytes]] = []
    with merges_path.open(encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split(" ")
            if len(parts) == 2:
                merges.append(tuple(bytes(byte_decoder[char] for char in part) for part in parts))
    return Tokenizer(vocab, merges, special_tokens=["<|endoftext|>"])


def load_model(config_path: Path, checkpoint_path: Path, device: str) -> TransformerLM:
    with config_path.open(encoding="utf-8") as f:
        config = json.load(f)
    model = TransformerLM(**config, device=device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    state_dict = checkpoint["model"] if "model" in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    model.eval()
    return model


def _message_text(message: MessageParam) -> str:
    content = message.get("content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            part.get("text", "") for part in content if isinstance(part, dict) and part.get("type") == "text"
        )
    return str(content or "")


class LocalInference:
    def __init__(self, model: TransformerLM, tokenizer: Tokenizer, device: str, model_name: str) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model_name = model_name
        self.lock = threading.Lock()
        self.eos_token_id = tokenizer.vocab_by_word.get(b"<|endoftext|>")

    @torch.inference_mode()
    def generate_tokens(
        self,
        prompt: str,
        max_tokens: int = 128,
        temperature: float = 0.8,
        top_p: float = 1.0,
    ) -> Generator[int, None, None]:
        token_ids = self.tokenizer.encode(prompt)
        if not token_ids:
            token_ids = [self.eos_token_id or 0]

        with self.lock:
            for _ in range(max_tokens):
                context = token_ids[-self.model.context_length :]
                inputs = torch.tensor([context], dtype=torch.long, device=self.device)
                logits = self.model(inputs)[0, -1].float()

                if temperature <= 0:
                    next_token = int(torch.argmax(logits))
                else:
                    probabilities = torch.softmax(logits / temperature, dim=-1)
                    if top_p < 1.0:
                        sorted_probs, sorted_indices = torch.sort(probabilities, descending=True)
                        cumulative = torch.cumsum(sorted_probs, dim=-1)
                        remove = cumulative - sorted_probs >= max(0.0, top_p)
                        sorted_probs[remove] = 0
                        sorted_probs /= sorted_probs.sum()
                        next_token = int(sorted_indices[torch.multinomial(sorted_probs, 1)])
                    else:
                        next_token = int(torch.multinomial(probabilities, 1))

                if next_token == self.eos_token_id:
                    return
                token_ids.append(next_token)
                yield next_token

    def complete(
        self, requested_model: str, messages: list[MessageParam], body: dict[str, Any]
    ) -> CompletionResult:
        del requested_model
        prompt = "\n".join(filter(None, (_message_text(message) for message in messages)))
        tokens = self.generate_tokens(
            prompt,
            max_tokens=int(body.get("max_tokens") or 128),
            temperature=float(body.get("temperature") if body.get("temperature") is not None else 0.8),
            top_p=float(body.get("top_p") if body.get("top_p") is not None else 1.0),
        )
        if body.get("stream", False):
            return (self.tokenizer.decode([token_id]) for token_id in tokens)
        return self.tokenizer.decode(list(tokens))


def create_app(engine: LocalInference) -> FastAPI:
    app = FastAPI(title="CS336 local inference")
    app.include_router(
        create_chat_completion_router(
            list_models=lambda: [engine.model_name],
            run_completion=engine.complete,
        )
    )
    return app


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Serve a local checkpoint through an OpenAI-compatible API.")
    parser.add_argument("--model-config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--vocab", type=Path, required=True)
    parser.add_argument("--merges", type=Path, required=True)
    parser.add_argument("--model-name", default="tinystories-46m")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    tokenizer = load_tokenizer(args.vocab, args.merges)
    model = load_model(args.model_config, args.checkpoint, args.device)
    engine = LocalInference(model, tokenizer, args.device, args.model_name)
    uvicorn.run(create_app(engine), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
