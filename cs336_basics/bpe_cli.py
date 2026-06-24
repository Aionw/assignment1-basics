from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from functools import lru_cache
from pathlib import Path


@lru_cache
def gpt2_bytes_to_unicode() -> dict[int, str]:
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(
        range(ord("®"), ord("ÿ") + 1)
    )
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    return dict(zip(bs, (chr(n) for n in cs)))


def gpt2_encode_token(token: bytes) -> str:
    byte_encoder = gpt2_bytes_to_unicode()
    return "".join(byte_encoder[b] for b in token)


def write_gpt2_vocab(vocab: dict[int, bytes], path: str | Path) -> None:
    encoded_vocab = {gpt2_encode_token(token): token_id for token_id, token in sorted(vocab.items())}
    with Path(path).open("w", encoding="utf-8") as f:
        json.dump(encoded_vocab, f, ensure_ascii=False, indent=2)
        f.write("\n")


def write_gpt2_merges(merges: Sequence[tuple[bytes, bytes]], path: str | Path) -> None:
    with Path(path).open("w", encoding="utf-8") as f:
        for left, right in merges:
            f.write(f"{gpt2_encode_token(left)} {gpt2_encode_token(right)}\n")


def train_bpe(
    input_path: str | Path,
    vocab_size: int,
    special_tokens: Sequence[str],
    threads: int,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    try:
        from cs336_basics import ctokenizer
    except ImportError:
        try:
            import ctokenizer
        except ImportError as top_level_import_error:
            raise ImportError(
                "Could not import ctokenizer. Reinstall/rebuild the package so the nanobind extension is installed "
                "as cs336_basics.ctokenizer."
            ) from top_level_import_error

    trainer = ctokenizer.BPETrainer(threads, str(input_path), list(special_tokens), vocab_size)
    trainer.train()
    return trainer.vocab(), trainer.merges()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a BPE tokenizer with ctokenizer.BPETrainer.")
    parser.add_argument("-i", "--input", required=True, help="Input corpus path.")
    parser.add_argument("-v", "--vocab-size", type=int, required=True, help="Target vocabulary size.")
    parser.add_argument("-t", "--threads", type=int, default=8, help="Number of trainer threads.")
    parser.add_argument("-s", "--special-token", action="append", default=[], help="Special token to add.")
    parser.add_argument("--vocab-output", default="vocab.json", help="Output GPT-2-style vocab JSON path.")
    parser.add_argument("--merges-output", default="merges.txt", help="Output GPT-2-style merges path.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    vocab, merges = train_bpe(
        input_path=args.input,
        vocab_size=args.vocab_size,
        special_tokens=args.special_token,
        threads=args.threads,
    )
    write_gpt2_vocab(vocab, args.vocab_output)
    write_gpt2_merges(merges, args.merges_output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
