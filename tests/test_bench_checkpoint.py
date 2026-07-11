from pathlib import Path

import pytest

from cs336_basics.bench_checkpoint import _benchmark_arguments, build_parser


def test_minimal_checkpoint_arguments_use_repository_tokenizer_defaults():
    args = build_parser().parse_args(["--checkpoint", "model.pt", "--model-config", "model.json"])
    assert args.vocab == Path("data/vocab-16k.json")
    assert args.merges == Path("data/merges-16k.txt")
    benchmark_args = _benchmark_arguments(args, 12345)
    assert benchmark_args[:2] == ["--base-url", "http://127.0.0.1:12345"]
    assert "--request-rate" not in benchmark_args


def test_checkpoint_and_model_config_are_required():
    with pytest.raises(SystemExit):
        build_parser().parse_args([])
