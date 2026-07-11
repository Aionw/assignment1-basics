"""Benchmark the OpenAI-compatible server implemented by :mod:`run_lm`."""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import random
import statistics
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from cs336_basics.run_lm import load_tokenizer


@dataclass(slots=True)
class RequestResult:
    success: bool
    latency_s: float
    input_tokens: int
    output_tokens: int = 0
    ttft_s: float | None = None
    itls_s: tuple[float, ...] = ()
    error: str | None = None

    @property
    def tpot_s(self) -> float | None:
        if self.output_tokens <= 1 or self.ttft_s is None:
            return None
        return (self.latency_s - self.ttft_s) / (self.output_tokens - 1)


def percentile(values: list[float], percent: float) -> float | None:
    if not values:
        return None
    values = sorted(values)
    rank = (len(values) - 1) * percent / 100
    lower, upper = math.floor(rank), math.ceil(rank)
    if lower == upper:
        return values[lower]
    return values[lower] + (values[upper] - values[lower]) * (rank - lower)


def _distribution(values_s: list[float]) -> dict[str, float | None]:
    values_ms = [value * 1000 for value in values_s]
    return {
        "mean_ms": statistics.fmean(values_ms) if values_ms else None,
        "p50_ms": percentile(values_ms, 50),
        "p90_ms": percentile(values_ms, 90),
        "p95_ms": percentile(values_ms, 95),
        "p99_ms": percentile(values_ms, 99),
    }


def summarize(results: list[RequestResult], duration_s: float) -> dict[str, Any]:
    completed = [result for result in results if result.success]
    input_tokens = sum(result.input_tokens for result in completed)
    output_tokens = sum(result.output_tokens for result in completed)
    return {
        "duration_s": duration_s,
        "submitted_requests": len(results),
        "completed_requests": len(completed),
        "failed_requests": len(results) - len(completed),
        "request_throughput_rps": len(completed) / duration_s if duration_s else 0.0,
        "input_throughput_tokens_per_s": input_tokens / duration_s if duration_s else 0.0,
        "output_throughput_tokens_per_s": output_tokens / duration_s if duration_s else 0.0,
        "total_throughput_tokens_per_s": (input_tokens + output_tokens) / duration_s if duration_s else 0.0,
        "total_input_tokens": input_tokens,
        "total_output_tokens": output_tokens,
        "latency": _distribution([result.latency_s for result in completed]),
        "ttft": _distribution([result.ttft_s for result in completed if result.ttft_s is not None]),
        "tpot": _distribution([result.tpot_s for result in completed if result.tpot_s is not None]),
        "itl": _distribution([itl for result in completed for itl in result.itls_s]),
    }


def _chunk_content(event: dict[str, Any]) -> str:
    choices = event.get("choices")
    if not isinstance(choices, list) or not choices:
        return ""
    choice = choices[0]
    if not isinstance(choice, dict):
        return ""
    delta = choice.get("delta")
    if isinstance(delta, dict) and isinstance(delta.get("content"), str):
        return delta["content"]
    return choice.get("text", "") if isinstance(choice.get("text"), str) else ""


def send_request(url: str, body: dict[str, Any], input_tokens: int, timeout_s: float) -> RequestResult:
    request = urllib.request.Request(
        url,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.perf_counter()
    first_token_at: float | None = None
    previous_token_at: float | None = None
    output_tokens = 0
    itls: list[float] = []
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            for raw_line in response:
                line = raw_line.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                if not _chunk_content(json.loads(data)):
                    continue
                now = time.perf_counter()
                if first_token_at is None:
                    first_token_at = now
                elif previous_token_at is not None:
                    itls.append(now - previous_token_at)
                previous_token_at = now
                output_tokens += 1
        ended = time.perf_counter()
        return RequestResult(
            success=True,
            latency_s=ended - started,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            ttft_s=None if first_token_at is None else first_token_at - started,
            itls_s=tuple(itls),
        )
    except (OSError, urllib.error.HTTPError, json.JSONDecodeError) as exc:
        return RequestResult(False, time.perf_counter() - started, input_tokens, error=str(exc))


async def run_benchmark(
    *,
    url: str,
    requests: list[tuple[dict[str, Any], int]],
    concurrency: int,
    request_rate: float,
    timeout_s: float,
    seed: int = 0,
) -> tuple[list[RequestResult], float]:
    semaphore = asyncio.Semaphore(concurrency)

    async def run_one(body: dict[str, Any], input_tokens: int) -> RequestResult:
        async with semaphore:
            return await asyncio.to_thread(send_request, url, body, input_tokens, timeout_s)

    started = time.perf_counter()
    tasks: list[asyncio.Task[RequestResult]] = []
    rng = random.Random(seed)
    for body, input_tokens in requests:
        tasks.append(asyncio.create_task(run_one(body, input_tokens)))
        if math.isfinite(request_rate):
            await asyncio.sleep(rng.expovariate(request_rate))
    results = await asyncio.gather(*tasks)
    return results, time.perf_counter() - started


def load_prompts(path: Path) -> list[str]:
    prompts: list[str] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        if path.suffix.lower() != ".jsonl":
            prompts.append(line)
            continue
        row = json.loads(line)
        prompt = row.get("prompt")
        if prompt is None and isinstance(row.get("messages"), list):
            prompt = "\n".join(
                str(message.get("content", "")) for message in row["messages"] if isinstance(message, dict)
            )
        if not isinstance(prompt, str):
            raise ValueError(f"{path}:{line_number}: expected a string 'prompt' or a 'messages' list")
        prompts.append(prompt)
    return prompts


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark the server started by run-lm")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default="tinystories-46m")
    parser.add_argument("--vocab", type=Path, required=True, help="same vocabulary passed to run-lm")
    parser.add_argument("--merges", type=Path, required=True, help="same merges passed to run-lm")
    parser.add_argument("--input-file", type=Path, help="text lines or JSONL prompt/messages rows")
    parser.add_argument("--prompt", default="Once upon a time")
    parser.add_argument("--num-prompts", type=int, default=100)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--request-rate", type=float, default=math.inf, help="Poisson arrival rate; default unlimited")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-json", type=Path)
    return parser


def _metric(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.2f}"


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.num_prompts <= 0 or args.concurrency <= 0 or args.request_rate <= 0 or args.warmup < 0:
        raise SystemExit("num-prompts, concurrency and request-rate must be positive; warmup cannot be negative")
    prompts = load_prompts(args.input_file) if args.input_file else [args.prompt]
    if not prompts:
        raise SystemExit("no prompts found")
    tokenizer = load_tokenizer(args.vocab, args.merges)
    url = args.base_url.rstrip("/") + "/v1/chat/completions"

    def make_request(prompt: str) -> tuple[dict[str, Any], int]:
        return (
            {
                "model": args.model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": args.max_tokens,
                "temperature": args.temperature,
                "top_p": args.top_p,
                "stream": True,
            },
            len(tokenizer.encode(prompt)),
        )

    if args.warmup:
        warmups = [make_request(prompts[index % len(prompts)]) for index in range(args.warmup)]
        warmup_results, _ = asyncio.run(
            run_benchmark(
                url=url,
                requests=warmups,
                concurrency=1,
                request_rate=math.inf,
                timeout_s=args.timeout,
                seed=args.seed,
            )
        )
        if any(not result.success for result in warmup_results):
            print(f"Warmup failed: {next(result.error for result in warmup_results if result.error)}", file=sys.stderr)
            return 1

    requests = [make_request(prompts[index % len(prompts)]) for index in range(args.num_prompts)]
    results, duration = asyncio.run(
        run_benchmark(
            url=url,
            requests=requests,
            concurrency=args.concurrency,
            request_rate=args.request_rate,
            timeout_s=args.timeout,
            seed=args.seed,
        )
    )
    report = summarize(results, duration)
    print("============ Serving Benchmark Result ============")
    print(f"Successful requests:       {report['completed_requests']}")
    print(f"Failed requests:           {report['failed_requests']}")
    print(f"Benchmark duration (s):    {duration:.2f}")
    print(f"Request throughput (req/s): {report['request_throughput_rps']:.2f}")
    print(f"Input token throughput:    {report['input_throughput_tokens_per_s']:.2f} tok/s")
    print(f"Output token throughput:   {report['output_throughput_tokens_per_s']:.2f} tok/s")
    for label, key in (("E2E latency", "latency"), ("TTFT", "ttft"), ("TPOT", "tpot"), ("ITL", "itl")):
        metric = report[key]
        print(
            f"{label} (ms): mean {_metric(metric['mean_ms'])}, p50 {_metric(metric['p50_ms'])}, "
            f"p90 {_metric(metric['p90_ms'])}, p99 {_metric(metric['p99_ms'])}"
        )
    if args.output_json:
        args.output_json.write_text(
            json.dumps({"summary": report, "results": [asdict(result) for result in results]}, indent=2),
            encoding="utf-8",
        )
    errors = [result.error for result in results if result.error]
    if errors:
        print(f"First error: {errors[0]}", file=sys.stderr)
    return int(bool(errors))


if __name__ == "__main__":
    raise SystemExit(main())
