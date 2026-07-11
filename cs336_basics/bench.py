"""Small, dependency-free benchmark client for OpenAI-compatible inference servers."""

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
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class RequestResult:
    success: bool
    latency_s: float
    ttft_s: float | None = None
    output_tokens: int = 0
    input_tokens: int = 0
    inter_token_latencies_s: tuple[float, ...] = ()
    error: str | None = None


def percentile(values: list[float], p: float) -> float | None:
    """Return a linearly interpolated percentile (p is in [0, 100])."""
    if not values:
        return None
    ordered = sorted(values)
    rank = (len(ordered) - 1) * p / 100
    lo, hi = math.floor(rank), math.ceil(rank)
    if lo == hi:
        return ordered[lo]
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (rank - lo)


def summarize(results: list[RequestResult], duration_s: float) -> dict[str, Any]:
    good = [r for r in results if r.success]
    latencies = [r.latency_s * 1000 for r in good]
    ttfts = [r.ttft_s * 1000 for r in good if r.ttft_s is not None]
    itls = [x * 1000 for r in good for x in r.inter_token_latencies_s]

    def distribution(xs: list[float]) -> dict[str, float | None]:
        return {
            "mean_ms": statistics.fmean(xs) if xs else None,
            "p50_ms": percentile(xs, 50),
            "p90_ms": percentile(xs, 90),
            "p95_ms": percentile(xs, 95),
            "p99_ms": percentile(xs, 99),
        }

    output_tokens = sum(r.output_tokens for r in good)
    return {
        "duration_s": duration_s,
        "requests": len(results),
        "successful_requests": len(good),
        "failed_requests": len(results) - len(good),
        "request_throughput_rps": len(good) / duration_s if duration_s else 0.0,
        "output_throughput_tokens_per_s": output_tokens / duration_s if duration_s else 0.0,
        "total_input_tokens": sum(r.input_tokens for r in good),
        "total_output_tokens": output_tokens,
        "latency": distribution(latencies),
        "ttft": distribution(ttfts),
        "itl": distribution(itls),
    }


def _event_token_count(event: dict[str, Any]) -> int:
    usage = event.get("usage") or {}
    if isinstance(usage.get("completion_tokens"), int):
        return 0  # Usage is cumulative and handled after streaming.
    choices = event.get("choices") or []
    if not choices:
        return 0
    choice = choices[0]
    text = choice.get("text")
    if text is None:
        text = (choice.get("delta") or {}).get("content")
    return int(bool(text))


def send_request(
    url: str, payload: dict[str, Any], headers: dict[str, str], timeout_s: float
) -> RequestResult:
    """Send one streaming request and collect timings. Intended for a worker thread."""
    body = json.dumps(payload).encode()
    request = urllib.request.Request(url, body, {"Content-Type": "application/json", **headers}, method="POST")
    started = time.perf_counter()
    first_token_at: float | None = None
    previous_token_at: float | None = None
    itls: list[float] = []
    output_tokens = 0
    input_tokens = 0
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            if payload.get("stream", True):
                for raw_line in response:
                    line = raw_line.decode("utf-8", "replace").strip()
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    event = json.loads(data)
                    usage = event.get("usage") or {}
                    input_tokens = usage.get("prompt_tokens", input_tokens)
                    if isinstance(usage.get("completion_tokens"), int):
                        output_tokens = max(output_tokens, usage["completion_tokens"])
                    emitted = _event_token_count(event)
                    if emitted:
                        now = time.perf_counter()
                        if first_token_at is None:
                            first_token_at = now
                        elif previous_token_at is not None:
                            itls.append(now - previous_token_at)
                        previous_token_at = now
                        output_tokens += emitted
            else:
                event = json.loads(response.read())
                usage = event.get("usage") or {}
                input_tokens = usage.get("prompt_tokens", 0)
                output_tokens = usage.get("completion_tokens", 0)
                first_token_at = time.perf_counter()
        ended = time.perf_counter()
        return RequestResult(
            success=True,
            latency_s=ended - started,
            ttft_s=None if first_token_at is None else first_token_at - started,
            output_tokens=output_tokens,
            input_tokens=input_tokens,
            inter_token_latencies_s=tuple(itls),
        )
    except (OSError, urllib.error.HTTPError, json.JSONDecodeError) as exc:
        return RequestResult(False, time.perf_counter() - started, error=str(exc))


async def run_benchmark(
    *,
    url: str,
    payloads: list[dict[str, Any]],
    concurrency: int,
    request_rate: float,
    headers: dict[str, str],
    timeout_s: float,
) -> tuple[list[RequestResult], float]:
    semaphore = asyncio.Semaphore(concurrency)

    async def run_one(payload: dict[str, Any]) -> RequestResult:
        async with semaphore:
            return await asyncio.to_thread(send_request, url, payload, headers, timeout_s)

    started = time.perf_counter()
    tasks = []
    rng = random.Random(0)
    for payload in payloads:
        tasks.append(asyncio.create_task(run_one(payload)))
        if math.isfinite(request_rate):
            await asyncio.sleep(rng.expovariate(request_rate))
    results = await asyncio.gather(*tasks)
    return results, time.perf_counter() - started


def load_prompts(path: Path) -> list[str]:
    prompts = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        if path.suffix == ".jsonl":
            item = json.loads(line)
            prompt = item.get("prompt")
            if prompt is None and isinstance(item.get("messages"), list):
                prompt = "\n".join(str(m.get("content", "")) for m in item["messages"])
            if not isinstance(prompt, str):
                raise ValueError("each JSONL row must contain a string 'prompt' (or 'messages')")
            prompts.append(prompt)
        else:
            prompts.append(line)
    return prompts


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark an OpenAI-compatible completion endpoint")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--endpoint", default="/v1/completions")
    parser.add_argument("--model", required=True)
    parser.add_argument("--num-prompts", type=int, default=100)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--request-rate", type=float, default=math.inf, help="requests/s; default sends immediately")
    parser.add_argument("--input-file", type=Path, help="one prompt per line, or JSONL rows with a prompt field")
    parser.add_argument("--prompt", default="Once upon a time", help="synthetic prompt (repeated when no file is given)")
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--api-key")
    parser.add_argument("--no-stream", action="store_true")
    parser.add_argument("--output-json", type=Path)
    return parser


def _format_ms(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.2f}"


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.num_prompts <= 0 or args.concurrency <= 0 or args.request_rate <= 0:
        raise SystemExit("num-prompts, concurrency, and request-rate must be positive")
    prompts = load_prompts(args.input_file) if args.input_file else [args.prompt]
    if not prompts:
        raise SystemExit("no prompts found")
    prompts = [prompts[i % len(prompts)] for i in range(args.num_prompts)]
    payloads = [
        {
            "model": args.model,
            "prompt": prompt,
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
            "stream": not args.no_stream,
            "stream_options": {"include_usage": True},
        }
        for prompt in prompts
    ]
    headers = {"Authorization": f"Bearer {args.api_key}"} if args.api_key else {}
    url = args.base_url.rstrip("/") + "/" + args.endpoint.lstrip("/")
    results, duration = asyncio.run(
        run_benchmark(
            url=url,
            payloads=payloads,
            concurrency=args.concurrency,
            request_rate=args.request_rate,
            headers=headers,
            timeout_s=args.timeout,
        )
    )
    report = summarize(results, duration)
    print(f"Requests: {report['successful_requests']} succeeded, {report['failed_requests']} failed")
    print(f"Duration: {duration:.2f} s")
    print(f"Request throughput: {report['request_throughput_rps']:.2f} req/s")
    print(f"Output throughput: {report['output_throughput_tokens_per_s']:.2f} tok/s")
    for label, key in (("Latency", "latency"), ("TTFT", "ttft"), ("ITL", "itl")):
        d = report[key]
        print(f"{label} (ms): mean {_format_ms(d['mean_ms'])}, p50 {_format_ms(d['p50_ms'])}, p99 {_format_ms(d['p99_ms'])}")
    errors = [r.error for r in results if r.error]
    if errors:
        print(f"First error: {errors[0]}", file=sys.stderr)
    if args.output_json:
        args.output_json.write_text(
            json.dumps({"summary": report, "results": [asdict(r) for r in results]}, indent=2), encoding="utf-8"
        )
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
