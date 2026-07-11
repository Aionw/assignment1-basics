import asyncio
import json
import math
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from cs336_basics.bench_lm import RequestResult, run_benchmark, summarize


class StreamingHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        assert body["messages"][0]["content"] == "hello"
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        for content in ("", "one", "two"):
            event = {"choices": [{"delta": {"content": content}}]}
            self.wfile.write(f"data: {json.dumps(event)}\n\n".encode())
            self.wfile.flush()
            time.sleep(0.003)
        self.wfile.write(b"data: [DONE]\n\n")

    def log_message(self, *_args):
        pass


def test_run_benchmark_reads_chat_stream():
    server = ThreadingHTTPServer(("127.0.0.1", 0), StreamingHandler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    body = {"model": "test", "messages": [{"role": "user", "content": "hello"}], "stream": True}
    try:
        results, duration = asyncio.run(
            run_benchmark(
                url=f"http://127.0.0.1:{server.server_port}/v1/chat/completions",
                requests=[(body, 3), (body, 3)],
                concurrency=2,
                request_rate=math.inf,
                timeout_s=2,
            )
        )
    finally:
        server.shutdown()
        server.server_close()
    report = summarize(results, duration)
    assert report["completed_requests"] == 2
    assert report["total_input_tokens"] == 6
    assert report["total_output_tokens"] == 4
    assert report["ttft"]["p50_ms"] is not None
    assert report["itl"]["p50_ms"] >= 1


def test_summary_handles_failures_and_tpot():
    results = [
        RequestResult(True, 0.5, 4, 3, 0.1, (0.2, 0.2)),
        RequestResult(False, 0.2, 4, error="failed"),
    ]
    report = summarize(results, 1.0)
    assert report["failed_requests"] == 1
    assert report["request_throughput_rps"] == 1
    assert report["tpot"]["mean_ms"] == 200
