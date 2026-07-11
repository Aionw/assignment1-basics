import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from cs336_basics.bench import run_benchmark, summarize


class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        length = int(self.headers["Content-Length"])
        payload = json.loads(self.rfile.read(length))
        assert payload["model"] == "test"
        chunks = [
            {"choices": [{"text": "a"}]},
            {"choices": [{"text": "b"}]},
            {"choices": [], "usage": {"prompt_tokens": 3, "completion_tokens": 2}},
        ]
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        for chunk in chunks:
            self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
            self.wfile.flush()
            time.sleep(0.005)
        self.wfile.write(b"data: [DONE]\n\n")

    def log_message(self, *_args):
        pass


def test_streaming_benchmark_collects_metrics():
    import asyncio

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        payload = {"model": "test", "prompt": "hello", "stream": True}
        results, duration = asyncio.run(
            run_benchmark(
                url=f"http://127.0.0.1:{server.server_port}/v1/completions",
                payloads=[payload, payload],
                concurrency=2,
                request_rate=float("inf"),
                headers={},
                timeout_s=2,
            )
        )
    finally:
        server.shutdown()
        server.server_close()
    report = summarize(results, duration)
    assert report["successful_requests"] == 2
    assert report["total_input_tokens"] == 6
    assert report["total_output_tokens"] == 4
    assert report["ttft"]["p50_ms"] is not None
    assert report["itl"]["p50_ms"] >= 1
