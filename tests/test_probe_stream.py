import importlib.util
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

_PATH = Path(__file__).resolve().parent.parent / "tools" / "probe_stream.py"
_spec = importlib.util.spec_from_file_location("probe_stream", _PATH)
probe = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(probe)


def _chunk(delta=None, usage=None, finish=None):
    chunk = {"choices": [] if delta is None and finish is None else [{"delta": delta or {}, "finish_reason": finish}]}
    if usage is not None:
        chunk["usage"] = usage
    return chunk


def test_iter_sse_data_stops_at_done():
    lines = [": keep-alive", "data: {\"a\": 1}", "", "event: x", "data: {\"b\": 2}", "data: [DONE]", "data: {\"c\": 3}"]
    assert list(probe.iter_sse_data(lines)) == ['{"a": 1}', '{"b": 2}']


def test_field_reasoning_with_continuous_usage():
    chunks = [
        _chunk({"role": "assistant"}, usage={"completion_tokens": 0}),
        _chunk({"reasoning_content": "use 32x32 tiles"}, usage={"completion_tokens": 5}),
        _chunk({"reasoning_content": " in shared memory"}, usage={"completion_tokens": 9}),
        _chunk({"content": "```cuda\n__global__ void k(){}\n```"}, usage={"completion_tokens": 20}),
        _chunk({}, finish="stop", usage={"completion_tokens": 21}),
        _chunk(usage={"completion_tokens": 21, "prompt_tokens": 40}),
    ]
    report = probe.analyze_chunks(chunks, arrival_s=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
    assert report["reasoning_mode"] == "field"
    assert report["reasoning_fields"] == ["reasoning_content"]
    assert report["usage_mode"] == "continuous"
    assert report["finish_reason"] == "stop"
    assert report["ttft_s"] == pytest.approx(0.1)
    assert report["text_fields"]["content"] > 0


def test_inline_reasoning_with_final_usage():
    chunks = [
        _chunk({"content": "<thi"}),
        _chunk({"content": "nk>plan the tiling</th"}),
        _chunk({"content": "ink>\n```python\nprint(1)\n```"}),
        _chunk({}, finish="stop"),
        _chunk(usage={"completion_tokens": 30}),
    ]
    report = probe.analyze_chunks(chunks)
    assert report["reasoning_mode"] == "inline"
    assert report["inline_starts_with_open_tag"] is True
    assert report["usage_mode"] == "final"
    suggestion = probe.suggest_profile(report, None)
    assert suggestion == {"reasoning": "inline", "inline_starts_in_reasoning": False, "usage_stats": "final"}


def test_no_reasoning_and_no_usage():
    report = probe.analyze_chunks([_chunk({"content": "hello"}), _chunk({}, finish="length")])
    assert report["reasoning_mode"] == "none"
    assert report["usage_mode"] == "none"
    assert report["finish_reason"] == "length"


def test_continuation_verdicts():
    ok = probe.analyze_continuation({"content": "```python\nimport torch\n```"})
    assert ok["usable"] is True
    again = probe.analyze_continuation({"content": "x", "reasoning_content": "let me think again"})
    assert again["usable"] is False and again["verdict"].startswith("no")
    inline_again = probe.analyze_continuation({"content": "<think>hmm</think>```python```"})
    assert inline_again["usable"] is False


class _FakeServer(BaseHTTPRequestHandler):
    """Streams reasoning + content; rejects continuous_usage_stats like a strict API."""

    requests: list[dict] = []

    def log_message(self, *args):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        type(self).requests.append(body)
        if body.get("stream"):
            if body.get("stream_options", {}).get("continuous_usage_stats"):
                self.send_response(400)
                self.end_headers()
                self.wfile.write(b'{"error": "unknown field continuous_usage_stats"}')
                return
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            for chunk in (
                _chunk({"reasoning_content": "tile size 32"}),
                _chunk({"content": "```cuda\n__global__ void k(){}\n```"}),
                _chunk({}, finish="stop"),
                _chunk(usage={"completion_tokens": 12}),
            ):
                self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
            self.wfile.write(b"data: [DONE]\n\n")
            return
        reply = {"content": "```python\nimport torch\n```"} if body.get("continue_final_message") else {
            "content": "x", "reasoning_content": "thinking again"}
        data = json.dumps({"choices": [{"message": reply}]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def test_probe_end_to_end_against_fake_server(tmp_path):
    server = ThreadingHTTPServer(("127.0.0.1", 0), _FakeServer)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        report_path = tmp_path / "report.json"
        url = f"http://127.0.0.1:{server.server_address[1]}/v1"
        assert probe.main(["--base-url", url, "--model-name", "fake", "--report", str(report_path)]) == 0
    finally:
        server.shutdown()
    report = json.loads(report_path.read_text())
    assert report["stream"]["continuous_usage_stats_accepted"] is False
    assert report["suggested_profile"] == {
        "reasoning": "field",
        "reasoning_field": "reasoning_content",
        "usage_stats": "final",
        "prefix_continuation": True,
    }
    streamed = [r for r in _FakeServer.requests if r.get("stream")]
    assert len(streamed) == 2 and "continuous_usage_stats" not in streamed[1]["stream_options"]
