"""Tests for the cache-bust linter, OSS pricing, and the OpenAI-compatible
provider -- including a live run against a stdlib mock server that returns
``prompt_tokens_details.cached_tokens`` exactly like vLLM / Z.ai GLM do."""
import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pytest

from tokensmith import (
    CacheBustDetector,
    Segment,
    lint_prompt,
    lint_text,
)
from tokensmith.pricing import MODELS, free_model_price
from tokensmith.providers.openai_provider import parse_openai_usage


# ---------------- cache-bust linter (static analysis) ----------------

def test_lint_flags_uuid_and_timestamp():
    text = ("You are a support agent. Session started 2026-06-26T14:03:55Z "
            "request_id: 7f3e9c12. trace 550e8400-e29b-41d4-a716-446655440000")
    findings = lint_text(text)
    kinds = {f.pattern for f in findings}
    assert "uuid" in kinds
    assert "iso_datetime" in kinds
    assert any(f.severity == "high" for f in findings)


def test_lint_clean_static_block_passes():
    text = "You are a helpful assistant. Always follow the policy. " * 20
    assert lint_text(text) == []


def test_lint_prompt_ignores_volatile_segments():
    segs = [
        Segment("sys", "You are a stable system prompt. " * 10, static=True),
        # volatile content is EXPECTED to change -> must not be flagged
        Segment("ctx", "Now is 2026-06-26T10:00:00Z, id 123e4567-e89b-"
                       "12d3-a456-426614174000", static=False),
    ]
    assert lint_prompt(segs).ok


def test_lint_offset_warns_when_at_prefix_start():
    text = "2026-06-26 is today. " + "stable text follows. " * 30
    report = lint_prompt([Segment("sys", text, static=True)])
    assert not report.ok
    assert report.worst_offset == 0


# ---------------- cache-bust detector (dynamic analysis) ----------------

def test_detector_catches_drifting_static_block():
    det = CacheBustDetector()
    # same name, "static", but content changes each call -> busts the cache
    det.record([Segment("sys", "policy v1 ... timestamp 1000", static=True)])
    changed = det.record([Segment("sys", "policy v1 ... timestamp 2000", static=True)])
    assert "sys" in changed
    assert det.bust_rate == 1.0


def test_detector_stable_block_no_bust():
    det = CacheBustDetector()
    seg = [Segment("sys", "perfectly stable prompt", static=True)]
    det.record(seg)
    det.record(seg)
    det.record(seg)
    assert det.bust_rate == 0.0


# ---------------- OSS pricing ----------------

def test_local_model_is_free_and_low_minimum():
    p = MODELS["local-oss"]
    assert p.input_per_m == 0.0
    assert p.cache_read_discount == 0.0  # no division-by-zero
    assert p.min_cache_tokens <= 64


def test_glm_reports_discounted_cache():
    p = MODELS["glm-5.2"]
    assert p.cached_read_per_m < p.input_per_m
    assert p.cache_read_discount > 0.5


def test_free_model_price_factory():
    p = free_model_price("my-llama", input_per_m=0.0)
    assert p.provider == "local"
    assert p.explicit_cache is False


# ---------------- usage parser (the field vLLM/GLM/OpenAI all use) ----------------

def test_parse_usage_with_cached_tokens_dict():
    u = {"prompt_tokens": 1000, "completion_tokens": 50,
         "prompt_tokens_details": {"cached_tokens": 900}}
    parsed = parse_openai_usage(u)
    assert parsed.cached_read == 900
    assert parsed.uncached_input == 100
    assert parsed.output == 50


def test_parse_usage_without_details_defaults_zero():
    u = {"prompt_tokens": 500, "completion_tokens": 20}
    parsed = parse_openai_usage(u)
    assert parsed.cached_read == 0
    assert parsed.uncached_input == 500


# ---------------- LIVE run against a mock OpenAI-compatible server ----------------

class _MockOpenAIHandler(BaseHTTPRequestHandler):
    calls = 0

    def log_message(self, *a):  # silence
        pass

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        self.rfile.read(length)
        type(self).calls += 1
        # First call cold (no cache); subsequent calls warm (prefix reused).
        prompt_tokens = 1200
        cached = 0 if type(self).calls == 1 else 1100
        body = {
            "id": "chatcmpl-mock",
            "object": "chat.completion",
            "choices": [{"index": 0, "message": {
                "role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": 5,
                "total_tokens": prompt_tokens + 5,
                "prompt_tokens_details": {"cached_tokens": cached},
            },
        }
        payload = json.dumps(body).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


@pytest.fixture
def mock_server(monkeypatch):
    # Ensure loopback isn't routed through any ambient (SOCKS) proxy.
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")
    monkeypatch.setenv("no_proxy", "127.0.0.1,localhost")
    _MockOpenAIHandler.calls = 0
    srv = HTTPServer(("127.0.0.1", 0), _MockOpenAIHandler)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    yield f"http://127.0.0.1:{srv.server_port}/v1"
    srv.shutdown()


def test_live_provider_sees_cache_on_warm_call(mock_server):
    openai = pytest.importorskip("openai")  # skip cleanly if SDK absent
    from tokensmith.providers.openai_provider import OpenAIProvider

    prov = OpenAIProvider("local-oss", base_url=mock_server, api_key="x",
                          price=free_model_price("local-oss"))
    seg = [Segment("scaffold", "stable system prompt " * 200, static=True)]
    c1 = prov.complete(seg, "hi", max_output_tokens=5)
    c2 = prov.complete(seg, "hi", max_output_tokens=5)
    assert c1.cache_hit is False
    assert c2.cache_hit is True
    assert c1.usage.cached_read == 0
    assert c2.usage.cached_read == 1100
