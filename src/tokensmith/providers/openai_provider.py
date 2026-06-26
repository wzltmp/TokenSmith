"""OpenAI-compatible adapter -- works with hosted *and* free/self-hosted models.

The OpenAI Chat Completions wire format is the lingua franca of inference
servers, so a single adapter (just change ``base_url``) talks to:

* **OpenAI** -- automatic prefix caching.
* **vLLM** (self-hosted, free) -- start with ``--enable-prefix-caching``; usage
  then carries ``prompt_tokens_details`` with ``cached_tokens``. You own the
  KV cache, so this is the cleanest demonstration of prefix reuse.
* **Z.ai GLM** (GLM-5.2 / GLM-4.6, open weights) -- context caching reports
  ``usage.prompt_tokens_details.cached_tokens`` and bills the repeated prefix
  at ~0.26/M vs 1.40/M input.
* **Ollama / LM Studio / OpenRouter** -- same interface (cache telemetry
  depends on the backend; missing fields are treated as zero cached tokens).

Caching here is *automatic*, so the only lever the caller controls is layout:
the ``CachePlanner`` still runs to keep static scaffolding at the front, which
is what lengthens the reusable prefix the server caches.
"""
from __future__ import annotations

import os
import time

from ..caching.planner import CachePlanner, Segment
from ..economics import Usage
from ..pricing import ModelPrice, get_price
from .base import Completion, Provider


class OpenAIProvider(Provider):
    def __init__(
        self,
        model: str = "gpt-5.5",
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        price: ModelPrice | None = None,
    ):
        super().__init__(model)
        try:
            import openai
        except ImportError as e:  # pragma: no cover
            raise ImportError("pip install openai to use OpenAIProvider") from e
        # Many free servers need no real key; send a placeholder if none given.
        key = api_key or os.environ.get("OPENAI_API_KEY") or "not-needed"
        # Don't route a local backend (vLLM/Ollama on loopback) through any
        # ambient HTTP/SOCKS proxy -- it would fail or add latency.
        http_client = None
        if base_url and any(h in base_url
                            for h in ("localhost", "127.0.0.1", "0.0.0.0")):
            import httpx  # provided by the openai dependency
            http_client = httpx.Client(trust_env=False, timeout=60)
        self._client = openai.OpenAI(
            api_key=key, base_url=base_url, http_client=http_client)
        self.base_url = base_url
        self.price = price or get_price(model)
        self.planner = CachePlanner(self.price.min_cache_tokens)

    def complete(
        self,
        segments: list[Segment],
        query: str,
        *,
        max_output_tokens: int = 512,
        cache_ttl: str = "5m",
    ) -> Completion:  # pragma: no cover - needs a live endpoint
        plan = self.planner.plan(segments)
        system_text = "\n".join(s.text for s in plan.ordered if s.static)
        volatile = "\n".join(s.text for s in plan.ordered if not s.static)
        user_text = f"{volatile}\n\n{query}".strip()

        t0 = time.monotonic()
        resp = self._client.chat.completions.create(
            model=self.model,
            max_tokens=max_output_tokens,
            messages=[
                {"role": "system", "content": system_text},
                {"role": "user", "content": user_text},
            ],
        )
        latency = (time.monotonic() - t0) * 1000
        usage = parse_openai_usage(resp.usage)
        text = resp.choices[0].message.content or ""
        return Completion(text, usage, latency, usage.cached_read > 0, self.model)


def parse_openai_usage(u) -> Usage:
    """Map an OpenAI-style usage object/dict to our Usage.

    Reads ``prompt_tokens_details.cached_tokens`` when present (OpenAI, vLLM
    with prefix caching, Z.ai GLM); absent fields default to zero cached.
    Pulled out so it can be unit-tested without a live server.
    """
    def get(obj, name, default=0):
        if obj is None:
            return default
        if isinstance(obj, dict):
            return obj.get(name, default)
        return getattr(obj, name, default) or default

    prompt_tokens = get(u, "prompt_tokens")
    completion_tokens = get(u, "completion_tokens")
    details = get(u, "prompt_tokens_details", None)
    cached = get(details, "cached_tokens", 0) if details else 0
    cached = min(cached, prompt_tokens)
    return Usage(
        uncached_input=prompt_tokens - cached,
        cached_read=cached,
        cache_write=0,
        output=completion_tokens,
    )
