"""Deterministic simulation provider.

No API key required. It reproduces the *economics and latency* of prompt
caching faithfully enough to benchmark layout decisions, which is the point
of this toolkit. Caching is modeled with a TTL keyed on the hash of the
cacheable prefix: the first time a prefix is seen it is "written" (a miss);
subsequent calls within the TTL are "reads" (hits).

Latency model: time-to-first-token scales with the number of *uncached*
input tokens, because cached prefixes skip the prefill compute. This mirrors
the real-world observation that caching cuts latency, not just cost.
"""
from __future__ import annotations

import hashlib
import time

from ..caching.planner import CachePlanner, Segment
from ..economics import Usage
from ..pricing import get_price
from .base import Completion, Provider

# Rough latency constants (ms). Tunable; used only for relative comparison.
_BASE_LATENCY_MS = 180.0
_PREFILL_MS_PER_1K = 22.0     # cost of prefilling uncached input
_CACHED_MS_PER_1K = 2.0       # cost of replaying a cached prefix
_DECODE_MS_PER_TOKEN = 6.0


class MockProvider(Provider):
    def __init__(self, model: str = "claude-sonnet-4.6", clock=time.monotonic):
        super().__init__(model)
        self.price = get_price(model)
        self.planner = CachePlanner(self.price.min_cache_tokens)
        self._clock = clock
        self._cache: dict[str, float] = {}  # prefix hash -> expiry time
        self.calls = 0

    def _ttl_seconds(self, cache_ttl: str) -> float:
        return 3600.0 if cache_ttl == "1h" else 300.0

    def _prefix_key(self, prefix_text: str) -> str:
        return hashlib.sha256(prefix_text.encode("utf-8")).hexdigest()

    def complete(
        self,
        segments: list[Segment],
        query: str,
        *,
        max_output_tokens: int = 512,
        cache_ttl: str = "5m",
        use_cache: bool = True,
    ) -> Completion:
        self.calls += 1
        plan = self.planner.plan(segments)
        prefix_text = "".join(
            s.text for s in plan.ordered[: plan.breakpoint_index + 1])
        cacheable = plan.cacheable_tokens
        volatile = plan.volatile_tokens + self._q_tokens(query)

        cacheable_ok = use_cache and cacheable >= self.price.min_cache_tokens
        now = self._clock()
        hit = False
        if cacheable_ok:
            key = self._prefix_key(prefix_text)
            expiry = self._cache.get(key, 0.0)
            hit = now < expiry
            self._cache[key] = now + self._ttl_seconds(cache_ttl)

        if not cacheable_ok:
            usage = Usage(uncached_input=cacheable + volatile,
                          output=max_output_tokens)
            uncached = cacheable + volatile
        elif hit:
            usage = Usage(uncached_input=volatile, cached_read=cacheable,
                          output=max_output_tokens)
            uncached = volatile
        else:  # write (first call / miss)
            if self.price.explicit_cache:
                usage = Usage(uncached_input=volatile, cache_write=cacheable,
                              output=max_output_tokens)
            else:  # automatic caches bill the write as normal input
                usage = Usage(uncached_input=cacheable + volatile,
                              output=max_output_tokens)
            uncached = cacheable + volatile

        latency = (
            _BASE_LATENCY_MS
            + uncached / 1000 * _PREFILL_MS_PER_1K
            + (cacheable / 1000 * _CACHED_MS_PER_1K if hit else 0.0)
            + max_output_tokens * _DECODE_MS_PER_TOKEN
        )
        text = f"[mock:{self.model}] answered {query[:60]!r} ({usage.output} out tokens)"
        return Completion(text, usage, latency, hit, self.model)

    @staticmethod
    def _q_tokens(query: str) -> int:
        from ..tokenizer import count_tokens
        return count_tokens(query)
