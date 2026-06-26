"""Turn token usage into money and latency.

This is the layer that makes the cost claims concrete: given how a prompt
is laid out and how often its static scaffolding repeats, what does caching
(or failing to cache) actually cost?
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .pricing import ModelPrice


@dataclass
class Usage:
    """Token accounting for a single LLM call."""
    uncached_input: int = 0   # input tokens billed at full rate
    cached_read: int = 0      # input tokens served from cache
    cache_write: int = 0      # input tokens written to cache this call
    output: int = 0

    @property
    def total_input(self) -> int:
        return self.uncached_input + self.cached_read + self.cache_write

    def __add__(self, other: "Usage") -> "Usage":
        return Usage(
            self.uncached_input + other.uncached_input,
            self.cached_read + other.cached_read,
            self.cache_write + other.cache_write,
            self.output + other.output,
        )


def call_cost(usage: Usage, price: ModelPrice, cache_ttl: str = "5m") -> float:
    """Cost in USD for one call given its usage and the model's pricing."""
    write_rate = (price.cache_write_1h_per_m if cache_ttl == "1h"
                  else price.cache_write_5m_per_m)
    return (
        usage.uncached_input / 1e6 * price.input_per_m
        + usage.cached_read / 1e6 * price.cached_read_per_m
        + usage.cache_write / 1e6 * write_rate
        + usage.output / 1e6 * price.output_per_m
    )


@dataclass
class VolumeReport:
    """Projected economics over a request volume (e.g. a month of traffic)."""
    requests: int
    naive_cost: float            # no caching: every call pays full input
    cached_cost: float           # with caching
    naive_input_tokens: int
    cached_read_tokens: int
    cache_hit_rate: float
    details: dict = field(default_factory=dict)

    @property
    def savings(self) -> float:
        return self.naive_cost - self.cached_cost

    @property
    def savings_pct(self) -> float:
        return 0.0 if self.naive_cost == 0 else self.savings / self.naive_cost

    def render(self) -> str:
        return (
            f"Requests:        {self.requests:,}\n"
            f"Cache hit rate:  {self.cache_hit_rate:.0%}\n"
            f"Cost (no cache): ${self.naive_cost:,.2f}\n"
            f"Cost (cached):   ${self.cached_cost:,.2f}\n"
            f"Savings:         ${self.savings:,.2f} ({self.savings_pct:.0%})"
        )


def project_volume(
    *,
    static_tokens: int,
    dynamic_input_tokens: int,
    output_tokens: int,
    requests: int,
    price: ModelPrice,
    cache_hit_rate: float = 1.0,
    cache_ttl: str = "5m",
) -> VolumeReport:
    """Project cost for a workload that repeats a static prefix every call.

    ``cache_hit_rate`` is the share of calls that land on a warm cache. The
    first call (and any miss) pays a write surcharge for the static prefix;
    hits read it at the discounted rate. Dynamic tokens are always billed at
    the full input rate.
    """
    cacheable = static_tokens >= price.min_cache_tokens
    if not cacheable:
        # Prefix too short to cache: caching never engages, so no writes.
        cache_hit_rate = 0.0

    hits = round(requests * cache_hit_rate)
    misses = requests - hits

    # Naive: every request pays full input for the entire prompt.
    naive = Usage(
        uncached_input=(static_tokens + dynamic_input_tokens) * requests,
        output=output_tokens * requests,
    )
    naive_cost = call_cost(naive, price, cache_ttl)

    if not cacheable:
        # No caching possible -> the "cached" path is identical to naive.
        return VolumeReport(
            requests=requests,
            naive_cost=naive_cost,
            cached_cost=naive_cost,
            naive_input_tokens=naive.total_input,
            cached_read_tokens=0,
            cache_hit_rate=0.0,
            details={"model": price.name, "uncacheable_prefix": True},
        )

    # Cached: misses write the static prefix; hits read it. Dynamic always full.
    if price.explicit_cache:
        write_static = static_tokens * misses
        read_static = static_tokens * hits
        full_input = dynamic_input_tokens * requests
    else:
        # Automatic caching: no separate write surcharge.
        write_static = 0
        read_static = static_tokens * hits
        full_input = static_tokens * misses + dynamic_input_tokens * requests

    cached = Usage(
        uncached_input=full_input,
        cached_read=read_static,
        cache_write=write_static,
        output=output_tokens * requests,
    )
    cached_cost = call_cost(cached, price, cache_ttl)

    return VolumeReport(
        requests=requests,
        naive_cost=naive_cost,
        cached_cost=cached_cost,
        naive_input_tokens=naive.total_input,
        cached_read_tokens=read_static,
        cache_hit_rate=cache_hit_rate,
        details={
            "static_tokens": static_tokens,
            "dynamic_input_tokens": dynamic_input_tokens,
            "output_tokens": output_tokens,
            "model": price.name,
            "cache_ttl": cache_ttl,
        },
    )
