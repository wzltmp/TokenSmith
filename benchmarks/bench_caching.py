"""Benchmark 1 -- the caching opportunity.

Simulates a heavily scaffolded agent: a large static system prompt (policies,
tool schemas) repeated across many calls with small per-call dynamic context.
Compares cost and latency with caching off vs. on, and sweeps the cache-hit
rate to show what the typical ~28% adoption rate is leaving on the table.

Run: python benchmarks/bench_caching.py
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from tokensmith import MockProvider, Segment, get_price  # noqa: E402
from tokensmith.economics import call_cost, project_volume  # noqa: E402

SCAFFOLD_TOKENS = 8000     # system prompt + tool guidance (report: 69% of input)
DYNAMIC_TOKENS = 600       # per-call user/context tokens
OUTPUT_TOKENS = 350
REQUESTS = 50_000
MODEL = "claude-sonnet-4.6"


def latency_demo() -> dict:
    """Measure simulated latency for a cold vs. warm call."""
    prov = MockProvider(MODEL)
    scaffold = Segment("sys", "policy and tool guidance. " * 1600, static=True)
    ctx = Segment("ctx", "user turn context. " * 30, static=False)
    cold = prov.complete([scaffold, ctx], "first question",
                         max_output_tokens=OUTPUT_TOKENS)
    warm = prov.complete([scaffold, ctx], "second question",
                         max_output_tokens=OUTPUT_TOKENS)
    return {
        "cold_ms": round(cold.latency_ms, 1),
        "warm_ms": round(warm.latency_ms, 1),
        "latency_reduction": round(1 - warm.latency_ms / cold.latency_ms, 3),
        "cold_cache_hit": cold.cache_hit,
        "warm_cache_hit": warm.cache_hit,
    }


def hit_rate_sweep() -> list[dict]:
    price = get_price(MODEL)
    rows = []
    for hr in (0.0, 0.28, 0.5, 0.8, 0.95, 1.0):
        rep = project_volume(
            static_tokens=SCAFFOLD_TOKENS, dynamic_input_tokens=DYNAMIC_TOKENS,
            output_tokens=OUTPUT_TOKENS, requests=REQUESTS, price=price,
            cache_hit_rate=hr)
        rows.append({
            "cache_hit_rate": hr,
            "monthly_cost_usd": round(rep.cached_cost, 2),
            "savings_vs_no_cache_usd": round(rep.naive_cost - rep.cached_cost, 2),
            "savings_pct": round(rep.savings_pct, 3),
        })
    return rows


def main():
    price = get_price(MODEL)
    naive = project_volume(
        static_tokens=SCAFFOLD_TOKENS, dynamic_input_tokens=DYNAMIC_TOKENS,
        output_tokens=OUTPUT_TOKENS, requests=REQUESTS, price=price,
        cache_hit_rate=0.0)

    result = {
        "model": MODEL,
        "scenario": {
            "scaffold_tokens": SCAFFOLD_TOKENS,
            "dynamic_tokens": DYNAMIC_TOKENS,
            "output_tokens": OUTPUT_TOKENS,
            "requests": REQUESTS,
        },
        "no_cache_monthly_cost_usd": round(naive.naive_cost, 2),
        "latency": latency_demo(),
        "hit_rate_sweep": hit_rate_sweep(),
    }

    print("=" * 64)
    print("BENCHMARK 1: Caching opportunity")
    print("=" * 64)
    print(f"Model: {MODEL}  |  {REQUESTS:,} requests/month")
    print(f"Static scaffold: {SCAFFOLD_TOKENS:,} tok  "
          f"({SCAFFOLD_TOKENS / (SCAFFOLD_TOKENS + DYNAMIC_TOKENS):.0%} of input)")
    print(f"\nNo-cache monthly cost: ${naive.naive_cost:,.2f}\n")
    lat = result["latency"]
    print(f"Latency  cold {lat['cold_ms']} ms -> warm {lat['warm_ms']} ms "
          f"({lat['latency_reduction']:.0%} faster on a cache hit)\n")
    print(f"{'hit rate':>9} | {'monthly $':>11} | {'saved $':>11} | saved %")
    print("-" * 52)
    for r in result["hit_rate_sweep"]:
        print(f"{r['cache_hit_rate']:>8.0%} | {r['monthly_cost_usd']:>11,.2f} | "
              f"{r['savings_vs_no_cache_usd']:>11,.2f} | {r['savings_pct']:>6.0%}")

    out = os.path.join(os.path.dirname(__file__), "results_caching.json")
    with open(out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nWrote {out}")
    return result


if __name__ == "__main__":
    main()
