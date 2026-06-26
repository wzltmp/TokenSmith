"""Provider pricing and cache mechanics.

All prices are USD per 1,000,000 tokens and reflect published rates as of
June 2026. They are *defaults*: override them with ``ModelPrice`` if a
provider changes pricing. Sources are listed in the project README.

Two cache-cost models exist in the wild and both are represented here:

* **Explicit (Anthropic-style):** the caller marks cacheable prefixes. The
  first call *writes* the cache at a surcharge (1.25x for a 5 min TTL, 2x for
  1 hour); later calls *read* it at 0.1x the base input price.
* **Automatic (OpenAI-style):** the provider caches long prefixes server-side
  with no write surcharge; cache reads are billed at a discounted rate that
  varies by model (50%-90% off).
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelPrice:
    name: str
    provider: str
    input_per_m: float          # base (uncached) input $/1M
    output_per_m: float         # output $/1M
    cached_read_per_m: float    # input read from cache $/1M
    cache_write_5m_per_m: float # one-time write surcharge, 5 min TTL $/1M
    cache_write_1h_per_m: float # one-time write surcharge, 1 hour TTL $/1M
    min_cache_tokens: int       # shortest cacheable prefix
    explicit_cache: bool        # True = caller sets breakpoints (Anthropic)

    @property
    def cache_read_discount(self) -> float:
        """Fraction saved on a cache hit vs. an uncached input token."""
        if self.input_per_m == 0:
            return 0.0  # self-hosted: savings are in tokens/latency, not $
        return 1.0 - (self.cached_read_per_m / self.input_per_m)


# Defaults reflect published June 2026 rates. Output prices for some tiers are
# representative; override per your contract. See README "Pricing sources".
MODELS: dict[str, ModelPrice] = {
    # --- Anthropic (explicit cache_control breakpoints) ---
    "claude-opus-4.8": ModelPrice(
        "claude-opus-4.8", "anthropic", 5.0, 25.0, 0.50, 6.25, 10.0, 1024, True),
    "claude-sonnet-4.6": ModelPrice(
        "claude-sonnet-4.6", "anthropic", 3.0, 15.0, 0.30, 3.75, 6.0, 1024, True),
    "claude-haiku-4.5": ModelPrice(
        "claude-haiku-4.5", "anthropic", 1.0, 5.0, 0.10, 1.25, 2.0, 2048, True),
    # --- OpenAI (automatic prefix caching, no write surcharge) ---
    "gpt-5.5": ModelPrice(
        "gpt-5.5", "openai", 5.0, 30.0, 0.50, 5.0, 5.0, 1024, False),
    "gpt-5.4": ModelPrice(
        "gpt-5.4", "openai", 2.5, 15.0, 0.25, 2.5, 2.5, 1024, False),
    "gpt-4.1": ModelPrice(
        "gpt-4.1", "openai", 2.0, 8.0, 0.50, 2.0, 2.0, 1024, False),
    # --- Open-weights / free-to-self-host (automatic prefix caching) ---
    # Z.ai GLM: context caching reports usage.prompt_tokens_details.cached_tokens.
    # Cached input ~$0.26/M vs $1.40/M input (~81% off the repeated part).
    "glm-4.6": ModelPrice(
        "glm-4.6", "zai", 1.40, 2.20, 0.26, 1.40, 1.40, 1024, False),
    "glm-5.2": ModelPrice(
        "glm-5.2", "zai", 1.40, 2.20, 0.26, 1.40, 1.40, 1024, False),
    # vLLM / Ollama / LM Studio self-hosted: $0 marginal cost. The win here is
    # token + latency reduction, which the toolkit still measures. min cache is
    # vLLM's default block size (16 tokens).
    "local-oss": ModelPrice(
        "local-oss", "local", 0.0, 0.0, 0.0, 0.0, 0.0, 16, False),
}


def free_model_price(
    name: str,
    *,
    min_cache_tokens: int = 16,
    input_per_m: float = 0.0,
    output_per_m: float = 0.0,
    cached_read_per_m: float = 0.0,
) -> ModelPrice:
    """Build a price entry for an arbitrary self-hosted/open model.

    Defaults to $0 (self-hosted): the toolkit then reports token and latency
    savings rather than dollars. Pass real per-1M rates for hosted OSS APIs.
    """
    return ModelPrice(
        name=name, provider="local", input_per_m=input_per_m,
        output_per_m=output_per_m, cached_read_per_m=cached_read_per_m,
        cache_write_5m_per_m=input_per_m, cache_write_1h_per_m=input_per_m,
        min_cache_tokens=min_cache_tokens, explicit_cache=False)


def get_price(model: str) -> ModelPrice:
    try:
        return MODELS[model]
    except KeyError:
        raise KeyError(
            f"Unknown model {model!r}. Known: {', '.join(MODELS)}"
        ) from None
