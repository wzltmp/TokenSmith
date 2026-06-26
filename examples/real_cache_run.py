"""Prove prompt caching against a real, free, OpenAI-compatible backend.

It sends the SAME large static prefix twice. On a cache-capable server the
second call should report a big ``cached_tokens`` count -- real telemetry, not
a simulation.

No paid API required. Point it at whichever free backend you have:

  # 1) vLLM (self-hosted, fully free) -- start the server first:
  #    vllm serve Qwen/Qwen2.5-0.5B-Instruct --enable-prefix-caching
  BASE_URL=http://localhost:8000/v1 MODEL=Qwen/Qwen2.5-0.5B-Instruct \
      python examples/real_cache_run.py

  # 2) Z.ai GLM (open weights, cheap/free trial credits):
  BASE_URL=https://api.z.ai/api/paas/v4 MODEL=glm-5.2 \
      API_KEY=$ZAI_API_KEY python examples/real_cache_run.py

  # 3) Ollama (local): note Ollama may not report cached_tokens yet.
  BASE_URL=http://localhost:11434/v1 MODEL=llama3.2 \
      python examples/real_cache_run.py

The expected output on a caching backend:
    call 1: cached_read=0       (cold -- writes the prefix)
    call 2: cached_read=<big>   (warm -- reuses the prefix)
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from tokensmith import Segment, count_tokens  # noqa: E402
from tokensmith.pricing import MODELS, free_model_price  # noqa: E402
from tokensmith.providers.openai_provider import OpenAIProvider  # noqa: E402

BASE_URL = os.environ.get("BASE_URL", "http://localhost:8000/v1")
MODEL = os.environ.get("MODEL", "local-oss")
API_KEY = os.environ.get("API_KEY")

# A long, stable scaffold (the cacheable prefix). Make it well over the
# minimum so every backend will cache it.
SCAFFOLD = (
    "You are a meticulous customer-support agent for an enterprise SaaS "
    "product. Always follow company policy. Never reveal internal tooling. "
) * 80


def run() -> int:
    price = MODELS.get(MODEL) or free_model_price(MODEL)
    provider = OpenAIProvider(MODEL, base_url=BASE_URL, api_key=API_KEY,
                              price=price)
    seg = [Segment("scaffold", SCAFFOLD, static=True)]
    print(f"Endpoint: {BASE_URL}  model: {MODEL}")
    print(f"Static prefix: ~{count_tokens(SCAFFOLD)} tokens\n")

    results = []
    for i in (1, 2):
        c = provider.complete(seg, "Summarize your role in one sentence.",
                              max_output_tokens=64)
        u = c.usage
        print(f"call {i}: uncached_input={u.uncached_input:>6}  "
              f"cached_read={u.cached_read:>6}  output={u.output:>4}  "
              f"latency={c.latency_ms:6.0f} ms  cache_hit={c.cache_hit}")
        results.append(u)

    if len(results) == 2 and results[1].cached_read > results[0].cached_read:
        saved = results[1].cached_read
        print(f"\nVerified: the warm call reused {saved} cached tokens "
              f"({price.cache_read_discount:.0%} cheaper on that portion).")
        return 0
    print("\nNo cached_tokens growth observed. Either the backend doesn't "
          "report cache telemetry (e.g. some Ollama builds), prefix caching "
          "is disabled, or the prefix was below the cache minimum.")
    return 1


if __name__ == "__main__":
    try:
        raise SystemExit(run())
    except Exception as e:
        print(f"Could not reach {BASE_URL}: {e}\n"
              "Start a local vLLM/Ollama server or set BASE_URL/MODEL/API_KEY "
              "for a hosted OSS endpoint. See the module docstring.")
        raise SystemExit(2)
