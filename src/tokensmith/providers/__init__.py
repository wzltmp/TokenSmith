"""Provider registry. Real adapters import lazily so the toolkit works with
zero third-party SDKs installed (the MockProvider is always available)."""
from __future__ import annotations

from .base import Completion, Provider
from .mock import MockProvider

__all__ = ["Completion", "Provider", "MockProvider", "get_provider"]


def get_provider(name: str, model: str | None = None, **kwargs) -> Provider:
    name = name.lower()
    if name == "mock":
        return MockProvider(model or "claude-sonnet-4.6", **kwargs)
    if name == "anthropic":
        from .anthropic_provider import AnthropicProvider
        return AnthropicProvider(model or "claude-sonnet-4.6", **kwargs)
    if name == "openai":
        from .openai_provider import OpenAIProvider
        return OpenAIProvider(model or "gpt-5.5", **kwargs)
    if name in ("openai_compatible", "vllm", "glm", "zai", "local", "ollama"):
        # Any OpenAI-compatible endpoint (free/self-hosted included).
        from .openai_provider import OpenAIProvider
        defaults = {
            "vllm": ("local-oss", "http://localhost:8000/v1"),
            "ollama": ("local-oss", "http://localhost:11434/v1"),
            "local": ("local-oss", "http://localhost:8000/v1"),
            "glm": ("glm-5.2", "https://api.z.ai/api/paas/v4"),
            "zai": ("glm-5.2", "https://api.z.ai/api/paas/v4"),
        }
        d_model, d_url = defaults.get(name, (None, None))
        kwargs.setdefault("base_url", d_url)
        return OpenAIProvider(model or d_model or "local-oss", **kwargs)
    raise ValueError(
        f"Unknown provider {name!r}. Use "
        f"mock|anthropic|openai|vllm|glm|ollama|openai_compatible.")
