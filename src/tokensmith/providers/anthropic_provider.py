"""Anthropic adapter (explicit prompt caching).

Maps our ``Segment`` layout onto Anthropic's ``cache_control`` breakpoints:
static segments become system blocks, and a ``{"type": "ephemeral"}``
breakpoint is placed on the last static block so the whole scaffolding prefix
is cached. Requires ``anthropic`` and ``ANTHROPIC_API_KEY``; import stays lazy
so the rest of the toolkit runs without the SDK installed.
"""
from __future__ import annotations

import os
import time

from ..caching.planner import CachePlanner, Segment
from ..economics import Usage
from ..pricing import get_price
from .base import Completion, Provider


class AnthropicProvider(Provider):
    def __init__(self, model: str = "claude-sonnet-4.6", api_key: str | None = None):
        super().__init__(model)
        try:
            import anthropic
        except ImportError as e:  # pragma: no cover
            raise ImportError(
                "pip install anthropic to use AnthropicProvider") from e
        self._client = anthropic.Anthropic(
            api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"))
        self.price = get_price(model)
        self.planner = CachePlanner(self.price.min_cache_tokens)

    def complete(
        self,
        segments: list[Segment],
        query: str,
        *,
        max_output_tokens: int = 512,
        cache_ttl: str = "5m",
    ) -> Completion:  # pragma: no cover - needs live API
        plan = self.planner.plan(segments)
        system_blocks = []
        for i, seg in enumerate(plan.ordered):
            if not seg.static:
                continue
            block = {"type": "text", "text": seg.text}
            if i == plan.breakpoint_index:
                ttl = "1h" if cache_ttl == "1h" else "5m"
                block["cache_control"] = {"type": "ephemeral", "ttl": ttl}
            system_blocks.append(block)

        volatile = "\n".join(s.text for s in plan.ordered if not s.static)
        user_text = f"{volatile}\n\n{query}".strip()

        t0 = time.monotonic()
        resp = self._client.messages.create(
            model=self.model,
            max_tokens=max_output_tokens,
            system=system_blocks,
            messages=[{"role": "user", "content": user_text}],
        )
        latency = (time.monotonic() - t0) * 1000
        u = resp.usage
        usage = Usage(
            uncached_input=u.input_tokens,
            cached_read=getattr(u, "cache_read_input_tokens", 0) or 0,
            cache_write=getattr(u, "cache_creation_input_tokens", 0) or 0,
            output=u.output_tokens,
        )
        text = "".join(b.text for b in resp.content if b.type == "text")
        return Completion(text, usage, latency, usage.cached_read > 0, self.model)
