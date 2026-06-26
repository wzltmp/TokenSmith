"""Cache-aware prompt layout.

In production agent systems, the bulk of input tokens (often ~69% in measured
traces) are system-prompt scaffolding, yet only a minority of calls that
*could* cache actually do. The most common culprit is prompt *layout*: dynamic
content injected too early, or stable blocks reordered/rewritten between
requests, breaking the prefix reuse that facilitates caching.

``CachePlanner`` models a prompt as ordered ``Segment``s, then reorders them
so every stable (cacheable) block precedes the first volatile one, producing
the longest possible reusable prefix. It also flags layout anti-patterns.
"""
from __future__ import annotations

from dataclasses import dataclass

from ..tokenizer import count_tokens


@dataclass
class Segment:
    """One piece of a prompt."""
    name: str
    text: str
    static: bool  # True = identical across requests (cacheable)

    @property
    def tokens(self) -> int:
        return count_tokens(self.text)


@dataclass
class CachePlan:
    ordered: list[Segment]
    breakpoint_index: int          # last segment included in the cached prefix
    cacheable_tokens: int
    volatile_tokens: int
    warnings: list[str]
    reordered: bool

    @property
    def cacheable_fraction(self) -> float:
        total = self.cacheable_tokens + self.volatile_tokens
        return 0.0 if total == 0 else self.cacheable_tokens / total

    def render(self) -> str:
        lines = [
            f"Cacheable prefix: {self.cacheable_tokens} tokens "
            f"({self.cacheable_fraction:.0%} of prompt)",
            f"Volatile suffix:  {self.volatile_tokens} tokens",
            f"Breakpoint after: "
            f"{self.ordered[self.breakpoint_index].name if self.breakpoint_index >= 0 else '(none)'}",
        ]
        if self.reordered:
            lines.append("Reordered segments to extend the cacheable prefix.")
        for w in self.warnings:
            lines.append(f"WARNING: {w}")
        return "\n".join(lines)


class CachePlanner:
    def __init__(self, min_cache_tokens: int = 1024):
        self.min_cache_tokens = min_cache_tokens

    def plan(self, segments: list[Segment]) -> CachePlan:
        warnings: list[str] = []

        # Detect the anti-pattern: a volatile block sitting *before* a static
        # one in the author's original ordering kills prefix reuse.
        first_volatile = next(
            (i for i, s in enumerate(segments) if not s.static), len(segments))
        static_after_volatile = any(
            s.static for s in segments[first_volatile:])
        if static_after_volatile:
            warnings.append(
                "Static blocks appear after volatile content; their cache "
                "prefix is broken. Reordering static-first.")

        # Stable sort: all static segments first, preserving original order.
        ordered = sorted(segments, key=lambda s: not s.static)
        reordered = [s.name for s in ordered] != [s.name for s in segments]

        cacheable_tokens = sum(s.tokens for s in ordered if s.static)
        volatile_tokens = sum(s.tokens for s in ordered if not s.static)
        breakpoint_index = max(
            (i for i, s in enumerate(ordered) if s.static), default=-1)

        if 0 < cacheable_tokens < self.min_cache_tokens:
            warnings.append(
                f"Cacheable prefix is {cacheable_tokens} tokens, below the "
                f"{self.min_cache_tokens}-token minimum; it will NOT be "
                f"cached. Consolidate static scaffolding or move more shared "
                f"context into the prefix.")

        return CachePlan(
            ordered=ordered,
            breakpoint_index=breakpoint_index,
            cacheable_tokens=cacheable_tokens,
            volatile_tokens=volatile_tokens,
            warnings=warnings,
            reordered=reordered,
        )
