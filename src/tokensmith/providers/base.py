"""Provider abstraction.

A ``Provider`` accepts a list of prompt ``Segment``s (some marked static /
cacheable) plus a user query, and returns a ``Completion`` carrying the text
and a ``Usage`` token breakdown. This is the seam that makes the toolkit
provider-agnostic: the same pipeline runs against the deterministic
``MockProvider`` (no keys, used for tests and the demo) or the real
Anthropic / OpenAI adapters.
"""
from __future__ import annotations

import abc
from dataclasses import dataclass

from ..caching.planner import Segment
from ..economics import Usage


@dataclass
class Completion:
    text: str
    usage: Usage
    latency_ms: float
    cache_hit: bool
    model: str


class Provider(abc.ABC):
    def __init__(self, model: str):
        self.model = model

    @abc.abstractmethod
    def complete(
        self,
        segments: list[Segment],
        query: str,
        *,
        max_output_tokens: int = 512,
        cache_ttl: str = "5m",
    ) -> Completion:
        ...
