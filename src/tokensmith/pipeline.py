"""End-to-end pipeline: context engineering -> cache-aware layout -> provider.

``Pipeline.run`` takes a query plus raw retrieved documents and a static
system scaffold, then:

1. **Context engineering** -- dedups documents, ranks them against the query,
   fits them to a token budget (compressing the marginal doc), and orders them
   to dodge "lost in the middle".
2. **Cache layout** -- keeps the static scaffold as a cacheable prefix and the
   engineered context as the volatile suffix.
3. **Dispatch** -- calls the chosen provider and returns a ``RunReport``
   comparing the naive prompt (raw docs, no caching) against the optimized one.
"""
from __future__ import annotations

from dataclasses import dataclass

from .caching.planner import Segment
from .context.budget import fit_to_budget
from .context.compress import Document, dedup_documents, order_for_recency
from .economics import Usage, call_cost
from .pricing import get_price
from .providers.base import Provider
from .tokenizer import count_tokens


@dataclass
class RunReport:
    answer: str
    naive_input_tokens: int
    optimized_input_tokens: int
    naive_cost: float
    optimized_cost: float
    latency_ms: float
    cache_hit: bool
    docs_in: int
    docs_kept: int

    @property
    def token_reduction(self) -> float:
        if self.naive_input_tokens == 0:
            return 0.0
        return 1 - self.optimized_input_tokens / self.naive_input_tokens

    @property
    def cost_reduction(self) -> float:
        if self.naive_cost == 0:
            return 0.0
        return 1 - self.optimized_cost / self.naive_cost

    def render(self) -> str:
        return (
            f"Docs: {self.docs_in} -> {self.docs_kept} kept\n"
            f"Input tokens: {self.naive_input_tokens:,} -> "
            f"{self.optimized_input_tokens:,} ({self.token_reduction:.0%} less)\n"
            f"Cost/call: ${self.naive_cost:.5f} -> ${self.optimized_cost:.5f} "
            f"({self.cost_reduction:.0%} less)\n"
            f"Latency: {self.latency_ms:.0f} ms  | cache hit: {self.cache_hit}"
        )


class Pipeline:
    def __init__(
        self,
        provider: Provider,
        *,
        context_budget_tokens: int = 2000,
        dedup_threshold: float = 0.8,
    ):
        self.provider = provider
        self.context_budget_tokens = context_budget_tokens
        self.dedup_threshold = dedup_threshold
        self.price = get_price(provider.model)

    def run(
        self,
        query: str,
        scaffold: str,
        documents: list[Document],
        *,
        max_output_tokens: int = 400,
    ) -> RunReport:
        # --- context engineering ---
        deduped = dedup_documents(documents, self.dedup_threshold)
        budgeted = fit_to_budget(query, deduped, self.context_budget_tokens)
        ordered = order_for_recency(budgeted.selected)

        context_text = "\n\n".join(d.text for d in ordered)
        segments = [
            Segment("scaffold", scaffold, static=True),
            Segment("context", context_text, static=False),
        ]
        completion = self.provider.complete(
            segments, query, max_output_tokens=max_output_tokens)

        # --- naive baseline: raw docs, no caching, full prompt every call ---
        naive_doc_tokens = sum(count_tokens(d.text) for d in documents)
        naive_input = (
            count_tokens(scaffold) + naive_doc_tokens + count_tokens(query))
        naive_usage = Usage(
            uncached_input=naive_input, output=max_output_tokens)
        naive_cost = call_cost(naive_usage, self.price)

        return RunReport(
            answer=completion.text,
            naive_input_tokens=naive_input,
            optimized_input_tokens=completion.usage.total_input,
            naive_cost=naive_cost,
            optimized_cost=call_cost(completion.usage, self.price),
            latency_ms=completion.latency_ms,
            cache_hit=completion.cache_hit,
            docs_in=len(documents),
            docs_kept=len(ordered),
        )
