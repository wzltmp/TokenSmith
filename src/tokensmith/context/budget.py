"""Token budgeting.

Given a token budget for retrieved context, greedily fit the highest-signal
documents first, optionally compressing the marginal document so it fits
rather than dropping it whole.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from ..tokenizer import count_tokens
from .compress import Document, compress_document, rank_by_relevance


@dataclass
class BudgetResult:
    selected: list[Document]
    used_tokens: int
    budget: int
    dropped: list[str] = field(default_factory=list)
    compressed: list[str] = field(default_factory=list)

    @property
    def utilization(self) -> float:
        return 0.0 if self.budget == 0 else self.used_tokens / self.budget


def fit_to_budget(
    query: str,
    docs: list[Document],
    budget_tokens: int,
    *,
    compress_marginal: bool = True,
) -> BudgetResult:
    ranked = rank_by_relevance(query, docs)
    selected: list[Document] = []
    used = 0
    dropped: list[str] = []
    compressed: list[str] = []

    for d in ranked:
        if used + d.tokens <= budget_tokens:
            selected.append(d)
            used += d.tokens
            continue
        remaining = budget_tokens - used
        if compress_marginal and remaining > 40:
            # Try to squeeze a compressed version into the leftover budget.
            shrunk = compress_document(query, d.text, keep_sentences=2)
            st = count_tokens(shrunk)
            if st <= remaining and st < d.tokens:
                selected.append(Document(d.id, shrunk, d.score))
                used += st
                compressed.append(d.id)
                continue
        dropped.append(d.id)

    return BudgetResult(selected, used, budget_tokens, dropped, compressed)
