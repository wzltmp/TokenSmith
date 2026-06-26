"""TokenSmith -- a provider-agnostic toolkit for prompt-cache optimization
and context engineering.

Built around two well-documented problems in production LLM systems:
  * Heavy scaffolding, underutilized caching: most input tokens are repeated
    system-prompt scaffolding, yet only a minority of cache-capable calls
    actually cache -> the caching layer.
  * Exploding context: context windows reached ~2M tokens, so quality (not
    volume) is the limiting factor -> the context-engineering layer.
"""
from .caching.lint import (
    CacheBustDetector,
    Finding,
    LintReport,
    lint_prompt,
    lint_text,
)
from .caching.planner import CachePlan, CachePlanner, Segment
from .context.budget import BudgetResult, fit_to_budget
from .context.compress import (
    Document,
    compress_document,
    dedup_documents,
    order_for_recency,
    rank_by_relevance,
)
from .economics import Usage, VolumeReport, call_cost, project_volume
from .pipeline import Pipeline, RunReport
from .pricing import MODELS, ModelPrice, get_price
from .providers import MockProvider, Provider, get_provider
from .tokenizer import count_tokens

__version__ = "0.1.0"

__all__ = [
    "CachePlan", "CachePlanner", "Segment",
    "CacheBustDetector", "Finding", "LintReport", "lint_prompt", "lint_text",
    "BudgetResult", "fit_to_budget",
    "Document", "compress_document", "dedup_documents", "order_for_recency",
    "rank_by_relevance",
    "Usage", "VolumeReport", "call_cost", "project_volume",
    "Pipeline", "RunReport",
    "MODELS", "ModelPrice", "get_price",
    "MockProvider", "Provider", "get_provider",
    "count_tokens", "__version__",
]
