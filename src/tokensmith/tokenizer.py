"""Token counting.

Uses ``tiktoken`` when available for accurate counts, and falls back to a
calibrated character heuristic otherwise so the toolkit runs with zero
dependencies. The heuristic (~3.9 chars/token for English prose) is close
enough for cost/latency planning, which is what this library is for.
"""
from __future__ import annotations

import functools
import re

_CHARS_PER_TOKEN = 3.9


@functools.lru_cache(maxsize=4)
def _encoder(model: str = "cl100k_base"):
    try:
        import tiktoken
    except Exception:  # pragma: no cover - exercised only when tiktoken absent
        return None
    try:
        return tiktoken.get_encoding(model)
    except Exception:  # pragma: no cover
        return None


def count_tokens(text: str, model: str = "cl100k_base") -> int:
    """Return an estimated token count for ``text``."""
    if not text:
        return 0
    enc = _encoder(model)
    if enc is not None:
        return len(enc.encode(text))
    # Heuristic fallback: blend char-based and word-based estimates.
    char_est = len(text) / _CHARS_PER_TOKEN
    word_est = len(re.findall(r"\S+", text)) * 1.3
    return max(1, round((char_est + word_est) / 2))


def split_sentences(text: str) -> list[str]:
    """Cheap sentence splitter used by the extractive compressor."""
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p.strip() for p in parts if p.strip()]
