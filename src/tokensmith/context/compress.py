"""Context engineering: keep the signal, drop the noise.

The problem, stated precisely: context windows have grown to ~2M tokens, so
volume is no longer the bottleneck -- *quality* is. As prompts grow to include
more history, retrieved documents, tool outputs, and guardrails, noise and
redundancy can drown out the signal, especially when critical details get
buried deep in long inputs.

This module provides dependency-free, deterministic context engineering:

* ``dedup_documents``    - drop near-duplicate documents (Jaccard on shingles)
* ``rank_by_relevance``  - TF-IDF cosine scoring of documents vs. the query
* ``compress_document``  - extractive summarization keeping top sentences
* ``order_for_recency``  - mitigate "lost in the middle" by placing the most
                           relevant items at the start and end of the context

Everything is embedding-free so it runs anywhere; swap in real embeddings
behind the same interface for production.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass

from ..tokenizer import count_tokens, split_sentences

_WORD = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> list[str]:
    return _WORD.findall(text.lower())


def _shingles(text: str, n: int = 3) -> set[tuple[str, ...]]:
    words = _tokens(text)
    return {tuple(words[i:i + n]) for i in range(max(0, len(words) - n + 1))}


def jaccard(a: str, b: str) -> float:
    sa, sb = _shingles(a), _shingles(b)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


@dataclass
class Document:
    id: str
    text: str
    score: float = 0.0

    @property
    def tokens(self) -> int:
        return count_tokens(self.text)


def dedup_documents(docs: list[Document], threshold: float = 0.8) -> list[Document]:
    """Remove documents whose shingle-overlap with a kept doc exceeds threshold."""
    kept: list[Document] = []
    for d in docs:
        if all(jaccard(d.text, k.text) < threshold for k in kept):
            kept.append(d)
    return kept


def _tfidf_vectors(corpus: list[list[str]]) -> tuple[list[Counter], dict[str, float]]:
    n = len(corpus)
    df: Counter = Counter()
    for toks in corpus:
        df.update(set(toks))
    idf = {t: math.log((1 + n) / (1 + df[t])) + 1.0 for t in df}
    vecs = []
    for toks in corpus:
        tf = Counter(toks)
        length = max(1, len(toks))
        vecs.append(Counter({t: (c / length) * idf[t] for t, c in tf.items()}))
    return vecs, idf


def _cosine(a: Counter, b: Counter) -> float:
    if not a or not b:
        return 0.0
    common = set(a) & set(b)
    dot = sum(a[t] * b[t] for t in common)
    na = math.sqrt(sum(v * v for v in a.values()))
    nb = math.sqrt(sum(v * v for v in b.values()))
    return 0.0 if na == 0 or nb == 0 else dot / (na * nb)


def rank_by_relevance(query: str, docs: list[Document]) -> list[Document]:
    """Score each document by TF-IDF cosine similarity to the query (desc)."""
    corpus = [_tokens(query)] + [_tokens(d.text) for d in docs]
    vecs, _ = _tfidf_vectors(corpus)
    qv, dvs = vecs[0], vecs[1:]
    scored = [
        Document(d.id, d.text, _cosine(qv, dv)) for d, dv in zip(docs, dvs)
    ]
    return sorted(scored, key=lambda d: d.score, reverse=True)


def compress_document(query: str, text: str, keep_sentences: int = 3) -> str:
    """Extractive compression: keep the sentences most relevant to the query."""
    sentences = split_sentences(text)
    if len(sentences) <= keep_sentences:
        return text
    corpus = [_tokens(query)] + [_tokens(s) for s in sentences]
    vecs, _ = _tfidf_vectors(corpus)
    qv, svs = vecs[0], vecs[1:]
    scored = sorted(
        range(len(sentences)), key=lambda i: _cosine(qv, svs[i]), reverse=True)
    chosen = sorted(scored[:keep_sentences])  # keep original order
    return " ".join(sentences[i] for i in chosen)


def order_for_recency(docs: list[Document]) -> list[Document]:
    """Place the highest-scoring docs at the edges of the context window.

    Models attend most strongly to the start and end of long inputs ("lost in
    the middle"). Given docs sorted best-first, interleave so the top items sit
    at both ends and weaker items fall in the middle.
    """
    head, tail = [], []
    for i, d in enumerate(docs):
        (head if i % 2 == 0 else tail).append(d)
    return head + tail[::-1]
