"""Benchmark 2 -- context engineering.

Builds a noisy retrieval context: a handful of documents that actually answer
the query, buried among duplicates and irrelevant filler (the "noise drowning
out signal"). Runs the context-engineering pipeline and
measures (a) token reduction and (b) answer-relevant recall -- i.e. did we keep
the documents that contain the answer while cutting the bloat?

Run: python benchmarks/bench_context.py
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from tokensmith import Document, count_tokens, fit_to_budget  # noqa: E402
from tokensmith.context.compress import dedup_documents, order_for_recency  # noqa: E402

QUERY = "what is the refund window for digital subscription products"

# 3 gold docs that answer the query (contain the key facts).
GOLD = [
    Document("gold-1", "Refund policy: digital subscription products are "
             "eligible for a full refund within 14 days of purchase, provided "
             "fewer than 3 sessions were used."),
    Document("gold-2", "For subscription refunds, the refund window is 14 "
             "calendar days. Requests must be filed through the billing portal "
             "and are processed within 5 business days."),
    Document("gold-3", "Digital products differ from physical goods: the "
             "refund eligibility window for subscriptions is fourteen days "
             "from the activation date, not the order date."),
]
# Noise: irrelevant filler + near-duplicates of the gold docs.
NOISE = [
    Document(f"noise-{i}",
             "Our company was founded in 2009 and operates data centers across "
             "four regions. " * 6)
    for i in range(20)
]
DUPES = [Document(f"dup-{i}", GOLD[0].text) for i in range(6)]


def recall(selected_ids: set[str]) -> float:
    gold_ids = {d.id for d in GOLD}
    return len(gold_ids & selected_ids) / len(gold_ids)


def main():
    docs = GOLD + DUPES + NOISE
    raw_tokens = sum(count_tokens(d.text) for d in docs)

    deduped = dedup_documents(docs, threshold=0.8)
    budget = 700
    res = fit_to_budget(QUERY, deduped, budget_tokens=budget)
    ordered = order_for_recency(res.selected)

    kept_ids = {d.id for d in ordered}
    result = {
        "query": QUERY,
        "docs_in": len(docs),
        "docs_after_dedup": len(deduped),
        "docs_kept": len(ordered),
        "raw_context_tokens": raw_tokens,
        "engineered_context_tokens": res.used_tokens,
        "token_reduction": round(1 - res.used_tokens / raw_tokens, 3),
        "budget_tokens": budget,
        "budget_utilization": round(res.utilization, 3),
        "gold_recall": round(recall(kept_ids), 3),
        "compressed_docs": res.compressed,
        "kept_doc_ids": sorted(kept_ids),
    }

    print("=" * 64)
    print("BENCHMARK 2: Context engineering")
    print("=" * 64)
    print(f"Query: {QUERY}\n")
    print(f"Documents:        {result['docs_in']} -> "
          f"{result['docs_after_dedup']} after dedup -> "
          f"{result['docs_kept']} kept")
    print(f"Context tokens:   {raw_tokens:,} -> {res.used_tokens:,} "
          f"({result['token_reduction']:.0%} reduction)")
    print(f"Budget:           {budget} tokens "
          f"({result['budget_utilization']:.0%} utilized)")
    print(f"Gold recall:      {result['gold_recall']:.0%} "
          f"(kept the answer-bearing docs)")
    print(f"Kept docs:        {', '.join(result['kept_doc_ids'])}")

    out = os.path.join(os.path.dirname(__file__), "results_context.json")
    with open(out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nWrote {out}")
    return result


if __name__ == "__main__":
    main()
