"""Test suite. Runs with zero third-party deps (MockProvider + heuristic
tokenizer), so `pytest` is green out of the box."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pytest

from tokensmith import (
    CachePlanner,
    Document,
    MockProvider,
    Pipeline,
    Segment,
    call_cost,
    count_tokens,
    dedup_documents,
    fit_to_budget,
    get_price,
    project_volume,
    rank_by_relevance,
)
from tokensmith.economics import Usage


# ---------------- tokenizer ----------------

def test_token_count_monotonic():
    assert count_tokens("") == 0
    assert count_tokens("hello world") >= 1
    assert count_tokens("word " * 100) > count_tokens("word " * 10)


# ---------------- cache planner ----------------

def test_planner_reorders_static_first():
    segs = [
        Segment("vol", "x " * 50, static=False),
        Segment("sys", "policy " * 50, static=True),
    ]
    plan = CachePlanner(min_cache_tokens=10).plan(segs)
    assert plan.reordered
    assert plan.ordered[0].name == "sys"
    assert any("Reordering" in w or "after" in w for w in plan.warnings)


def test_planner_warns_on_short_prefix():
    segs = [Segment("sys", "short", static=True),
            Segment("vol", "q " * 20, static=False)]
    plan = CachePlanner(min_cache_tokens=1024).plan(segs)
    assert any("below" in w for w in plan.warnings)


def test_planner_cacheable_fraction():
    segs = [Segment("sys", "a " * 100, static=True),
            Segment("vol", "b " * 100, static=False)]
    plan = CachePlanner(min_cache_tokens=10).plan(segs)
    assert 0.0 < plan.cacheable_fraction < 1.0


# ---------------- economics ----------------

def test_call_cost_components():
    price = get_price("claude-sonnet-4.6")
    # 1M uncached input at $3 = $3
    assert call_cost(Usage(uncached_input=1_000_000), price) == pytest.approx(3.0)
    # 1M cached read at $0.30
    assert call_cost(Usage(cached_read=1_000_000), price) == pytest.approx(0.30)


def test_project_volume_saves_money():
    price = get_price("claude-sonnet-4.6")
    rep = project_volume(
        static_tokens=8000, dynamic_input_tokens=500, output_tokens=300,
        requests=10_000, price=price, cache_hit_rate=0.95)
    assert rep.cached_cost < rep.naive_cost
    assert 0 < rep.savings_pct < 1


def test_short_prefix_cannot_cache():
    price = get_price("claude-sonnet-4.6")  # min 1024
    rep = project_volume(
        static_tokens=200, dynamic_input_tokens=100, output_tokens=50,
        requests=100, price=price, cache_hit_rate=1.0)
    # prefix below minimum -> hit rate forced to 0 -> no savings
    assert rep.cache_hit_rate == 0.0
    assert rep.cached_cost == pytest.approx(rep.naive_cost)


# ---------------- context engineering ----------------

def test_dedup_removes_duplicates():
    docs = [
        Document("a", "the cat sat on the mat in the sun"),
        Document("b", "the cat sat on the mat in the sun"),
        Document("c", "quantum tunneling in semiconductors"),
    ]
    kept = dedup_documents(docs, threshold=0.8)
    assert len(kept) == 2


def test_relevance_ranking_orders_by_query():
    docs = [
        Document("net", "the python socket library handles tcp networking"),
        Document("cook", "to bake bread you need flour water yeast and salt"),
    ]
    ranked = rank_by_relevance("how do I open a tcp network socket", docs)
    assert ranked[0].id == "net"
    assert ranked[0].score >= ranked[1].score


def test_budget_respects_limit():
    docs = [Document(str(i), "lorem ipsum dolor sit amet " * 30)
            for i in range(10)]
    res = fit_to_budget("lorem", docs, budget_tokens=300)
    assert res.used_tokens <= 300
    assert len(res.selected) < len(docs)


# ---------------- mock provider caching ----------------

def test_mock_provider_cache_hit_after_first_call():
    t = [0.0]
    prov = MockProvider("claude-sonnet-4.6", clock=lambda: t[0])
    scaffold = Segment("sys", "policy guidance " * 400, static=True)
    vol = Segment("ctx", "user context " * 5, static=False)
    c1 = prov.complete([scaffold, vol], "question one")
    c2 = prov.complete([scaffold, vol], "question two")
    assert c1.cache_hit is False        # first call writes
    assert c2.cache_hit is True         # second call reads
    assert c2.usage.cached_read > 0
    assert c2.latency_ms < c1.latency_ms  # cache cuts latency


def test_mock_cache_expires():
    t = [0.0]
    prov = MockProvider("claude-sonnet-4.6", clock=lambda: t[0])
    scaffold = Segment("sys", "policy guidance " * 400, static=True)
    prov.complete([scaffold], "q")
    t[0] = 10_000  # advance well past the 5 min TTL
    c = prov.complete([scaffold], "q")
    assert c.cache_hit is False


# ---------------- pipeline ----------------

def test_pipeline_reduces_tokens_and_cost():
    prov = MockProvider("claude-sonnet-4.6")
    pipe = Pipeline(prov, context_budget_tokens=400)
    scaffold = "You are a support agent. " * 300
    docs = [Document(str(i), f"fact {i}: " + "filler text here " * 40)
            for i in range(12)]
    docs += [Document("dup", docs[0].text)]  # a duplicate to be removed
    rep = pipe.run("what is fact 3", scaffold, docs)
    assert rep.optimized_input_tokens < rep.naive_input_tokens
    assert rep.docs_kept < rep.docs_in
