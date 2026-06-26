"""`python -m tokensmith` -- quick demo of the full pipeline on the mock
provider, plus a one-line cost projection. Useful smoke test with no keys."""
from __future__ import annotations

from . import Document, MockProvider, Pipeline, get_price, project_volume


def main() -> None:
    print("TokenSmith demo (mock provider, no API key needed)\n")

    prov = MockProvider("claude-sonnet-4.6")
    pipe = Pipeline(prov, context_budget_tokens=600)
    scaffold = "You are a meticulous support agent. Follow policy. " * 250
    docs = [Document(str(i), f"Knowledge item {i}. " + "detail " * 60)
            for i in range(15)]
    docs.append(Document("dup", docs[0].text))
    rep = pipe.run("summarize knowledge item 3", scaffold, docs)
    print(rep.render())

    print("\nMonthly projection @ 50k requests:")
    price = get_price("claude-sonnet-4.6")
    vr = project_volume(static_tokens=8000, dynamic_input_tokens=600,
                        output_tokens=350, requests=50_000, price=price,
                        cache_hit_rate=0.95)
    print(vr.render())


if __name__ == "__main__":
    main()
