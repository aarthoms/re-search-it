"""Pure scoring functions for the eval harness -- no network, no LLM, no I/O.

Kept separate from run_eval.py so the scoring logic itself (what counts as a
refusal, a pass, a hallucination) is unit-testable without ever touching the
real pipeline.
"""

import re

STRESS_TAGS = ["table", "multi-hop", "false-premise", "near-miss-oos", "numeric-reasoning", "multi-turn"]

# Only checked against the first two sentences of an answer (see is_refusal)
# so a correct answer that ends with a hedge/caveat isn't miscounted as one.
REFUSAL_PATTERNS = [re.compile(p, re.IGNORECASE) for p in [
    r"not (mentioned|discussed|covered|stated|provided|reported|specified)",
    r"does(?:n't| not) (?:discuss|mention|provide|report|include|cover|specify|state)",
    r"couldn't find",
    r"no (?:information|evidence|mention)",
    r"outside the scope",
    r"not in (?:this|the) paper",
]]


def _first_two_sentences(text: str) -> str:
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    return " ".join(sentences[:2])


def is_refusal(answer: dict) -> bool:
    if not answer.get("grounded") and not answer.get("sources"):
        return True
    lead = _first_two_sentences(answer.get("text", ""))
    return any(p.search(lead) for p in REFUSAL_PATTERNS)


def keyword_score(text: str, groups: list[list[str]]) -> float:
    if not groups:
        return 1.0
    text_lower = text.lower()
    hits = sum(1 for group in groups if any(variant.lower() in text_lower for variant in group))
    return hits / len(groups)


def forbidden_hit(text: str, forbidden: list[str] | None) -> bool:
    if not forbidden:
        return False
    text_lower = text.lower()
    return any(f.lower() in text_lower for f in forbidden)


def section_hit(sources: list[dict], expected: list[str] | None) -> bool | None:
    if not expected:
        return None
    source_sections = [s["section"].lower() for s in sources]
    expected_lower = [e.lower() for e in expected]
    return any(exp in sec or sec in exp for sec in source_sections for exp in expected_lower)


def score_item(item: dict, answer: dict) -> dict:
    text = answer.get("text", "")
    refused = is_refusal(answer)
    hit_forbidden = forbidden_hit(text, item.get("forbidden"))
    answerable = item["answerable"]

    kw_score = keyword_score(text, item["keywords"]) if answerable and item.get("keywords") else None
    sec_hit = section_hit(answer.get("sources", []), item.get("sections"))

    if answerable:
        # A false-premise item (answerable=true, no forbidden hit) also
        # passes by refusing cleanly -- declining the false premise is the
        # correct behaviour, not a missed answer.
        passed = not hit_forbidden and (refused or (kw_score == 1.0))
    else:
        passed = refused and not hit_forbidden

    return {
        "id": item["id"],
        "difficulty": item["difficulty"],
        "stress": item.get("stress", []),
        "answerable": answerable,
        "refused": refused,
        "forbidden_hit": hit_forbidden,
        "keyword_score": kw_score,
        "section_hit": sec_hit,
        "grounded": answer.get("grounded"),
        "confidence": answer.get("confidence"),
        "passed": passed,
    }


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _metrics_for(results: list[dict], latencies: dict[str, float] | None = None) -> dict:
    if not results:
        return {}
    answerable = [r for r in results if r["answerable"]]
    unanswerable = [r for r in results if not r["answerable"]]
    non_refused_answerable = [r for r in answerable if not r["refused"] and r["keyword_score"] is not None]
    section_checked = [r for r in results if r["section_hit"] is not None]
    grounded_checked = [r for r in answerable if r["grounded"] is not None]

    refusal_correct = [
        (not r["refused"]) if r["answerable"] else r["refused"]
        for r in results
    ]
    hallucinated = [
        r["forbidden_hit"] or (not r["answerable"] and not r["refused"])
        for r in results
    ]

    metrics = {
        "n": len(results),
        "pass_rate": _mean([1.0 if r["passed"] else 0.0 for r in results]),
        "refusal_accuracy": _mean([1.0 if c else 0.0 for c in refusal_correct]),
        "false_refusal_rate": _mean([1.0 if r["refused"] else 0.0 for r in answerable]) if answerable else None,
        "hallucination_rate": _mean([1.0 if h else 0.0 for h in hallucinated]),
        "mean_keyword_score": _mean([r["keyword_score"] for r in non_refused_answerable]),
        "section_hit_rate": _mean([1.0 if r["section_hit"] else 0.0 for r in section_checked]),
        "grounded_rate": _mean([1.0 if r["grounded"] else 0.0 for r in grounded_checked]),
    }
    if latencies:
        item_latencies = [latencies[r["id"]] for r in results if r["id"] in latencies]
        metrics["mean_latency_s"] = _mean(item_latencies)
    return metrics


def summarize(results: list[dict], latencies: dict[str, float] | None = None) -> dict:
    """Compute metrics per difficulty, for core (easy+medium), overall, and
    per-stress-tag pass rates among hard items."""
    by_difficulty = {
        diff: _metrics_for([r for r in results if r["difficulty"] == diff], latencies)
        for diff in ("easy", "medium", "hard")
    }
    core = _metrics_for([r for r in results if r["difficulty"] in ("easy", "medium")], latencies)
    overall = _metrics_for(results, latencies)

    hard = [r for r in results if r["difficulty"] == "hard"]
    stress_pass_rates = {
        tag: _mean([1.0 if r["passed"] else 0.0 for r in hard if tag in r["stress"]])
        for tag in STRESS_TAGS
    }

    return {
        "by_difficulty": by_difficulty,
        "core": core,
        "stress_by_tag": stress_pass_rates,
        "overall": overall,
    }
