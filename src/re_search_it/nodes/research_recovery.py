"""Node: bounded recovery when arXiv retrieval fails or returns a weak match.

Reached ONLY via graph.py's routing, when arxiv_retrieval found zero
candidates OR selection_ranking's answerability gate rejected the top match
-- never on the normal path, so the common case pays nothing extra.

Resolves unfamiliar terminology (memory lookup, then LLM discovery + arXiv-
evidence validation if memory misses -- see tools/research_discovery.py),
persists validated concepts for future queries, then retries arXiv with
several loose broadened formulations. Bounded to ONE attempt per query via
state["recovery_attempted"] (checked by graph.py's routing) so a genuinely
unanswerable query can't loop forever burning LLM/API calls.

This node does NOT bypass the existing answerability gate: it only expands
the candidate pool and hands control back to selection_ranking, which is
still the sole authority on whether the (possibly larger) pool contains an
actual answerable match.
"""

from re_search_it import research_memory
from re_search_it.state import PaperState
from re_search_it.tools.arxiv_client import search_by_topic
from re_search_it.tools.cohere_client import generate_recovery_queries
from re_search_it.tools.research_discovery import discover_and_persist

MAX_RESULTS_PER_FORMULATION = 15


def research_recovery(state: PaperState) -> PaperState:
    query = state["query"]
    print(f"[arxiv-recovery] retrying with expanded formulations for {query!r}")

    known = research_memory.find_known_concepts(query)
    if known:
        concept = known[0]
        print(f"[research-memory] HIT: {concept['canonical_name']}")
    else:
        print(f"[research-memory] MISS: {query!r} not in memory -- attempting discovery")
        concept = discover_and_persist(query, query)

    formulations = generate_recovery_queries(query, concept)

    existing = state.get("candidates", [])
    seen_ids = {c["arxiv_id"] for c in existing}
    recovered: list[dict] = []
    for phrase in formulations:
        for paper in search_by_topic(phrase, max_results=MAX_RESULTS_PER_FORMULATION):
            if paper["arxiv_id"] not in seen_ids:
                seen_ids.add(paper["arxiv_id"])
                recovered.append(paper)

    all_candidates = existing + recovered
    print(
        f"[arxiv-recovery] {len(formulations)} formulations -> "
        f"{len(recovered)} new candidates (pool: {len(existing)} -> {len(all_candidates)})"
    )

    if not all_candidates:
        return {
            **state,
            "recovery_attempted": True,
            "error": f"No arXiv candidates found for query: {query}",
        }

    return {**state, "recovery_attempted": True, "candidates": all_candidates, "error": None}
