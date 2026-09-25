"""Node: rerank topic-search candidates against the original query, pick the best match."""

from re_search_it.state import PaperState
from re_search_it.tools.cohere_client import rerank


def selection_ranking(state: PaperState) -> PaperState:
    if state.get("is_direct_id"):
        return state

    candidates = state["candidates"]
    documents = [f"{c['title']}\n{c['summary']}" for c in candidates]
    ranked = rerank(state["query"], documents)

    ranked_candidates = [candidates[r["index"]] for r in ranked]
    for candidate, r in zip(ranked_candidates, ranked):
        candidate["relevance_score"] = r["relevance_score"]

    return {**state, "candidates": ranked_candidates, "selected_paper": ranked_candidates[0]}
