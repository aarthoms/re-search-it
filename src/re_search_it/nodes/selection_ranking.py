"""Node: rerank topic-search candidates against the original query, pick the best match."""

from re_search_it.state import PaperState
from re_search_it.tools.cohere_client import rerank

# Below this, the top match is still shown but flagged as shaky.
RELEVANCE_FLOOR = 0.3
# Below THIS, the match is not a match -- committing to the full fetch/parse/
# chunk/embed/summarize pipeline for it would present garbage (e.g. a 0.03
# score) as if it were "the answer". Below this floor we refuse to select a
# paper at all rather than confidently answering from a non-match.
ANSWERABLE_FLOOR = 0.15


def selection_ranking(state: PaperState) -> PaperState:
    if state.get("is_direct_id"):
        return state

    candidates = state["candidates"]
    documents = [f"{c['title']}\n{c['summary']}" for c in candidates]
    ranked = rerank(state["query"], documents)

    ranked_candidates = [candidates[r["index"]] for r in ranked]
    for candidate, r in zip(ranked_candidates, ranked):
        candidate["relevance_score"] = r["relevance_score"]

    top_score = ranked_candidates[0]["relevance_score"]
    answerable = top_score >= ANSWERABLE_FLOOR
    print(f"[ranking] top relevance: {top_score:.3f}")

    return {
        **state,
        "candidates": ranked_candidates,
        "selected_paper": ranked_candidates[0] if answerable else None,
        "low_confidence": top_score < RELEVANCE_FLOOR,
        "answerable": answerable,
    }
