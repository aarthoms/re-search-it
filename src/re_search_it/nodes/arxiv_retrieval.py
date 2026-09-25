"""Node: fetch candidate metadata from arXiv — direct ID lookup or topic search."""

from re_search_it.state import PaperState
from re_search_it.tools.arxiv_client import get_by_id, search_by_topic

MAX_RESULTS_PER_TERM = 5


def arxiv_retrieval(state: PaperState) -> PaperState:
    if state.get("is_direct_id"):
        paper = get_by_id(state["arxiv_id"])
        if paper is None:
            return {**state, "error": f"No arXiv paper found for ID: {state['arxiv_id']}"}
        return {**state, "selected_paper": paper, "candidates": [paper]}

    seen_ids: set[str] = set()
    candidates: list[dict] = []
    for term in state.get("search_terms", [state["query"]]):
        for paper in search_by_topic(term, max_results=MAX_RESULTS_PER_TERM):
            if paper["arxiv_id"] not in seen_ids:
                seen_ids.add(paper["arxiv_id"])
                candidates.append(paper)

    if not candidates:
        return {**state, "error": f"No arXiv candidates found for query: {state['query']}"}

    return {**state, "candidates": candidates}
