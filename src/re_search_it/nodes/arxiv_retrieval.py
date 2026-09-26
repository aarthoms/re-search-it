"""Node: fetch candidate metadata from arXiv — direct ID lookup or topic search."""

from re_search_it.state import PaperState
from re_search_it.tools.arxiv_client import get_by_id, search_by_topic

MAX_RESULTS_PER_TERM = 20


def _build_field_query(term: str, category: str | None) -> str:
    """Scope a keyword phrase to title/abstract fields, optionally AND'd with a category."""
    field_query = f'abs:"{term}" OR ti:"{term}"'
    if category:
        return f"({field_query}) AND cat:{category}"
    return field_query


def arxiv_retrieval(state: PaperState) -> PaperState:
    if state.get("is_direct_id"):
        paper = get_by_id(state["arxiv_id"])
        if paper is None:
            return {**state, "error": f"No arXiv paper found for ID: {state['arxiv_id']}"}
        return {**state, "selected_paper": paper, "candidates": [paper]}

    if state.get("candidates"):
        # query_understanding's exact-title fast path already resolved this
        # -- nothing to search for, let selection_ranking score/confirm it.
        return state

    category = state.get("category")
    seen_ids: set[str] = set()
    candidates: list[dict] = []
    for term in state.get("search_terms", [state["query"]]):
        field_query = _build_field_query(term, category)
        for paper in search_by_topic(field_query, max_results=MAX_RESULTS_PER_TERM):
            if paper["arxiv_id"] not in seen_ids:
                seen_ids.add(paper["arxiv_id"])
                candidates.append(paper)

    # Category filter may have been too strict (e.g. LLM mis-guessed it) — retry unfiltered.
    if not candidates and category:
        for term in state.get("search_terms", [state["query"]]):
            for paper in search_by_topic(_build_field_query(term, None), max_results=MAX_RESULTS_PER_TERM):
                if paper["arxiv_id"] not in seen_ids:
                    seen_ids.add(paper["arxiv_id"])
                    candidates.append(paper)

    if not candidates:
        return {**state, "error": f"No arXiv candidates found for query: {state['query']}"}

    return {**state, "candidates": candidates}
