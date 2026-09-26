"""Node: detect direct arXiv ID vs. topic search, expand topic queries into search terms."""

from re_search_it.state import PaperState
from re_search_it.tools.arxiv_client import is_arxiv_id, search_by_exact_title
from re_search_it.tools.cohere_client import expand_query


def query_understanding(state: PaperState) -> PaperState:
    query = state["query"].strip()

    if is_arxiv_id(query):
        return {**state, "is_direct_id": True, "arxiv_id": query}

    # Fast path: the query may already BE a paper's title verbatim (e.g.
    # "Attention is all you need"). One title-field search settles it with
    # zero LLM calls and zero query-expansion fan-out, instead of running
    # the full 6-formulation semantic discovery pass for what's really an
    # exact lookup -- and avoids landing on a lookalike paper by relevance
    # when the real one was a direct title match away.
    exact_match = search_by_exact_title(query)
    if exact_match:
        return {
            **state,
            "is_direct_id": False,
            "search_terms": [query],
            "category": None,
            "candidates": [exact_match],
        }

    expansion = expand_query(query)
    return {
        **state,
        "is_direct_id": False,
        "search_terms": expansion["terms"],
        "category": expansion["category"],
    }
