"""Node: detect direct arXiv ID vs. topic search, expand topic queries into search terms."""

from re_search_it.state import PaperState
from re_search_it.tools.arxiv_client import is_arxiv_id
from re_search_it.tools.cohere_client import expand_query


def query_understanding(state: PaperState) -> PaperState:
    query = state["query"].strip()

    if is_arxiv_id(query):
        return {**state, "is_direct_id": True, "arxiv_id": query}

    expansion = expand_query(query)
    return {
        **state,
        "is_direct_id": False,
        "search_terms": expansion["terms"],
        "category": expansion["category"],
    }
