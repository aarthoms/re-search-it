"""LangGraph wiring for the retrieval stage: query understanding -> arXiv fetch -> ranking.

Direct-ID queries skip ranking (there's nothing to rank against). Topic queries
are expanded, searched, and reranked. A zero-candidates/not-found result sets
state["error"] and routes straight to END instead of crashing downstream nodes.
"""

from langgraph.graph import END, StateGraph

from re_search_it.nodes.arxiv_retrieval import arxiv_retrieval
from re_search_it.nodes.chunk_embed import chunk_embed
from re_search_it.nodes.fetch_parse import fetch_parse
from re_search_it.nodes.query_understanding import query_understanding
from re_search_it.nodes.selection_ranking import selection_ranking
from re_search_it.state import PaperState


def _route_after_retrieval(state: PaperState) -> str:
    if state.get("error"):
        return "failed"
    if state.get("is_direct_id"):
        return "resolved"
    return "needs_ranking"


def _route_after_fetch_parse(state: PaperState) -> str:
    return "failed" if state.get("error") else "needs_chunking"


def _route_after_chunk_embed(state: PaperState) -> str:
    return "failed" if state.get("error") else "done"


def build_retrieval_graph():
    graph = StateGraph(PaperState)

    graph.add_node("query_understanding", query_understanding)
    graph.add_node("arxiv_retrieval", arxiv_retrieval)
    graph.add_node("selection_ranking", selection_ranking)
    graph.add_node("fetch_parse", fetch_parse)
    graph.add_node("chunk_embed", chunk_embed)

    graph.set_entry_point("query_understanding")
    graph.add_edge("query_understanding", "arxiv_retrieval")
    graph.add_conditional_edges(
        "arxiv_retrieval",
        _route_after_retrieval,
        {
            "failed": END,
            "resolved": "fetch_parse",
            "needs_ranking": "selection_ranking",
        },
    )
    graph.add_edge("selection_ranking", "fetch_parse")
    graph.add_conditional_edges(
        "fetch_parse",
        _route_after_fetch_parse,
        {
            "failed": END,
            "needs_chunking": "chunk_embed",
        },
    )
    graph.add_conditional_edges(
        "chunk_embed",
        _route_after_chunk_embed,
        {
            "failed": END,
            "done": END,
        },
    )

    return graph.compile()
