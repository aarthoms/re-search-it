"""LangGraph wiring: query understanding -> arXiv fetch -> ranking -> parse -> chunk/embed -> summarize.

Direct-ID queries skip ranking (there's nothing to rank against). Topic queries
are expanded, searched, and reranked. Any stage that sets state["error"] routes
straight to END instead of letting downstream nodes crash on missing data.
"""

from langgraph.graph import END, StateGraph

from re_search_it.nodes.arxiv_retrieval import arxiv_retrieval
from re_search_it.nodes.chunk_embed import chunk_embed
from re_search_it.nodes.fetch_parse import fetch_parse
from re_search_it.nodes.query_understanding import query_understanding
from re_search_it.nodes.selection_ranking import selection_ranking
from re_search_it.nodes.summarize import summarize
from re_search_it.state import PaperState


def _route_after_retrieval(state: PaperState) -> str:
    if state.get("error"):
        return "failed"
    if state.get("is_direct_id"):
        return "resolved"
    return "needs_ranking"


def _route_after_ranking(state: PaperState) -> str:
    if state.get("answerable", True):
        return "resolved"
    # Best match was below ANSWERABLE_FLOOR -- don't spend a full fetch/
    # parse/chunk/embed/summarize pass presenting a non-match as an answer.
    return "unanswerable"


def _route_after_fetch_parse(state: PaperState) -> str:
    return "failed" if state.get("error") else "needs_chunking"


def _route_after_chunk_embed(state: PaperState) -> str:
    return "failed" if state.get("error") else "needs_summary"


def build_retrieval_graph():
    graph = StateGraph(PaperState)

    graph.add_node("query_understanding", query_understanding)
    graph.add_node("arxiv_retrieval", arxiv_retrieval)
    graph.add_node("selection_ranking", selection_ranking)
    graph.add_node("fetch_parse", fetch_parse)
    graph.add_node("chunk_embed", chunk_embed)
    graph.add_node("summarize", summarize)

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
    graph.add_conditional_edges(
        "selection_ranking",
        _route_after_ranking,
        {
            "resolved": "fetch_parse",
            "unanswerable": END,
        },
    )
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
            "needs_summary": "summarize",
        },
    )
    graph.add_edge("summarize", END)

    return graph.compile()
