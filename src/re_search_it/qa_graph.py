"""LangGraph wiring for the QA loop: plan -> retrieve/rerank -> (refine once if weak) -> answer.

    question
       |
   retrieval_planner  (direct vs. decomposed, section hints)
       |
   retrieve_chunks  <---------+
       |                      |
   evidence sufficient,       |
   or MAX_RETRIEVAL_ROUNDS    |
   reached?                   |
     | no                     |
     +---> refine_query ------+
     | yes
     v
    answer
       |
      END

Runs once per question against an already-populated PaperState (selected_paper,
parsed_sections, vector_collection_id must already be set by the retrieval
pipeline). Capped at MAX_RETRIEVAL_ROUNDS retrieval passes so a stubborn
question can't loop forever.
"""

from langgraph.graph import END, StateGraph

from re_search_it.nodes.answer import answer
from re_search_it.nodes.refine_query import refine_query
from re_search_it.nodes.retrieval_planner import retrieval_planner
from re_search_it.nodes.retrieve_chunks import MAX_RETRIEVAL_ROUNDS, retrieve_chunks
from re_search_it.state import PaperState


def _route_after_retrieve(state: PaperState) -> str:
    if state.get("evidence_sufficient") or state.get("retrieval_rounds", 0) >= MAX_RETRIEVAL_ROUNDS:
        return "answer"
    return "refine"


def build_qa_graph():
    graph = StateGraph(PaperState)

    graph.add_node("retrieval_planner", retrieval_planner)
    graph.add_node("retrieve_chunks", retrieve_chunks)
    graph.add_node("refine_query", refine_query)
    graph.add_node("answer", answer)

    graph.set_entry_point("retrieval_planner")
    graph.add_edge("retrieval_planner", "retrieve_chunks")
    graph.add_conditional_edges(
        "retrieve_chunks",
        _route_after_retrieve,
        {
            "answer": "answer",
            "refine": "refine_query",
        },
    )
    graph.add_edge("refine_query", "retrieve_chunks")
    graph.add_edge("answer", END)

    return graph.compile()
