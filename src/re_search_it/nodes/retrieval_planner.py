"""Node: decide direct vs. decomposed retrieval for a QA question, with section hints.

Standard RAG asks "which chunks are similar to this question?" This node asks
"what evidence do I need, and where does it likely live?" -- cheap for simple
questions (falls straight through to a single retrieval), and meaningfully
better for multi-hop ones ("did X outperform Y, and why?") that no single
embedding search reliably answers in full.
"""

from re_search_it.state import PaperState
from re_search_it.tools.cohere_client import plan_retrieval


def retrieval_planner(state: PaperState) -> PaperState:
    available_sections = list(state["parsed_sections"].keys())
    plan = plan_retrieval(state["question"], available_sections)

    return {
        **state,
        "retrieval_plan": plan.model_dump(),
        "retrieval_rounds": 0,
        "retrieved_chunks": [],
    }
