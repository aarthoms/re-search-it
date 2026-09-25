"""Node: when retrieved evidence is weak, generate one targeted follow-up query.

E.g. an initial retrieval for "why did the authors claim their method was
better?" might surface only the results table -- this node asks the LLM what's
still missing (the authors' explanation) and turns that into a fresh subquery
for another retrieve_chunks pass. This is the constrained iterative-retrieval
step (V3); retrieve_chunks.MAX_RETRIEVAL_ROUNDS caps how many times it can fire.
"""

from re_search_it.state import PaperState
from re_search_it.tools.cohere_client import refine_query as refine_query_llm


def refine_query(state: PaperState) -> PaperState:
    evidence_so_far = [c["text"] for c in state.get("retrieved_chunks", [])]
    new_query = refine_query_llm(state["question"], evidence_so_far)

    plan = state["retrieval_plan"]
    return {**state, "retrieval_plan": {**plan, "subqueries": [new_query]}}
