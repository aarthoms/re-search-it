"""Node: generate a grounded answer from the best evidence found, update conversation history.

Returns a structured contract rather than a bare string -- {text, sources,
grounded, confidence} -- so the CLI (or any other consumer) can render
sources and confidence separately from the prose answer, instead of the two
being tangled together in one block of text.
"""

from re_search_it.nodes.retrieve_chunks import RELEVANCE_FLOOR
from re_search_it.state import PaperState
from re_search_it.tools.cohere_client import answer_question

TOP_N_FOR_ANSWER = 4
HIGH_CONFIDENCE_FLOOR = 0.4


def _confidence(top_score: float) -> str:
    if top_score >= HIGH_CONFIDENCE_FLOOR:
        return "high"
    if top_score >= RELEVANCE_FLOOR:
        return "medium"
    return "low"


def answer(state: PaperState) -> PaperState:
    question = state["question"]
    evidence = state.get("retrieved_chunks", [])[:TOP_N_FOR_ANSWER]
    history = state.get("conversation_history", [])

    if not evidence:
        answer_result = {
            "text": "I couldn't find anything in this paper relevant to that question.",
            "sources": [],
            "grounded": False,
            "confidence": "low",
        }
    else:
        text = answer_question(question, evidence, history)
        answer_result = {
            "text": text,
            "sources": [{"section": c["section"], "chunk_id": c["id"]} for c in evidence],
            "grounded": bool(state.get("evidence_sufficient")),
            "confidence": _confidence(evidence[0]["relevance_score"]),
        }

    updated_history = history + [
        {"role": "user", "content": question},
        {"role": "assistant", "content": answer_result["text"]},
    ]

    return {**state, "answer": answer_result, "conversation_history": updated_history}
