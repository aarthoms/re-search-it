"""Node: generate a grounded answer from the best evidence found, update conversation history."""

from re_search_it.state import PaperState
from re_search_it.tools.cohere_client import answer_question

TOP_N_FOR_ANSWER = 4


def answer(state: PaperState) -> PaperState:
    question = state["question"]
    evidence = state.get("retrieved_chunks", [])[:TOP_N_FOR_ANSWER]
    history = state.get("conversation_history", [])

    if not evidence:
        text = "I couldn't find anything in this paper relevant to that question."
    else:
        text = answer_question(question, evidence, history)

    updated_history = history + [
        {"role": "user", "content": question},
        {"role": "assistant", "content": text},
    ]

    return {**state, "answer": text, "conversation_history": updated_history}
