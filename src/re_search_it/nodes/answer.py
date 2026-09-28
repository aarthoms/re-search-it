"""Node: generate a grounded answer from the best evidence found, update conversation history.

Returns a structured contract rather than a bare string -- {text, sources,
grounded, confidence} -- so the CLI (or any other consumer) can render
sources and confidence separately from the prose answer, instead of the two
being tangled together in one block of text.
"""

from re_search_it.nodes.retrieve_chunks import RELEVANCE_FLOOR
from re_search_it.state import PaperState
from re_search_it.tools.cohere_client import answer_question

# retrieve_chunks.py already curates the right number for the question's
# complexity (5 direct / 8 decomposed, some neighbor-expanded into wider
# passages) -- this just needs to be >= its largest output so answer
# generation doesn't re-truncate evidence retrieval deliberately kept.
TOP_N_FOR_ANSWER = 8
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
    grounded = bool(state.get("evidence_sufficient"))

    if not evidence:
        text = "I couldn't find anything in this paper relevant to that question."
        sources = []
    else:
        text = answer_question(question, evidence, history)
        sources = [{"section": c["section"], "chunk_id": c["id"]} for c in evidence]
        if not grounded:
            # Code-enforced, not prompt-only: the prompt already asks the
            # model to hedge when evidence is weak, but nothing verified it
            # actually did -- a low-but-nonzero-score chunk set could still
            # produce a confident-sounding answer the model itself never
            # flagged as shaky. This disclaimer can't be silently skipped.
            top_score = evidence[0]["relevance_score"]
            text = (
                f"(Low-confidence -- the best evidence I found scored {top_score:.2f} "
                f"relevance, below what I'd normally trust.)\n\n{text}"
            )

    answer_result = {
        "text": text,
        "sources": sources,
        "grounded": grounded,
        "confidence": _confidence(evidence[0]["relevance_score"]) if evidence else "low",
    }

    # An ungrounded answer isn't added to history -- letting a shaky earlier
    # answer anchor later turns (the model treating its own weak guess as an
    # established fact) is worse than a follow-up question losing a bit of
    # continuity.
    updated_history = history
    if grounded:
        updated_history = history + [
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer_result["text"]},
        ]

    return {**state, "answer": answer_result, "conversation_history": updated_history}
