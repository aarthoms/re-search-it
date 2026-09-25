"""Node: dense-retrieve each subquery (with soft section hints), merge, dedupe, rerank.

Section hints are a soft preference, not a hard filter -- our section-split
heuristic misfires on some papers (e.g. Roman-numeral headings), so a hard
`where` filter could silently hide the right chunk. Instead we run a
section-scoped query AND a whole-paper query per subquery and merge both.

Accumulates across retrieval rounds: chunks found in an earlier round (before
a refine_query loop) are merged with newly retrieved ones before reranking,
so a refinement round adds evidence rather than replacing it.
"""

from re_search_it.state import PaperState
from re_search_it.tools.cohere_client import embed, rerank
from re_search_it.tools.vector_store import query_chunks

TOP_K_PER_SUBQUERY = 8
FINAL_TOP_N = 5
# Chunk-level rerank scores run lower than paper-level ones (short excerpts vs.
# dense abstracts), so this floor is looser than selection_ranking's 0.3.
RELEVANCE_FLOOR = 0.15
MAX_RETRIEVAL_ROUNDS = 2


def _chunk_from_result(collection_result: dict, i: int) -> dict:
    meta = collection_result["metadatas"][0][i]
    return {
        "id": collection_result["ids"][0][i],
        "text": collection_result["documents"][0][i],
        "section": meta["section"],
        "chunk_index": meta["chunk_index"],
    }


def retrieve_chunks(state: PaperState) -> PaperState:
    plan = state["retrieval_plan"]
    collection_id = state["vector_collection_id"]
    section_hints = plan.get("section_hints") or []

    candidates: dict[str, dict] = {c["id"]: c for c in state.get("retrieved_chunks", [])}

    for subquery in plan["subqueries"]:
        query_vec = embed([subquery], input_type="search_query")[0]

        if section_hints:
            hinted = query_chunks(
                collection_id, query_vec, top_k=TOP_K_PER_SUBQUERY,
                where={"section": {"$in": section_hints}},
            )
            for i in range(len(hinted["ids"][0])):
                chunk = _chunk_from_result(hinted, i)
                candidates[chunk["id"]] = chunk

        broad = query_chunks(collection_id, query_vec, top_k=TOP_K_PER_SUBQUERY)
        for i in range(len(broad["ids"][0])):
            chunk = _chunk_from_result(broad, i)
            candidates.setdefault(chunk["id"], chunk)

    candidate_list = list(candidates.values())
    if not candidate_list:
        return {**state, "retrieved_chunks": [], "evidence_sufficient": False}

    ranked = rerank(state["question"], [c["text"] for c in candidate_list], top_n=FINAL_TOP_N)
    retrieved_chunks = []
    for r in ranked:
        chunk = candidate_list[r["index"]]
        retrieved_chunks.append({**chunk, "relevance_score": r["relevance_score"]})

    evidence_sufficient = bool(retrieved_chunks) and retrieved_chunks[0]["relevance_score"] >= RELEVANCE_FLOOR

    return {
        **state,
        "retrieved_chunks": retrieved_chunks,
        "evidence_sufficient": evidence_sufficient,
        "retrieval_rounds": state.get("retrieval_rounds", 0) + 1,
    }
