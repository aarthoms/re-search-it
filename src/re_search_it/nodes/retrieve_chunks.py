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
from re_search_it.tools.chunker import contextualize
from re_search_it.tools.cohere_client import embed, rerank
from re_search_it.tools.vector_store import get_chunks_by_ids, query_chunks

TOP_K_PER_SUBQUERY = 8
# Decomposed questions ("did X outperform Y, and why?") genuinely need more
# evidence than a direct one-fact question -- 5 chunks (~1000 words) is
# often not enough to cover method + baseline + results + explanation, so
# those get more room.
FINAL_TOP_N_DIRECT = 5
FINAL_TOP_N_DECOMPOSED = 8
# How many of the top-ranked chunks get expanded with their neighbors
# (chunk_index +-1) -- bounded so a "deep" question doesn't balloon into a
# huge number of extra vector-store lookups.
NEIGHBOR_EXPANSION_COUNT = 3
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


def _expand_with_neighbors(chunk: dict, arxiv_id: str, collection_id: str) -> dict:
    """Merge in the immediately-preceding and immediately-following chunk
    (same section) so the model reads a contiguous passage instead of a
    200-word fragment that may cut off mid-thought. Chunk IDs are
    deterministic (f"{arxiv_id}_{section}_{chunk_index}"), so neighbors can
    be looked up directly by ID rather than another similarity search.

    Idempotent: a chunk from an EARLIER retrieval round can survive into a
    later one (retrieved_chunks carries forward across refine_query rounds,
    and this function's own output becomes next round's input), so without
    the `expanded` check an already-widened chunk re-selected in round 2
    would get expanded a second time on top of its already-merged text --
    duplicating the neighbor content in what the model reads.
    """
    if chunk.get("expanded"):
        return chunk

    section, idx = chunk["section"], chunk["chunk_index"]
    neighbor_ids = [f"{arxiv_id}_{section}_{idx - 1}", f"{arxiv_id}_{section}_{idx + 1}"]
    fetched = get_chunks_by_ids(collection_id, neighbor_ids)

    by_index = {meta["chunk_index"]: doc for doc, meta in zip(fetched["documents"], fetched["metadatas"])}
    before = by_index.get(idx - 1, "")
    after = by_index.get(idx + 1, "")
    expanded_text = " ".join(t for t in (before, chunk["text"], after) if t)

    return {**chunk, "text": expanded_text, "expanded": True}


def retrieve_chunks(state: PaperState) -> PaperState:
    plan = state["retrieval_plan"]
    collection_id = state["vector_collection_id"]
    arxiv_id = state["selected_paper"]["arxiv_id"]
    section_hints = plan.get("section_hints") or []
    final_top_n = FINAL_TOP_N_DECOMPOSED if plan.get("mode") == "decomposed" else FINAL_TOP_N_DIRECT

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

    # Rerank on the same [section]-prefixed text the embedding saw -- chunks
    # passed on to the answer node keep their raw text (candidate_list is
    # unmodified).
    contextualized = [contextualize(c["section"], c["text"]) for c in candidate_list]
    ranked = rerank(state["question"], contextualized, top_n=final_top_n)
    retrieved_chunks = []
    for i, r in enumerate(ranked):
        chunk = candidate_list[r["index"]]
        if i < NEIGHBOR_EXPANSION_COUNT:
            chunk = _expand_with_neighbors(chunk, arxiv_id, collection_id)
        retrieved_chunks.append({**chunk, "relevance_score": r["relevance_score"]})

    evidence_sufficient = bool(retrieved_chunks) and retrieved_chunks[0]["relevance_score"] >= RELEVANCE_FLOOR

    return {
        **state,
        "retrieved_chunks": retrieved_chunks,
        "evidence_sufficient": evidence_sufficient,
        "retrieval_rounds": state.get("retrieval_rounds", 0) + 1,
    }
