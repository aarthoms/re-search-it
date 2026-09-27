"""Node: detect direct arXiv ID vs. exact title vs. topic search."""

from re_search_it import research_memory
from re_search_it.state import PaperState
from re_search_it.tools.arxiv_client import is_arxiv_id, search_by_exact_title, search_by_near_title
from re_search_it.tools.cohere_client import expand_query


def query_understanding(state: PaperState) -> PaperState:
    query = state["query"].strip()

    if is_arxiv_id(query):
        print("[lookup] mode: direct-id")
        return {**state, "is_direct_id": True, "arxiv_id": query}

    # Fast path: the query may already BE a paper's title verbatim (e.g.
    # "Attention is all you need"). One title-field search settles it with
    # zero LLM calls and zero query-expansion fan-out.
    exact_match = search_by_exact_title(query)
    if exact_match:
        print("[lookup] mode: title")
        print(f"[lookup] exact title match: {exact_match['title']}")
        return {
            **state,
            "is_direct_id": False,
            "search_terms": [query],
            "category": None,
            "candidates": [exact_match],
        }

    # Near-exact title: not confirmed exact, but title-word overlap is a
    # much stronger signal than generic semantic search. Seed the candidate
    # pool with these so a real (but non-verbatim-matching) title doesn't
    # lose to an unrelated lookalike purely by semantic relevance -- the
    # existing reranker (against the ORIGINAL query) still makes the final
    # call, this only ensures the right paper is IN the running.
    near_matches = search_by_near_title(query)
    if near_matches:
        print(f"[lookup] mode: near-title ({len(near_matches)} candidates)")

    # Cheap persistent-memory lookup (no network/LLM cost) before the LLM
    # expansion call -- if the query mentions a previously-learned concept
    # (e.g. "JEPA" resolved in an earlier session), fold its known aliases/
    # related terms in so expand_query benefits from that vocabulary.
    known_concepts = research_memory.find_known_concepts(query)
    if known_concepts:
        names = ", ".join(c["canonical_name"] for c in known_concepts)
        print(f"[research-memory] HIT: {names}")
    else:
        print("[research-memory] MISS: no known terminology matched")

    expansion = expand_query(query, known_concepts=known_concepts)
    # Guarantee the raw query itself is one of the searched formulations --
    # an LLM paraphrase can drift away from a real title's actual wording,
    # so this gives a near-exact title a direct shot via the title field.
    terms = list(dict.fromkeys([query, *expansion["terms"]]))[:10]

    return {
        **state,
        "is_direct_id": False,
        "search_terms": terms,
        "category": expansion["category"],
        "lookup_authors": expansion["authors"],
        "lookup_year": expansion["year"],
        "seed_candidates": near_matches,
    }
