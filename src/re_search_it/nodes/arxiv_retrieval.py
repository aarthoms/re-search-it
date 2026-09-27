"""Node: fetch candidate metadata from arXiv — direct ID lookup or topic search.

First-stage retrieval optimizes for RECALL, not precision -- that's
selection_ranking.py's job. Two things previously killed recall here:

1. Verbatim-phrase quoting (`abs:"Mojo GPU benchmark"`) requires that exact
   wording to appear adjacent, in that order, in the abstract/title. A real
   paper's abstract phrases things differently even when it's exactly the
   right paper. Boolean AND-of-words within the field (`abs:(Mojo AND GPU
   AND benchmark)`) finds the same paper regardless of phrasing.
2. Category as a hard AND filter: a wrong LLM-guessed category (or a paper
   that legitimately spans multiple categories) could zero out an otherwise
   good candidate pool. Category is now always additive -- the unconstrained
   search always runs, and a category-scoped search runs alongside it
   (not instead of it) when a category was inferred, then both are unioned.
"""

import re

from re_search_it.state import PaperState
from re_search_it.tools.arxiv_client import get_by_id, search_by_author, search_by_topic

MAX_RESULTS_PER_TERM = 20
_WORD_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9+/#.\-]*")


def _field_query(term: str) -> str:
    """Boolean AND-of-words within abs:/ti:, not a quoted verbatim phrase."""
    words = _WORD_PATTERN.findall(term)
    if not words:
        return f'abs:"{term}" OR ti:"{term}"'
    clause = " AND ".join(words)
    return f"abs:({clause}) OR ti:({clause})"


def arxiv_retrieval(state: PaperState) -> PaperState:
    if state.get("is_direct_id"):
        paper = get_by_id(state["arxiv_id"])
        if paper is None:
            return {**state, "error": f"No arXiv paper found for ID: {state['arxiv_id']}"}
        return {**state, "selected_paper": paper, "candidates": [paper]}

    if state.get("candidates"):
        # query_understanding's exact-title fast path already resolved this
        # -- nothing to search for, let selection_ranking score/confirm it.
        return state

    category = state.get("category")
    formulations = state.get("search_terms") or [state["query"]]

    print(f"[query] original: {state['query']!r}")
    print(f"[query] formulations: {formulations}")

    seen_ids: set[str] = set()
    candidates: list[dict] = []

    # Seeded by query_understanding's near-title search, if any -- merged in
    # (not a replacement for the formulation search below), so a near-exact
    # title paper is IN the candidate pool even if none of the formulations
    # happen to surface it.
    for paper in state.get("seed_candidates") or []:
        if paper["arxiv_id"] not in seen_ids:
            seen_ids.add(paper["arxiv_id"])
            candidates.append(paper)

    # Additive author-aware search: arXiv's au: field is the correct,
    # precise way to search by author -- the keyword/abs:/ti: formulations
    # above never use it, so a query naming an author ("Almazrouei 2023
    # Falcon") gets a real structured lookup here rather than relying on the
    # author's surname coincidentally appearing in an abstract.
    authors = state.get("lookup_authors") or []
    year = state.get("lookup_year")
    if authors:
        print("[lookup] author/year candidate search")
        keyword = state["query"] if len(state["query"].split()) <= 6 else None
        author_candidates = search_by_author(authors[0], year=year, keyword=keyword)
        print(f"[lookup] candidates: {len(author_candidates)}")
        for paper in author_candidates:
            if paper["arxiv_id"] not in seen_ids:
                seen_ids.add(paper["arxiv_id"])
                candidates.append(paper)

    raw_counts: dict[str, int] = {}

    for term in formulations:
        raw_counts[term] = 0

        unconstrained = search_by_topic(_field_query(term), max_results=MAX_RESULTS_PER_TERM)
        raw_counts[term] += len(unconstrained)
        for paper in unconstrained:
            if paper["arxiv_id"] not in seen_ids:
                seen_ids.add(paper["arxiv_id"])
                candidates.append(paper)

        # Additive, never instead-of: a wrong or overly-narrow category
        # guess can only add candidates here, never remove ones the
        # unconstrained search already found.
        if category:
            scoped = search_by_topic(f"({_field_query(term)}) AND cat:{category}", max_results=MAX_RESULTS_PER_TERM)
            raw_counts[term] += len(scoped)
            for paper in scoped:
                if paper["arxiv_id"] not in seen_ids:
                    seen_ids.add(paper["arxiv_id"])
                    candidates.append(paper)

    print(f"[arxiv] raw candidates per formulation: {raw_counts}")
    print(f"[arxiv] unique candidates: {len(candidates)}")

    if not candidates:
        return {**state, "error": f"No arXiv candidates found for query: {state['query']}"}

    return {**state, "candidates": candidates}
