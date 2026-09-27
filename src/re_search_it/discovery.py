"""Multi-paper discovery: broad search, dedupe, rerank, relevance filter.

Deliberately stops before fetch/parse/chunk/embed -- discovery should be
cheap to run broadly across many candidates; a paper only gets the expensive
full pipeline (in graph.py) once the user picks one via lookup. This is the
opposite of the single-paper retrieval_graph, which commits to one paper
immediately.

Supports explicit author/date-range filters ("papers by Almazrouei on LLMs
published in the past year", "papers on XYZ before 2005") -- these are
extracted from the request and applied as hard arXiv-side filters (unlike
the topic search itself, which stays loose/OR-ish for recall).
"""

from re_search_it.tools.arxiv_client import search_with_filters
from re_search_it.tools.cohere_client import describe_relations, extract_discovery_filters, generate_discovery_queries, rerank

RESULTS_PER_QUERY = 15
MAX_RESULTS = 8
RELAXED_MAX_RESULTS = 15
# Looser than selection_ranking's single-paper floor (0.3): discovery favors
# recall, and a paper worth listing needn't be the single best match.
DISCOVERY_RELEVANCE_FLOOR = 0.1
# Used for search_refinement ("even weak matches", "show me more") -- the
# user explicitly asked to relax the filter, so let nearly everything through.
RELAXED_RELEVANCE_FLOOR = 0.02


def discover_papers(topic: str, relaxed: bool = False) -> dict:
    """Broad multi-query search for a topic, with optional author/date
    filters extracted from the request itself. Returns diagnostics + a
    ranked, relevance-filtered list of paper dicts (each with
    relevance_score and a one-line `relation` to the topic) -- never more
    than one paper's worth of LLM/API cost beyond filter extraction, query
    expansion, and one batched relation call, regardless of how many
    candidates were found.

    `relaxed=True` is for search_refinement turns ("even weak matches") --
    lowers the relevance floor and raises the result cap instead of treating
    the refinement phrase itself as a brand new topic to embed.
    """
    filters = extract_discovery_filters(topic)
    search_topic = filters.topic or topic

    if filters.authors or filters.date_from or filters.date_to:
        print(
            f"[discovery] filters -- authors: {filters.authors or 'none'}, "
            f"date range: {filters.date_from or '-inf'} to {filters.date_to or 'now'}"
        )

    queries = generate_discovery_queries(search_topic)

    raw_count = 0
    seen_ids: set[str] = set()
    candidates: list[dict] = []
    for query in queries:
        found = search_with_filters(
            query,
            authors=filters.authors,
            date_from=filters.date_from,
            date_to=filters.date_to,
            max_results=RESULTS_PER_QUERY,
        )
        raw_count += len(found)
        for paper in found:
            if paper["arxiv_id"] not in seen_ids:
                seen_ids.add(paper["arxiv_id"])
                candidates.append(paper)

    if not candidates:
        return {
            "queries": queries,
            "raw_count": 0,
            "unique_count": 0,
            "results": [],
            "filters": filters.model_dump(),
        }

    max_results = RELAXED_MAX_RESULTS if relaxed else MAX_RESULTS
    floor = RELAXED_RELEVANCE_FLOOR if relaxed else DISCOVERY_RELEVANCE_FLOOR

    documents = [f"{c['title']}\n{c['summary']}" for c in candidates]
    ranked = rerank(topic, documents, top_n=max_results)

    results = [
        {**candidates[r["index"]], "relevance_score": r["relevance_score"]}
        for r in ranked
        if r["relevance_score"] >= floor
    ]

    if results:
        relations = describe_relations(search_topic, results)
        for paper, relation in zip(results, relations):
            paper["relation"] = relation

    return {
        "queries": queries,
        "raw_count": raw_count,
        "unique_count": len(candidates),
        "results": results,
        "filters": filters.model_dump(),
    }
