"""Multi-paper discovery: broad search, dedupe, rerank, relevance filter.

Deliberately stops before fetch/parse/chunk/embed -- discovery should be
cheap to run broadly across many candidates; a paper only gets the expensive
full pipeline (in graph.py) once the user picks one via lookup. This is the
opposite of the single-paper retrieval_graph, which commits to one paper
immediately.
"""

from re_search_it.tools.arxiv_client import search_by_topic
from re_search_it.tools.cohere_client import describe_relations, generate_discovery_queries, rerank

RESULTS_PER_QUERY = 15
MAX_RESULTS = 8
# Looser than selection_ranking's single-paper floor (0.3): discovery favors
# recall, and a paper worth listing needn't be the single best match.
DISCOVERY_RELEVANCE_FLOOR = 0.1


def discover_papers(topic: str) -> dict:
    """Broad multi-query search for a topic. Returns diagnostics + a ranked,
    relevance-filtered list of paper dicts (each with relevance_score and a
    one-line `relation` to the topic) -- never more than one paper's worth of
    LLM/API cost beyond the initial query expansion and one batched relation
    call, regardless of how many candidates were found.
    """
    queries = generate_discovery_queries(topic)

    raw_count = 0
    seen_ids: set[str] = set()
    candidates: list[dict] = []
    for query in queries:
        found = search_by_topic(query, max_results=RESULTS_PER_QUERY)
        raw_count += len(found)
        for paper in found:
            if paper["arxiv_id"] not in seen_ids:
                seen_ids.add(paper["arxiv_id"])
                candidates.append(paper)

    if not candidates:
        return {"queries": queries, "raw_count": 0, "unique_count": 0, "results": []}

    documents = [f"{c['title']}\n{c['summary']}" for c in candidates]
    ranked = rerank(topic, documents, top_n=MAX_RESULTS)

    results = [
        {**candidates[r["index"]], "relevance_score": r["relevance_score"]}
        for r in ranked
        if r["relevance_score"] >= DISCOVERY_RELEVANCE_FLOOR
    ]

    if results:
        relations = describe_relations(topic, results)
        for paper, relation in zip(results, relations):
            paper["relation"] = relation

    return {
        "queries": queries,
        "raw_count": raw_count,
        "unique_count": len(candidates),
        "results": results,
    }
