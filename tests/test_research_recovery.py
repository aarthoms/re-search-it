"""Regression tests for the research_recovery node and its graph routing."""

from unittest.mock import patch

from re_search_it import graph as graph_module
from re_search_it.nodes import research_recovery as recovery_module


def _fake_paper(arxiv_id: str) -> dict:
    return {"arxiv_id": arxiv_id, "title": f"Paper {arxiv_id}", "summary": "..."}


# 4. Zero-result query -> recovery generates broader formulations and retries arXiv.
def test_recovery_expands_candidate_pool_on_success():
    state = {"query": "GNN adversarial attacks cybersecurity", "candidates": []}

    with patch.object(recovery_module, "generate_recovery_queries", return_value=["GNN adversarial", "GNN security"]), \
         patch.object(recovery_module, "search_by_topic", return_value=[_fake_paper("1"), _fake_paper("2")]), \
         patch.object(recovery_module, "discover_and_persist", return_value=None):
        result = recovery_module.research_recovery(state)

    assert result["recovery_attempted"] is True
    assert result.get("error") is None
    assert len(result["candidates"]) > 0


# 5. Recovery still fails -> existing answerability/relevance behavior remains intact.
def test_recovery_sets_error_when_nothing_found():
    state = {"query": "totally obscure nonexistent thing", "candidates": []}

    with patch.object(recovery_module, "generate_recovery_queries", return_value=["nonexistent thing"]), \
         patch.object(recovery_module, "search_by_topic", return_value=[]), \
         patch.object(recovery_module, "discover_and_persist", return_value=None):
        result = recovery_module.research_recovery(state)

    assert result["recovery_attempted"] is True
    assert result["error"] is not None

    # And the graph's own routing correctly treats this as final failure,
    # not another recovery loop (recovery_attempted is already True).
    route = graph_module._route_after_recovery(result)
    assert route == "failed"


def test_route_after_ranking_recovers_once_then_gives_up():
    unanswerable_first_attempt = {"answerable": False, "is_direct_id": False, "recovery_attempted": False}
    assert graph_module._route_after_ranking(unanswerable_first_attempt) == "recover"

    unanswerable_second_attempt = {"answerable": False, "is_direct_id": False, "recovery_attempted": True}
    assert graph_module._route_after_ranking(unanswerable_second_attempt) == "unanswerable"


# 6. Direct arXiv ID -> discovery/recovery must not interfere with direct lookup.
def test_direct_id_failure_never_routes_to_recovery():
    direct_id_not_found = {"error": "No arXiv paper found for ID: 9999.99999", "is_direct_id": True}
    assert graph_module._route_after_retrieval(direct_id_not_found) == "failed"


def test_topic_search_zero_candidates_routes_to_recovery_once():
    first_failure = {"error": "No arXiv candidates found for query: x", "is_direct_id": False}
    assert graph_module._route_after_retrieval(first_failure) == "recover"

    already_recovered = {**first_failure, "recovery_attempted": True}
    assert graph_module._route_after_retrieval(already_recovered) == "failed"


# 7. Loaded-paper follow-up must not trigger discovery/recovery -- the QA
# graph is a structurally separate pipeline that never imports this module.
def test_qa_graph_has_no_dependency_on_research_discovery():
    from re_search_it import qa_graph

    import inspect

    qa_source = inspect.getsource(qa_graph)
    assert "research_discovery" not in qa_source
    assert "research_recovery" not in qa_source
