"""Regression tests for discovery's high-recall formulation + bounded fallback."""

from unittest.mock import patch

from re_search_it import discovery as discovery_module
from re_search_it.discovery import discover_papers
from re_search_it.schemas import DiscoveryFilters
from re_search_it.tools import cohere_client


def _fake_paper(arxiv_id: str) -> dict:
    return {"arxiv_id": arxiv_id, "title": f"Paper {arxiv_id}", "summary": "...", "authors": [], "published": "2025-01-01"}


def test_generate_discovery_queries_strips_quotes_and_numbering():
    fake_response = type(
        "R", (), {"message": type("M", (), {"content": [type("C", (), {"text": (
            '1. "KV cache compression"\n'
            '2. key-value cache compression\n'
            '- KV cache memory reduction\n'
        )})()]})()},
    )()
    with patch.object(cohere_client._client, "chat", return_value=fake_response):
        phrases = cohere_client.generate_discovery_queries("KV cache compression for LLMs")

    assert phrases == ["KV cache compression", "key-value cache compression", "KV cache memory reduction"]
    assert all('"' not in p for p in phrases)


def test_discover_papers_falls_back_to_core_topic_on_zero_candidates():
    filters = DiscoveryFilters(topic="KV cache compression", authors=[], date_from=None, date_to=None)

    with patch.object(discovery_module, "extract_discovery_filters", return_value=filters), \
         patch.object(discovery_module, "generate_discovery_queries", return_value=["a", "b", "c"]), \
         patch.object(
             discovery_module, "search_with_filters",
             side_effect=lambda query, **kw: [_fake_paper("42")] if query == "KV cache compression" else [],
         ) as mock_search, \
         patch.object(discovery_module, "rerank", return_value=[{"index": 0, "relevance_score": 0.5}]), \
         patch.object(discovery_module, "describe_relations", return_value=["relevant"]):
        result = discover_papers("recent work on KV cache compression for LLMs")

    assert result["unique_count"] == 1
    assert len(result["results"]) == 1
    # The fallback call happened with the plain core topic, not a formulation.
    fallback_calls = [c for c in mock_search.call_args_list if c[0][0] == "KV cache compression"]
    assert len(fallback_calls) == 1


def test_discover_papers_fallback_retains_author_and_date_filters():
    filters = DiscoveryFilters(topic="neural networks", authors=["Rosenblatt"], date_from=None, date_to="1997-12-31")

    with patch.object(discovery_module, "extract_discovery_filters", return_value=filters), \
         patch.object(discovery_module, "generate_discovery_queries", return_value=["formulation one"]), \
         patch.object(discovery_module, "search_with_filters", return_value=[]) as mock_search:
        discover_papers("papers by Rosenblatt on neural networks before 1998")

    for call in mock_search.call_args_list:
        assert call.kwargs.get("authors") == ["Rosenblatt"]
        assert call.kwargs.get("date_to") == "1997-12-31"


def test_discover_papers_no_redundant_fallback_when_topic_already_searched():
    filters = DiscoveryFilters(topic="KV cache compression", authors=[], date_from=None, date_to=None)

    with patch.object(discovery_module, "extract_discovery_filters", return_value=filters), \
         patch.object(discovery_module, "generate_discovery_queries", return_value=["KV cache compression"]), \
         patch.object(discovery_module, "search_with_filters", return_value=[]) as mock_search:
        result = discover_papers("KV cache compression")

    # Only one call -- the formulation list already contained the exact
    # core topic, so no separate fallback attempt is needed.
    assert mock_search.call_count == 1
    assert result["unique_count"] == 0
