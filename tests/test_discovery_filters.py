"""Regression tests for author/date-range filtering in discovery."""

from unittest.mock import patch

from re_search_it.tools import arxiv_client
from re_search_it.tools.arxiv_client import _date_range_clause, search_with_filters


def _fake_paper(arxiv_id: str) -> dict:
    return {"arxiv_id": arxiv_id, "title": f"Paper {arxiv_id}", "summary": "...", "authors": [], "published": "..."}


def test_date_range_clause_both_bounds():
    clause = _date_range_clause("2026-01-01", "2026-04-03")
    assert clause == "submittedDate:[20260101000000 TO 20260403235959]"


def test_date_range_clause_open_lower_bound():
    clause = _date_range_clause(None, "2004-12-31")
    assert clause.startswith("submittedDate:[19910101000000 TO 20041231235959")


def test_date_range_clause_open_upper_bound_uses_today():
    clause = _date_range_clause("2021-01-01", None)
    assert clause.startswith("submittedDate:[20210101000000 TO ")
    assert clause.endswith("235959]")


def test_date_range_clause_none_when_no_bounds():
    assert _date_range_clause(None, None) is None


def test_search_with_filters_combines_topic_authors_and_date():
    with patch.object(arxiv_client, "search_by_topic", return_value=[_fake_paper("1")]) as mock_search:
        search_with_filters("XYZ", authors=["Almazrouei", "Touvron"], date_from="2026-01-01", date_to="2026-04-03")

    called_query = mock_search.call_args[0][0]
    assert "(XYZ)" in called_query
    assert "au:Almazrouei OR au:Touvron" in called_query
    assert "submittedDate:[20260101000000 TO 20260403235959]" in called_query
    assert " AND " in called_query


def test_search_with_filters_no_filters_is_plain_topic_search():
    with patch.object(arxiv_client, "search_by_topic", return_value=[]) as mock_search:
        search_with_filters("XYZ")

    called_query = mock_search.call_args[0][0]
    assert called_query == "(XYZ)"


def test_search_with_filters_single_author_no_or():
    with patch.object(arxiv_client, "search_by_topic", return_value=[]) as mock_search:
        search_with_filters("XYZ", authors=["Almazrouei"])

    called_query = mock_search.call_args[0][0]
    assert "au:Almazrouei" in called_query
    assert "OR" not in called_query
