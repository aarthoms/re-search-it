"""Regression tests for first-stage arXiv candidate recall.

LLM-dependent classification (does the router correctly send "has anyone
benchmarked X" to discovery, or "what did this paper find" to qa) is verified
live against the real Cohere API rather than here -- a unit test mocking the
LLM's response only proves the surrounding code handles that response
correctly, not that the prompt actually produces it. What IS deterministic
and worth unit-testing is the candidate-generation machinery itself: no
verbatim-phrase quoting, category as an additive/soft signal never a hard
filter, and direct-ID lookups staying untouched by any of this.
"""

from unittest.mock import patch

from re_search_it.nodes import arxiv_retrieval as arxiv_retrieval_module
from re_search_it.nodes.arxiv_retrieval import _field_query, arxiv_retrieval


def _fake_paper(arxiv_id: str) -> dict:
    return {"arxiv_id": arxiv_id, "title": f"Paper {arxiv_id}", "summary": "..."}


def test_field_query_has_no_verbatim_phrase_quoting():
    query = _field_query("Mojo GPU benchmark")
    assert '"' not in query
    assert "abs:(Mojo AND GPU AND benchmark)" in query
    assert "ti:(Mojo AND GPU AND benchmark)" in query


def test_field_query_falls_back_to_quoting_when_no_words_found():
    # Degenerate input (punctuation only) -- shouldn't crash, falls back safely.
    query = _field_query("???")
    assert query == 'abs:"???" OR ti:"???"'


# Test A: "GNN adversarial attacks cybersecurity" -- candidate generation > 0,
# reranking (not this stage) determines the final selection.
def test_multiple_formulations_produce_nonzero_candidates():
    state = {
        "query": "GNN adversarial attacks cybersecurity",
        "is_direct_id": False,
        "search_terms": [
            "graph neural network adversarial",
            "GNN adversarial attack",
            "graph neural network security",
            "GNN intrusion detection",
        ],
        "category": None,
    }

    def fake_search(field_query, max_results):
        # Different formulations surface different (overlapping) papers.
        if "adversarial" in field_query:
            return [_fake_paper("1"), _fake_paper("2")]
        return [_fake_paper("2"), _fake_paper("3")]

    with patch.object(arxiv_retrieval_module, "search_by_topic", side_effect=fake_search):
        result = arxiv_retrieval(state)

    assert result.get("error") is None
    assert len(result["candidates"]) == 3  # deduped: 1, 2, 3
    ids = {c["arxiv_id"] for c in result["candidates"]}
    assert ids == {"1", "2", "3"}


# Test B: "Mojo GPU benchmark" -- candidate generation > 0.
def test_single_formulation_produces_candidates():
    state = {
        "query": "Mojo GPU benchmark",
        "is_direct_id": False,
        "search_terms": ["Mojo GPU benchmark"],
        "category": None,
    }
    with patch.object(arxiv_retrieval_module, "search_by_topic", return_value=[_fake_paper("42")]):
        result = arxiv_retrieval(state)

    assert len(result["candidates"]) == 1


def test_category_is_additive_not_a_hard_filter():
    """A wrong/narrow category-scoped search returning nothing must not
    erase candidates the unconstrained search already found."""
    state = {
        "query": "some query",
        "is_direct_id": False,
        "search_terms": ["some formulation"],
        "category": "cs.WRONG",
    }

    def fake_search(field_query, max_results):
        if "cat:cs.WRONG" in field_query:
            return []  # the guessed category finds nothing
        return [_fake_paper("1")]  # but the unconstrained search does

    with patch.object(arxiv_retrieval_module, "search_by_topic", side_effect=fake_search):
        result = arxiv_retrieval(state)

    assert result.get("error") is None
    assert len(result["candidates"]) == 1


def test_category_constrained_results_are_unioned_with_unconstrained():
    """Category-scoped and unconstrained searches both run and are merged --
    neither replaces the other."""
    state = {
        "query": "some query",
        "is_direct_id": False,
        "search_terms": ["some formulation"],
        "category": "cs.LG",
    }

    def fake_search(field_query, max_results):
        if "cat:cs.LG" in field_query:
            return [_fake_paper("only-in-category")]
        return [_fake_paper("only-unconstrained")]

    with patch.object(arxiv_retrieval_module, "search_by_topic", side_effect=fake_search):
        result = arxiv_retrieval(state)

    ids = {c["arxiv_id"] for c in result["candidates"]}
    assert ids == {"only-in-category", "only-unconstrained"}


def test_zero_candidates_across_all_formulations_sets_error():
    state = {
        "query": "totally obscure nonexistent thing",
        "is_direct_id": False,
        "search_terms": ["nonexistent thing", "obscure query"],
        "category": None,
    }
    with patch.object(arxiv_retrieval_module, "search_by_topic", return_value=[]):
        result = arxiv_retrieval(state)

    assert result.get("error") is not None
    assert result.get("candidates") is None or result.get("candidates") == []


# Test E: direct arXiv ID -- unaffected by any of the recall changes above.
def test_direct_id_lookup_unaffected_by_recall_changes():
    state = {"query": "1706.03762", "is_direct_id": True, "arxiv_id": "1706.03762"}

    with patch.object(arxiv_retrieval_module, "get_by_id", return_value=_fake_paper("1706.03762")) as mock_get, \
         patch.object(arxiv_retrieval_module, "search_by_topic") as mock_search:
        result = arxiv_retrieval(state)

    mock_get.assert_called_once_with("1706.03762")
    mock_search.assert_not_called()  # never touches the formulation/category machinery
    assert result["selected_paper"]["arxiv_id"] == "1706.03762"
