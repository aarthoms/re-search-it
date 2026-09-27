"""Regression tests for paper-level bibliographic lookup and reference resolution."""

from unittest.mock import patch

from re_search_it.nodes import arxiv_retrieval as arxiv_retrieval_module
from re_search_it.nodes.arxiv_retrieval import arxiv_retrieval
from re_search_it.tools import reference_resolver as resolver_module
from re_search_it.tools.reference_parser import parse_references
from re_search_it.tools.reference_resolver import looks_like_citation, resolve_reference


def _fake_paper(arxiv_id: str, authors=None, published="2023-01-01T00:00:00") -> dict:
    return {
        "arxiv_id": arxiv_id,
        "title": f"Paper {arxiv_id}",
        "summary": "...",
        "authors": authors or [],
        "published": published,
    }


# Test A: direct arXiv ID -- existing behavior unchanged, no lookup machinery touched.
def test_direct_id_untouched_by_bibliographic_machinery():
    state = {"query": "1706.03762", "is_direct_id": True, "arxiv_id": "1706.03762"}

    with patch.object(arxiv_retrieval_module, "get_by_id", return_value=_fake_paper("1706.03762")) as mock_get, \
         patch.object(arxiv_retrieval_module, "search_by_author") as mock_author, \
         patch.object(arxiv_retrieval_module, "search_by_topic") as mock_topic:
        result = arxiv_retrieval(state)

    mock_get.assert_called_once_with("1706.03762")
    mock_author.assert_not_called()
    mock_topic.assert_not_called()
    assert result["selected_paper"]["arxiv_id"] == "1706.03762"


# Test B: exact title -- seed_candidates from a near-title search are merged
# into (not replacing) the normal formulation search, so a real title has a
# fair shot without an unrelated lookalike winning purely on prior candidates.
def test_seed_candidates_from_near_title_are_merged_not_replaced():
    state = {
        "query": "Attention Is All You Need",
        "is_direct_id": False,
        "search_terms": ["Attention Is All You Need"],
        "category": None,
        "seed_candidates": [_fake_paper("1706.03762")],
    }
    with patch.object(arxiv_retrieval_module, "search_by_topic", return_value=[_fake_paper("9999.99999")]):
        result = arxiv_retrieval(state)

    ids = {c["arxiv_id"] for c in result["candidates"]}
    assert ids == {"1706.03762", "9999.99999"}  # both present; reranking (elsewhere) decides


# Test C: reference in the loaded paper's bibliography -- resolved locally,
# bibliography checked BEFORE any arXiv author search.
def test_reference_resolves_from_local_bibliography_first():
    references = parse_references(
        "[Almazrouei et al., 2023] E. Almazrouei, H. Alobeidli. "
        "The Falcon Series of Open Language Models. arXiv:2311.16867, 2023."
    )
    with patch.object(resolver_module, "search_by_author") as mock_author_search:
        target = resolve_reference("Falcon [Almazrouei et al., 2023]", references)

    assert target == "2311.16867"
    mock_author_search.assert_not_called()  # never escalated to arXiv -- local match had an ID


# Test D: author/year mention (no brackets) -- produces candidates via a
# structured au: field search, both through the resolver and through
# arxiv_retrieval's standalone (no-paper-loaded) additive path.
def test_bare_author_year_mention_is_detected_as_a_citation():
    assert looks_like_citation("Almazrouei 2023 Falcon") is True
    assert looks_like_citation("What paper is Almazrouei et al. 2023?") is True
    assert looks_like_citation("paper by Almazrouei in 2023") is True


def test_author_year_resolver_produces_candidates_via_structured_search():
    with patch.object(
        resolver_module, "search_by_author",
        return_value=[_fake_paper("2311.16867", authors=["Ebtesam Almazrouei"])],
    ) as mock_search:
        target = resolve_reference("Almazrouei 2023 Falcon", references=[])

    assert target == "2311.16867"
    mock_search.assert_called_once()
    _, kwargs = mock_search.call_args
    assert kwargs.get("year") == 2023 or mock_search.call_args[0][1] == 2023


def test_arxiv_retrieval_author_lookup_is_additive():
    state = {
        "query": "Almazrouei 2023 Falcon",
        "is_direct_id": False,
        "search_terms": ["Falcon language model"],
        "category": None,
        "lookup_authors": ["Almazrouei"],
        "lookup_year": 2023,
    }
    with patch.object(arxiv_retrieval_module, "search_by_author", return_value=[_fake_paper("2311.16867")]), \
         patch.object(arxiv_retrieval_module, "search_by_topic", return_value=[_fake_paper("other-id")]):
        result = arxiv_retrieval(state)

    ids = {c["arxiv_id"] for c in result["candidates"]}
    assert ids == {"2311.16867", "other-id"}  # both present, not a replacement


# Test E: an unknown reference (no local match, no arXiv author match) falls
# back to None -- the caller then falls through to the normal LLM router
# rather than failing outright.
def test_unknown_reference_falls_back_to_none_not_failure():
    with patch.object(resolver_module, "search_by_author", return_value=[]):
        target = resolve_reference("Nobody [Fakename et al., 1999]", references=[])

    assert target is None


# Test F: paper QA follow-up -- no bibliographic/author-lookup machinery
# involved at all (structural: QA graph never imports it).
def test_qa_graph_has_no_dependency_on_bibliographic_lookup():
    import inspect

    from re_search_it import qa_graph

    qa_source = inspect.getsource(qa_graph)
    assert "reference_resolver" not in qa_source
    assert "search_by_author" not in qa_source


# Test G: literature discovery phrasing is NOT mistaken for a reference --
# no bracket, no year at all, so looks_like_citation must be False.
def test_literature_discovery_phrasing_is_not_a_citation():
    assert looks_like_citation("Has anyone else benchmarked Mojo against CUDA?") is False
    assert looks_like_citation("papers about GNN adversarial attacks") is False


# reference_parser: normalized shape includes title/authors, tolerates missing fields.
def test_parsed_reference_includes_title_and_authors_when_extractable():
    refs = parse_references(
        "[Almazrouei et al., 2023] E. Almazrouei, H. Alobeidli. "
        "The Falcon Series of Open Language Models. arXiv:2311.16867, 2023."
    )
    assert len(refs) == 1
    ref = refs[0]
    assert ref["arxiv_id"] == "2311.16867"
    assert ref["year"] == 2023
    assert "authors" in ref and "title" in ref  # present even when extraction is imperfect


def test_parsed_reference_tolerates_missing_fields():
    refs = parse_references("Some malformed reference text with no clear structure at all here")
    assert len(refs) == 1
    ref = refs[0]
    assert ref["arxiv_id"] is None
    assert ref["year"] is None
