"""Regression tests for structure-degradation detection and deeper QA retrieval."""

from unittest.mock import patch

from re_search_it.nodes import retrieve_chunks as retrieve_chunks_module
from re_search_it.nodes.retrieve_chunks import retrieve_chunks
from re_search_it.nodes.summarize import MAX_SECTIONS_CHARS, _build_sections_text
from re_search_it.tools.pdf_parser import is_structure_degraded, split_sections


# Custom numbered headings not in the fixed list, e.g. "7 Mojo for Financial
# LLMs", weren't recognized at all -- the whole paper fell into "preamble".
def test_generic_numbered_heading_is_recognized():
    text = "7 Mojo for Financial LLMs\nSome content about Mojo.\n7.1 Benchmark Setup\nSetup details here."
    sections = split_sections(text)
    assert "mojo for financial llms" in sections
    assert "benchmark setup" in sections


def test_ordinary_sentence_starting_with_a_number_is_not_a_heading():
    text = "3 kernels were tested on the hardware.\nMore prose continues here for context."
    sections = split_sections(text)
    # A sentence (ends in a period) must never be mistaken for a heading.
    assert "kernels were tested on the hardware" not in sections


def test_structure_degraded_when_everything_is_preamble():
    sections = {"preamble": "the entire paper text" * 100, "references": "bib entries"}
    assert is_structure_degraded(sections) is True


def test_structure_degraded_false_with_real_sections():
    sections = {
        "abstract": "short abstract",
        "introduction": "intro text",
        "results": "results text",
        "limitations": "limitations text",
    }
    assert is_structure_degraded(sections) is False


def test_structure_degraded_when_preamble_dominates():
    sections = {"preamble": "x" * 9000, "conclusion": "y" * 1000}
    assert is_structure_degraded(sections) is True


# Bug: references leaked into the summarizer via the priority list's
# catch-all "append every remaining section" fallback.
def test_build_sections_text_always_excludes_references():
    sections = {"abstract": "real content", "references": "bibliography entries " * 500}
    text = _build_sections_text(sections, structure_degraded=False)
    assert "bibliography entries" not in text


def test_build_sections_text_samples_start_middle_end_when_degraded():
    body = "START_MARKER " + ("filler word " * 5000) + "MIDDLE_MARKER " + ("filler word " * 5000) + "END_MARKER"
    sections = {"preamble": body}
    text = _build_sections_text(sections, structure_degraded=True)
    assert "START_MARKER" in text
    assert "MIDDLE_MARKER" in text
    assert "END_MARKER" in text
    assert len(text) <= MAX_SECTIONS_CHARS + 200  # small allowance for the omission markers


# Deeper QA: decomposed questions get more chunks than direct ones.
def test_decomposed_plan_retrieves_more_chunks_than_direct():
    def _fake_query_chunks(collection_id, query_vec, top_k, where=None):
        return {
            "ids": [[f"id_{i}" for i in range(top_k)]],
            "documents": [[f"doc {i}" for i in range(top_k)]],
            "metadatas": [[{"section": "results", "chunk_index": i} for i in range(top_k)]],
        }

    def _fake_rerank(question, texts, top_n):
        return [{"index": i, "relevance_score": 0.5} for i in range(min(top_n, len(texts)))]

    state_direct = {
        "question": "q", "selected_paper": {"arxiv_id": "1234.56789"},
        "vector_collection_id": "c", "retrieval_plan": {"mode": "direct", "subqueries": ["q"]},
    }
    state_decomposed = {**state_direct, "retrieval_plan": {"mode": "decomposed", "subqueries": ["q1", "q2"]}}

    with patch.object(retrieve_chunks_module, "embed", return_value=[[0.1]]), \
         patch.object(retrieve_chunks_module, "query_chunks", side_effect=_fake_query_chunks), \
         patch.object(retrieve_chunks_module, "rerank", side_effect=_fake_rerank), \
         patch.object(retrieve_chunks_module, "get_chunks_by_ids", return_value={"ids": [], "documents": [], "metadatas": []}):
        result_direct = retrieve_chunks(state_direct)
        result_decomposed = retrieve_chunks(state_decomposed)

    assert len(result_direct["retrieved_chunks"]) <= 5
    assert len(result_decomposed["retrieved_chunks"]) <= 8
    assert len(result_decomposed["retrieved_chunks"]) > len(result_direct["retrieved_chunks"])


# Deeper QA: top chunks get expanded with their neighbors instead of staying
# isolated 200-word fragments.
def test_top_chunks_are_expanded_with_neighbors():
    def _fake_query_chunks(collection_id, query_vec, top_k, where=None):
        return {
            "ids": [["1234.56789_results_5"]],
            "documents": [["middle chunk text"]],
            "metadatas": [[{"section": "results", "chunk_index": 5}]],
        }

    def _fake_rerank(question, texts, top_n):
        return [{"index": 0, "relevance_score": 0.5}]

    def _fake_get_chunks_by_ids(collection_id, ids):
        return {
            "ids": ids,
            "documents": ["before chunk text", "after chunk text"],
            "metadatas": [{"chunk_index": 4}, {"chunk_index": 6}],
        }

    state = {
        "question": "q", "selected_paper": {"arxiv_id": "1234.56789"},
        "vector_collection_id": "c", "retrieval_plan": {"mode": "direct", "subqueries": ["q"]},
    }

    with patch.object(retrieve_chunks_module, "embed", return_value=[[0.1]]), \
         patch.object(retrieve_chunks_module, "query_chunks", side_effect=_fake_query_chunks), \
         patch.object(retrieve_chunks_module, "rerank", side_effect=_fake_rerank), \
         patch.object(retrieve_chunks_module, "get_chunks_by_ids", side_effect=_fake_get_chunks_by_ids):
        result = retrieve_chunks(state)

    text = result["retrieved_chunks"][0]["text"]
    assert "before chunk text" in text
    assert "middle chunk text" in text
    assert "after chunk text" in text
