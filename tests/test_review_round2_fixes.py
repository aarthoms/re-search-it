"""Regression tests for two bugs confirmed by live code review of commit 32758d3."""

from unittest.mock import patch

from re_search_it.nodes import retrieve_chunks as retrieve_chunks_module
from re_search_it.nodes.retrieve_chunks import _expand_with_neighbors, retrieve_chunks
from re_search_it.tools.pdf_parser import _match_heading


# Bug 1: the generic numbered-heading fallback over-matched non-headings --
# affiliation lines, hardware specs, year-prefixed venue lines, table rows.
def test_year_prefixed_line_is_not_a_heading():
    assert _match_heading("2024 IEEE Conference") is None


def test_hardware_spec_with_digit_word_is_not_a_heading():
    assert _match_heading("2 Apple M2 Max") is None


def test_version_number_line_is_not_a_heading():
    assert _match_heading("10 Python 3") is None


def test_legitimate_custom_headings_still_recognized():
    assert _match_heading("7 Mojo for Financial LLMs") == "mojo for financial llms"
    assert _match_heading("7.1 Benchmark Setup") == "benchmark setup"


# Bug 2: a chunk expanded with neighbors in round 1 got expanded AGAIN if
# re-selected in round 2 (after a refine_query loop), duplicating the
# neighbor text in what the model reads.
def test_already_expanded_chunk_is_not_expanded_twice():
    chunk = {"id": "p_s_5", "section": "s", "chunk_index": 5, "text": "before X middle X after", "expanded": True}
    with patch.object(retrieve_chunks_module, "get_chunks_by_ids") as mock_get:
        result = _expand_with_neighbors(chunk, "p", "collection")

    mock_get.assert_not_called()
    assert result is chunk


def test_second_retrieval_round_does_not_duplicate_neighbor_text():
    """End-to-end: simulate round 1 (produces an expanded chunk), then feed
    that as round 2's seed (state["retrieved_chunks"]) the way refine_query
    -> retrieve_chunks actually does, and confirm the neighbor text appears
    only once even if the same chunk is re-selected."""

    def _fake_query_chunks(collection_id, query_vec, top_k, where=None):
        return {
            "ids": [["1234.56789_results_5"]],
            "documents": [["middle chunk text"]],
            "metadatas": [[{"section": "results", "chunk_index": 5}]],
        }

    def _fake_rerank(question, texts, top_n):
        return [{"index": 0, "relevance_score": 0.1}]  # deliberately low -> triggers another round

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
        round1 = retrieve_chunks(state)
        # refine_query -> retrieve_chunks again, seeded with round 1's output
        # (mirrors qa_graph.py's actual loop).
        round2 = retrieve_chunks({**state, "retrieved_chunks": round1["retrieved_chunks"]})

    text = round2["retrieved_chunks"][0]["text"]
    assert text.count("before chunk text") == 1
    assert text.count("middle chunk text") == 1
    assert text.count("after chunk text") == 1
