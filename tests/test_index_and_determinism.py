"""Regression tests for contextualized chunk embedding, index versioning,
history-free answer calls, and deterministic (temperature=0) chat calls."""

from unittest.mock import patch

from re_search_it.nodes import chunk_embed as chunk_embed_module
from re_search_it.nodes import fetch_parse as fetch_parse_module
from re_search_it.nodes import retrieve_chunks as retrieve_chunks_module
from re_search_it.nodes.chunk_embed import chunk_embed
from re_search_it.nodes.fetch_parse import _collection_id
from re_search_it.nodes.retrieve_chunks import retrieve_chunks
from re_search_it.tools import cohere_client
from re_search_it.tools.chunker import INDEX_VERSION
from re_search_it.tools.cohere_client import answer_question


class _FakeText:
    def __init__(self, text):
        self.text = text


class _FakeMessage:
    def __init__(self, text):
        self.content = [_FakeText(text)]


class _FakeResponse:
    def __init__(self, text):
        self.message = _FakeMessage(text)


# a. chunk_embed embeds prefixed text but stores raw text as documents.
def test_chunk_embed_prefixes_for_embedding_but_stores_raw_documents():
    state = {
        "selected_paper": {"arxiv_id": "1234.56789"},
        "vector_collection_id": "paper_1234_56789_v2",
        "parsed_sections": {"results": "one two three four five"},
    }

    with patch.object(chunk_embed_module, "collection_chunk_count", return_value=0), \
         patch.object(chunk_embed_module, "embed", return_value=[[0.1]]) as mock_embed, \
         patch.object(chunk_embed_module, "add_chunks") as mock_add:
        chunk_embed(state)

    embedded_texts = mock_embed.call_args.args[0]
    assert embedded_texts == ["[results] one two three four five"]

    stored_documents = mock_add.call_args.kwargs["documents"]
    assert stored_documents == ["one two three four five"]


# b. retrieve_chunks passes prefixed text to rerank, returned chunks carry raw text.
def test_retrieve_chunks_reranks_prefixed_text_but_returns_raw_text():
    def _fake_query_chunks(collection_id, query_vec, top_k, where=None):
        return {
            "ids": [["1234.56789_results_1"]],
            "documents": [["raw chunk text"]],
            "metadatas": [[{"section": "results", "chunk_index": 1}]],
        }

    def _fake_rerank(question, documents, top_n):
        assert documents == ["[results] raw chunk text"]
        return [{"index": 0, "relevance_score": 0.9}]

    state = {
        "question": "q", "selected_paper": {"arxiv_id": "1234.56789"},
        "vector_collection_id": "c", "retrieval_plan": {"mode": "direct", "subqueries": ["q"]},
    }

    with patch.object(retrieve_chunks_module, "embed", return_value=[[0.1]]), \
         patch.object(retrieve_chunks_module, "query_chunks", side_effect=_fake_query_chunks), \
         patch.object(retrieve_chunks_module, "rerank", side_effect=_fake_rerank), \
         patch.object(retrieve_chunks_module, "get_chunks_by_ids", return_value={"ids": [], "documents": [], "metadatas": []}):
        result = retrieve_chunks(state)

    assert result["retrieved_chunks"][0]["text"] == "raw chunk text"


# c. collection ID includes INDEX_VERSION and stays a valid Chroma name for old-style IDs too.
def test_collection_id_includes_index_version():
    assert _collection_id("2401.12345v1") == f"paper_2401_12345v1_v{INDEX_VERSION}"


def test_collection_id_valid_chroma_name_for_old_style_id():
    collection_id = _collection_id("hep-th/9711200")
    assert collection_id == f"paper_hep-th_9711200_v{INDEX_VERSION}"
    import re

    assert re.fullmatch(r"[a-zA-Z0-9_-]+", collection_id)


# d. answer_question sends exactly one message, regardless of conversation history.
def test_answer_question_sends_single_message_with_no_history():
    evidence = [{"section": "results", "text": "some evidence"}]

    with patch.object(cohere_client, "_client") as mock_client:
        mock_client.chat.return_value = _FakeResponse("the answer")
        answer_question("What did they find?", evidence)

    messages = mock_client.chat.call_args.kwargs["messages"]
    assert len(messages) == 1
    assert messages[0]["role"] == "user"


# e. every chat call in cohere_client passes temperature=0.0.
def test_all_chat_calls_use_temperature_zero():
    calls = [
        (cohere_client.expand_query, ("query",), '{"entities":[],"topics":[],"tasks":[],"comparators":[],"domains":[],"category":null,"authors":[],"year":null,"search_formulations":["x"]}'),
        (cohere_client.summarize_paper, ("title", ["author"], "text"), '{"tldr":"t","problem":"p","significance":"s","approach":["a"],"key_findings":["f"],"limitations":["l"],"follow_up_questions":["q"]}'),
        (cohere_client.plan_retrieval, ("question", ["abstract"]), '{"mode":"direct","subqueries":["q"],"section_hints":[]}'),
        (cohere_client.refine_query, ("question", ["evidence"]), "refined query"),
        (cohere_client.answer_question, ("question", [{"section": "s", "text": "t"}]), "an answer"),
        (cohere_client.route_top_level, ("message", [], None, None), '{"mode":"qa","topic":null,"paper_id":null,"selection":null,"standalone_query":"message"}'),
        (cohere_client.extract_discovery_filters, ("query",), '{"topic":"t","authors":[],"date_from":null,"date_to":null}'),
        (cohere_client.generate_discovery_queries, ("topic",), "phrase one\nphrase two"),
        (cohere_client.describe_relations, ("topic", [{"title": "t", "summary": "s"}]), '{"relations":["relevant"]}'),
        (cohere_client.discover_terminology, ("term", "query"), '{"canonical_name":"c","aliases":[],"related_terms":[],"domains":[],"confidence":0.9}'),
        (cohere_client.generate_recovery_queries, ("query",), "phrase one\nphrase two"),
    ]

    for fn, args, stub_text in calls:
        with patch.object(cohere_client, "_client") as mock_client:
            mock_client.chat.return_value = _FakeResponse(stub_text)
            fn(*args)
            assert mock_client.chat.call_args.kwargs["temperature"] == cohere_client.CHAT_TEMPERATURE, fn.__name__
