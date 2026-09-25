"""Cohere wrapper: query expansion (chat) and candidate reranking."""

import cohere

from re_search_it.config import COHERE_API_KEY

CHAT_MODEL = "command-a-03-2025"
RERANK_MODEL = "rerank-v3.5"

_client = cohere.ClientV2(api_key=COHERE_API_KEY)

_EXPAND_PROMPT = """You turn a user's research question into arXiv search terms.
arXiv search matches keywords in titles/abstracts, not natural language intent,
so rewrite the query into 2-3 short keyword phrases a paper's title or abstract
would actually contain. Output ONLY the phrases, one per line, no numbering,
no extra commentary.

User query: {query}"""


def expand_query(query: str) -> list[str]:
    """Turn a natural-language query into arXiv-friendly keyword search phrases."""
    response = _client.chat(
        model=CHAT_MODEL,
        messages=[{"role": "user", "content": _EXPAND_PROMPT.format(query=query)}],
    )
    text = response.message.content[0].text.strip()
    phrases = [line.strip("-* ").strip() for line in text.splitlines() if line.strip()]
    return phrases or [query]


def rerank(query: str, documents: list[str], top_n: int | None = None) -> list[dict]:
    """Rerank documents (e.g. paper abstracts) against a query.

    Returns a list of {"index": int, "relevance_score": float}, sorted by
    relevance_score descending, indices referring to positions in `documents`.
    """
    if not documents:
        return []
    response = _client.rerank(
        model=RERANK_MODEL,
        query=query,
        documents=documents,
        top_n=top_n or len(documents),
    )
    return [{"index": r.index, "relevance_score": r.relevance_score} for r in response.results]
