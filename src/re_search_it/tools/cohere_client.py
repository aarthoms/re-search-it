"""Cohere wrapper: query expansion (chat) and candidate reranking."""

import cohere

from re_search_it.config import COHERE_API_KEY

CHAT_MODEL = "command-a-03-2025"
RERANK_MODEL = "rerank-v3.5"

_client = cohere.ClientV2(api_key=COHERE_API_KEY)

_EXPAND_PROMPT = """You turn a user's research question into arXiv search terms.
arXiv search matches keywords in titles/abstracts, not natural language intent,
so rewrite the query into 2-3 short keyword phrases a paper's title or abstract
would actually contain.

Also identify the single arXiv category the query most clearly belongs to
(e.g. cs.CL, cs.LG, cs.CV, cs.AI, stat.ML, physics.optics), or NONE if it
doesn't clearly belong to one.

Output in EXACTLY this format, no extra commentary:
CATEGORY: <code or NONE>
TERMS:
- <phrase 1>
- <phrase 2>
- <phrase 3>

User query: {query}"""


def expand_query(query: str) -> dict:
    """Turn a natural-language query into arXiv-friendly search terms + category.

    Returns {"terms": list[str], "category": str | None}.
    """
    response = _client.chat(
        model=CHAT_MODEL,
        messages=[{"role": "user", "content": _EXPAND_PROMPT.format(query=query)}],
    )
    text = response.message.content[0].text.strip()

    category = None
    terms: list[str] = []
    for line in text.splitlines():
        line = line.strip()
        if line.upper().startswith("CATEGORY:"):
            value = line.split(":", 1)[1].strip()
            category = None if value.upper() == "NONE" else value
        elif line.startswith("-"):
            terms.append(line.lstrip("-* ").strip())

    return {"terms": terms or [query], "category": category}


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
