"""Cohere wrapper: query expansion (chat), candidate reranking, embedding, summarization."""

import json

import cohere
from pydantic import ValidationError

from re_search_it.config import COHERE_API_KEY
from re_search_it.schemas import Briefing

CHAT_MODEL = "command-a-03-2025"
RERANK_MODEL = "rerank-v3.5"
EMBED_MODEL = "embed-english-v3.0"
EMBED_BATCH_SIZE = 96

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


def embed(texts: list[str], input_type: str = "search_document") -> list[list[float]]:
    """Embed a batch of texts.

    input_type="search_document" for chunks going into the vector store,
    "search_query" for a user question at retrieval time -- Cohere embeds
    these two asymmetrically, so passing the wrong one degrades relevance.
    """
    if not texts:
        return []

    vectors: list[list[float]] = []
    for i in range(0, len(texts), EMBED_BATCH_SIZE):
        batch = texts[i : i + EMBED_BATCH_SIZE]
        response = _client.embed(
            model=EMBED_MODEL,
            texts=batch,
            input_type=input_type,
            embedding_types=["float"],
        )
        vectors.extend(response.embeddings.float_)

    return vectors


_BRIEFING_SCHEMA_HINT = """Respond with ONLY a JSON object matching this exact shape:
{
  "tldr": "1-2 sentence plain-language summary",
  "problem": "what problem the paper addresses and why it matters",
  "approach": "the core method/technique",
  "key_findings": ["finding 1", "finding 2", "..."],
  "limitations": ["limitation 1 the paper itself acknowledges", "..."]
}
key_findings and limitations must each contain at least one item. If the
paper text doesn't explicitly state limitations, infer the most defensible
ones from its scope (e.g. dataset size, domain, evaluation setup) rather
than leaving the list empty."""

_SUMMARIZE_PROMPT = """Write a structured executive briefing for this paper.

Title: {title}
Authors: {authors}

Paper text (sections may be truncated):
{sections_text}

{schema_hint}"""

MAX_SUMMARIZE_RETRIES = 2


def summarize_paper(title: str, authors: list[str], sections_text: str) -> Briefing:
    """Generate a validated Briefing for a paper. Retries on schema validation failure."""
    prompt = _SUMMARIZE_PROMPT.format(
        title=title,
        authors=", ".join(authors),
        sections_text=sections_text,
        schema_hint=_BRIEFING_SCHEMA_HINT,
    )
    messages = [{"role": "user", "content": prompt}]

    last_error: Exception | None = None
    for _ in range(MAX_SUMMARIZE_RETRIES):
        response = _client.chat(
            model=CHAT_MODEL,
            messages=messages,
            response_format={"type": "json_object"},
        )
        text = response.message.content[0].text
        try:
            return Briefing.model_validate(json.loads(text))
        except (json.JSONDecodeError, ValidationError) as exc:
            last_error = exc
            messages.append({"role": "assistant", "content": text})
            messages.append(
                {
                    "role": "user",
                    "content": f"That JSON was invalid: {exc}. Return corrected JSON only.",
                }
            )

    raise ValueError(f"Failed to produce a valid briefing after retries: {last_error}")
