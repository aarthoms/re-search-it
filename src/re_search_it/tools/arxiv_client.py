"""arXiv tool: topic search, direct ID lookup, and PDF download."""

import re
from pathlib import Path

import arxiv
import requests

_ID_PATTERN = re.compile(r"^\d{4}\.\d{4,5}(v\d+)?$")
_ID_IN_TEXT_PATTERN = re.compile(r"\d{4}\.\d{4,5}(?:v\d+)?")

_client = arxiv.Client()


def is_arxiv_id(text: str) -> bool:
    """Detect whether a query string looks like a direct arXiv ID (e.g. 2301.12345)."""
    return bool(_ID_PATTERN.match(text.strip()))


def find_arxiv_id(text: str) -> str | None:
    """Find an arXiv ID anywhere inside a longer message, e.g. a chat message
    like "what does 2301.12345 say?" -- unlike is_arxiv_id, the ID need not be
    the whole string."""
    match = _ID_IN_TEXT_PATTERN.search(text)
    return match.group(0) if match else None


def _result_to_dict(result: arxiv.Result) -> dict:
    return {
        "arxiv_id": result.get_short_id(),
        "title": result.title,
        "authors": [a.name for a in result.authors],
        "summary": result.summary,
        "published": result.published.isoformat() if result.published else None,
        "primary_category": result.primary_category,
        "pdf_url": result.pdf_url,
        "entry_id": result.entry_id,
    }


def search_by_topic(query: str, max_results: int = 5) -> list[dict]:
    """Search arXiv by topic/keywords, return candidate metadata dicts."""
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance,
    )
    return [_result_to_dict(r) for r in _client.results(search)]


def _normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", text.lower())


def search_by_exact_title(query: str) -> dict | None:
    """If `query` already IS (or is very close to) a paper's actual title,
    resolve it directly via one title-field-scoped search -- no LLM
    expansion call, no 6-query semantic discovery, no rerank needed. This is
    what "Attention is all you need" should hit instead of burning a full
    discovery pass and then landing on a lookalike paper by relevance.
    """
    results = search_by_topic(f'ti:"{query}"', max_results=3)
    normalized_query = _normalize(query)
    for r in results:
        normalized_title = _normalize(r["title"])
        if normalized_query == normalized_title:
            return r
    return None


def search_by_near_title(query: str, max_results: int = 5) -> list[dict]:
    """When `query` isn't an EXACT title match, still give title-similar
    results a fair shot as reranker candidates instead of falling straight
    through to unrelated semantic discovery. This is what prevents "Attention
    Is All You Need" from resolving to "Is Attention All What You Need?"
    merely because that lookalike scored well on a generic semantic search --
    a boolean AND-of-title-words search surfaces the real paper as a
    candidate, and the existing reranker (against the ORIGINAL query) picks
    between them.
    """
    words = re.findall(r"[A-Za-z0-9][A-Za-z0-9+/#.\-]*", query)
    if not words:
        return []
    clause = " AND ".join(words)
    return search_by_topic(f"ti:({clause})", max_results=max_results)


def search_by_author(surname: str, year: int | None = None, keyword: str | None = None, max_results: int = 8) -> list[dict]:
    """arXiv's au: field alone is NOT a strict author filter -- it returns
    unrelated papers by general relevance, so results are re-checked against
    the actual author list before being trusted. `keyword` (e.g. an
    informal model/paper name) sharply improves precision when present --
    au:Surname alone can fail to surface the right paper in its own top
    results even when au:Surname AND all:Keyword puts it first.

    Returns only results where `surname` genuinely appears in the author
    list (and, if `year` is given, whose published year matches) -- never a
    low-confidence top hit that merely happened to rank first.
    """
    query = f"au:{surname}"
    if keyword:
        query += f" AND all:{keyword}"

    results = search_by_topic(query, max_results=max_results)
    surname_lower = surname.lower()
    verified = [r for r in results if any(surname_lower in a.lower() for a in r["authors"])]
    if year:
        year_str = str(year)
        verified = [r for r in verified if r.get("published", "")[:4] == year_str]
    return verified


def get_by_id(arxiv_id: str) -> dict | None:
    """Fetch a single paper's metadata by direct arXiv ID. Returns None if not found."""
    search = arxiv.Search(id_list=[arxiv_id])
    results = list(_client.results(search))
    if not results:
        return None
    return _result_to_dict(results[0])


def download_pdf(arxiv_id: str, dest_dir: str = "./data/papers") -> str:
    """Download a paper's PDF by arXiv ID, return the local file path.

    arXiv PDFs are immutable per version (a new revision gets a new v-suffix,
    e.g. v1 -> v2), so if the file's already on disk from a previous run,
    reuse it instead of re-fetching over HTTP.
    """
    Path(dest_dir).mkdir(parents=True, exist_ok=True)

    search = arxiv.Search(id_list=[arxiv_id])
    results = list(_client.results(search))
    if not results:
        raise ValueError(f"No arXiv paper found for ID: {arxiv_id}")

    result = results[0]
    dest_path = Path(dest_dir) / f"{result.get_short_id()}.pdf"

    if dest_path.exists() and dest_path.stat().st_size > 0:
        return str(dest_path)

    response = requests.get(result.pdf_url, timeout=30)
    response.raise_for_status()
    dest_path.write_bytes(response.content)

    return str(dest_path)
