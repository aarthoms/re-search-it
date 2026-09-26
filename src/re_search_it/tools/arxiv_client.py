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
