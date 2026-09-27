"""arXiv tool: topic search, direct ID lookup, and PDF download."""

import re
from datetime import datetime, timezone
from pathlib import Path

import arxiv
import requests

_NEW_ID = r"\d{4}\.\d{4,5}(?:v\d+)?"
# Old-style IDs (pre-2007): <archive>[.<category>]/<7 digits>[v<n>], e.g.
# "cs/0112017" or "math.CO/0409461" -- a fixed archive-name whitelist avoids
# false-positiving on arbitrary "word/7digits" text elsewhere in a message.
_OLD_ARCHIVES = (
    r"astro-ph|cond-mat|gr-qc|hep-ex|hep-lat|hep-ph|hep-th|math-ph|nlin|"
    r"nucl-ex|nucl-th|physics|quant-ph|math|cs|q-bio|q-fin|stat|eess|econ|"
    r"alg-geom|adap-org|chao-dyn|cmp-lg|comp-gas|dg-ga|funct-an|mtrl-th|"
    r"patt-sol|plasm-ph|solv-int|supr-con|acc-phys|ao-sci|atom-ph|bayes-an|"
    r"chem-ph|q-alg"
)
_OLD_ID = rf"(?:{_OLD_ARCHIVES})(?:\.[A-Z]{{2}})?/\d{{7}}(?:v\d+)?"

_ID_PATTERN = re.compile(rf"^(?:{_NEW_ID}|{_OLD_ID})$", re.IGNORECASE)
_ID_IN_TEXT_PATTERN = re.compile(rf"(?:{_NEW_ID}|{_OLD_ID})", re.IGNORECASE)

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


def _date_range_clause(date_from: str | None, date_to: str | None) -> str | None:
    """arXiv's submittedDate field query, e.g.
    submittedDate:[20200101000000 TO 20201231235959]. `date_from`/`date_to`
    are ISO YYYY-MM-DD strings; either side may be open (unbounded)."""
    if not date_from and not date_to:
        return None
    start = (date_from or "19910101").replace("-", "") + "000000"
    end = (date_to or datetime.now(timezone.utc).strftime("%Y%m%d")).replace("-", "") + "235959"
    return f"submittedDate:[{start} TO {end}]"


def search_with_filters(
    query: str,
    authors: list[str] | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
    max_results: int = 15,
) -> list[dict]:
    """A topic search AND-combined with explicit author/date constraints.

    Unlike the soft/additive category signal elsewhere in this project,
    author names and date ranges here are things the USER explicitly stated
    ("papers by Almazrouei before 2020"), not an LLM guess -- so they're
    applied as genuine hard filters. Multiple authors are OR'd together
    (papers by any of them), each individually AND'd with the topic query
    and the date range.
    """
    clauses = [f"({query})"] if query else []
    if authors:
        author_clause = " OR ".join(f"au:{a}" for a in authors[:5])
        clauses.append(f"({author_clause})")
    date_clause = _date_range_clause(date_from, date_to)
    if date_clause:
        clauses.append(date_clause)

    full_query = " AND ".join(clauses) if clauses else query
    return search_by_topic(full_query, max_results=max_results)


def get_by_id(arxiv_id: str) -> dict | None:
    """Fetch a single paper's metadata by direct arXiv ID. Returns None if not found."""
    search = arxiv.Search(id_list=[arxiv_id])
    results = list(_client.results(search))
    if not results:
        return None
    return _result_to_dict(results[0])


def download_pdf(arxiv_id: str, dest_dir: str = "./data/papers") -> str:
    """Download a paper's PDF by arXiv ID, return the local file path.

    Builds the PDF URL directly from the ID (https://arxiv.org/pdf/<id>)
    instead of doing a second arxiv.Search just to read back the URL --
    callers always pass an already-resolved, version-qualified ID (from
    get_by_id/search_by_topic's own get_short_id()), so nothing is lost by
    skipping that redundant, rate-limited round trip. A 404 (bad ID) surfaces
    via raise_for_status() instead of an empty-search check.

    arXiv PDFs are immutable per version (a new revision gets a new v-suffix,
    e.g. v1 -> v2), so if the file's already on disk from a previous run,
    reuse it instead of re-fetching over HTTP.
    """
    Path(dest_dir).mkdir(parents=True, exist_ok=True)
    dest_path = Path(dest_dir) / f"{arxiv_id}.pdf"

    if dest_path.exists() and dest_path.stat().st_size > 0:
        return str(dest_path)

    response = requests.get(f"https://arxiv.org/pdf/{arxiv_id}", timeout=30)
    response.raise_for_status()
    dest_path.write_bytes(response.content)

    return str(dest_path)
