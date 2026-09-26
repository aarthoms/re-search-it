"""Resolve a citation mention (e.g. "Falcon [Almazrouei et al., 2023]") to an
arXiv lookup target, without treating it as a fresh literature search.

Deterministic hierarchy, cheapest/most-precise first -- an explicit citation
should never fall straight into semantic discovery on the raw mention text
(that's what previously turned "Falcon [Almazrouei et al., 2023]" into a
search for "Falcon model, financial relationship analysis, evaluation
metrics"):
  1. arXiv ID literally in the mention -- handled upstream by find_arxiv_id.
  2. Match against the active paper's own extracted references list.
  3. Author+year arXiv search (arXiv's au: field), not a blind topic search.
  4. Give up (return None) and let the caller fall back to the normal LLM
     router, rather than guessing with a low-confidence fallback.
"""

import re

from re_search_it.tools.arxiv_client import search_by_topic

_CITATION_PATTERN = re.compile(r"\[[^\[\]]*?\b(?:19|20)\d{2}\b[^\[\]]*?\]")
_SURNAME_PATTERN = re.compile(r"([A-Z][a-zA-Z\-]+)")
_YEAR_PATTERN = re.compile(r"\b(19|20)\d{2}\b")


def looks_like_citation(message: str) -> bool:
    """Cheap heuristic: a bracketed year, e.g. "[Almazrouei et al., 2023]"."""
    return bool(_CITATION_PATTERN.search(message))


def _citation_bracket(mention: str) -> str:
    """The bracketed part itself, e.g. "[Almazrouei et al., 2023]" -- the
    author surname must come from HERE, not from the informal name that
    often precedes it ("Falcon [Almazrouei et al., 2023]" -- "Falcon" is not
    the citation author)."""
    match = _CITATION_PATTERN.search(mention)
    return match.group(0) if match else mention


def _match_local_reference(mention: str, references: list[dict]) -> dict | None:
    bracket = _citation_bracket(mention)
    year_match = _YEAR_PATTERN.search(bracket)
    if not year_match:
        return None
    year = int(year_match.group(0))
    surname_match = _SURNAME_PATTERN.search(bracket)

    same_year = [r for r in references if r.get("year") == year]
    if not same_year:
        return None

    if surname_match:
        surname = surname_match.group(1).lower()
        # Check the full raw_text, not just the extracted first_author field --
        # citation styles vary ("Surname, F." vs "F. Surname"), and the wanted
        # surname may not be the first token either way.
        by_surname = [r for r in same_year if surname in r["raw_text"].lower()]
        if by_surname:
            return by_surname[0]
        return None

    # No surname to disambiguate -- only safe to guess if there's exactly one
    # reference from that year.
    return same_year[0] if len(same_year) == 1 else None


def resolve_reference(mention: str, references: list[dict]) -> str | None:
    """Return an arXiv lookup target -- or None if nothing resolved with
    confidence, in which case the caller should fall back to the normal LLM
    router rather than assuming this is a discovery request.

    Priority: local bibliography match with an arXiv ID already in it (most
    precise) > au:+keyword arXiv search, verified against the actual author
    list > the local match's own raw text as a last resort.

    Two things had to be fixed to make the au:+keyword step reliable:
    arXiv's au: field alone is NOT a strict author filter -- it returns
    unrelated papers by general relevance, so results are re-checked against
    the actual author list before being trusted. And the informal name that
    usually precedes the bracket ("Falcon" in "Falcon [Almazrouei et al.,
    2023]") turns out to be exactly the keyword needed to disambiguate --
    au:Almazrouei alone doesn't surface Falcon-40B in its own top results,
    au:Almazrouei AND all:Falcon does.

    Raw-text search is deliberately LOWEST priority: a bibliography entry is
    mostly a list of co-author names, and dumping that whole blob into
    arXiv's general query dilutes ranking enough to land on an unrelated
    paper that merely sounds topically similar -- exactly the "wrong
    identity" failure this resolver exists to prevent.
    """
    local = _match_local_reference(mention, references)
    if local and local.get("arxiv_id"):
        return local["arxiv_id"]

    bracket = _citation_bracket(mention)
    surname_match = _SURNAME_PATTERN.search(bracket)
    year_match = _YEAR_PATTERN.search(bracket)
    if surname_match and year_match:
        informal_name = mention[: mention.find("[")].strip()
        query = f"au:{surname_match.group(1)}"
        if informal_name:
            query += f" AND all:{informal_name}"

        results = search_by_topic(query, max_results=8)
        surname_lower = surname_match.group(1).lower()
        for r in results:
            if r.get("published", "")[:4] != year_match.group(0):
                continue
            if any(surname_lower in a.lower() for a in r["authors"]):
                return r["arxiv_id"]
        # Nothing both author-verified and year-matched -- don't guess with
        # an unrelated top hit.

    if local:
        return local["raw_text"]

    return None
