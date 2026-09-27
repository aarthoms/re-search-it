"""Resolve a citation/reference mention (e.g. "Falcon [Almazrouei et al.,
2023]", "Sobal et al. (2025)", "Almazrouei 2023 Falcon", "paper by
Almazrouei in 2023") to an arXiv lookup target, without treating it as a
fresh literature search.

Deterministic hierarchy, cheapest/most-precise first -- an explicit
reference should never fall straight into semantic discovery on the raw
mention text (that's what previously turned "Falcon [Almazrouei et al.,
2023]" into a search for "Falcon model, financial relationship analysis,
evaluation metrics"):
  1. arXiv ID literally in the mention -- handled upstream by find_arxiv_id.
  2. Match against the active paper's own extracted references list.
  3. Author(+year)+keyword arXiv search (arXiv's au: field, verified against
     the real author list), not a blind topic search.
  4. Give up (return None) and let the caller fall back to the normal LLM
     router, rather than guessing with a low-confidence fallback.
"""

import re

from re_search_it.tools.arxiv_client import search_by_author

# Two ways a reference mention shows up:
#   - wrapped in brackets/parens with a year inside: "[Author, 2023]" / "(Author, 2023)"
#   - bare, no required punctuation: "Author et al. 2025", "Author 2023 Falcon",
#     "paper by Author in 2023" -- surname followed by a year within a short
#     word window (optionally with "et al." in between).
_CITATION_PATTERN = re.compile(
    r"[\[\(][^\[\]()]*?\b(?:19|20)\d{2}\b[^\[\]()]*?[\]\)]"
    r"|\b[A-Z][a-zA-Z\-]+\b(?:\s+et\s+al\.?)?(?:\s+\w+){0,3}?\s+(?:19|20)\d{2}\b"
)
_SURNAME_PATTERN = re.compile(r"([A-Z][a-zA-Z\-]+)")
_YEAR_PATTERN = re.compile(r"\b(19|20)\d{2}\b")


def looks_like_citation(message: str) -> bool:
    """Cheap heuristic: a bracketed/parenthesized year, or a bare "Surname
    [et al.] ... YEAR" mention. Deliberately permissive -- a false positive
    here just means resolve_reference is tried and returns None, falling
    through to the normal router; a false negative means a real reference
    gets treated as a fresh literature search, which is the actual bug this
    exists to prevent.
    """
    return bool(_CITATION_PATTERN.search(message))


def _citation_match(mention: str) -> re.Match | None:
    return _CITATION_PATTERN.search(mention)


def _citation_span(mention: str) -> str:
    """The matched citation text itself -- the author surname must come
    from HERE, not from an informal name that may precede it ("Falcon
    [Almazrouei et al., 2023]" -- "Falcon" is not the citation author)."""
    match = _citation_match(mention)
    return match.group(0) if match else mention


def _match_local_reference(mention: str, references: list[dict]) -> dict | None:
    span = _citation_span(mention)
    year_match = _YEAR_PATTERN.search(span)
    if not year_match:
        return None
    year = int(year_match.group(0))
    surname_match = _SURNAME_PATTERN.search(span)

    same_year = [r for r in references if r.get("year") == year]
    if not same_year:
        return None

    if surname_match:
        surname = surname_match.group(1).lower()
        # Check the full raw_text, not just the extracted first_author/authors
        # fields -- citation styles vary ("Surname, F." vs "F. Surname"), and
        # the wanted surname may not be the first token either way.
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
    precise) > author(+year)+keyword arXiv search, verified against the
    actual author list > the local match's own raw text as a last resort.

    Raw-text search is deliberately LOWEST priority: a bibliography entry is
    mostly a list of co-author names, and dumping that whole blob into
    arXiv's general query dilutes ranking enough to land on an unrelated
    paper that merely sounds topically similar -- exactly the "wrong
    identity" failure this resolver exists to prevent.
    """
    print("[reference] searching current bibliography")
    local = _match_local_reference(mention, references)
    if local and local.get("arxiv_id"):
        print(f"[reference] matched: {local.get('title') or local['raw_text'][:60]}")
        return local["arxiv_id"]

    match = _citation_match(mention)
    span = match.group(0) if match else mention
    surname_match = _SURNAME_PATTERN.search(span)
    year_match = _YEAR_PATTERN.search(span)
    if surname_match and year_match:
        # Text surrounding the citation match, if any -- e.g. "Falcon" in
        # "Falcon [Almazrouei et al., 2023]" (before) OR "Almazrouei 2023
        # Falcon" (after, for the bare no-punctuation style). Without this,
        # au:Surname alone can surface an unrelated paper by relevance --
        # verified live: au:Almazrouei alone did NOT return Falcon-40B in
        # its own results, but au:Almazrouei AND all:Falcon does.
        informal_name = None
        if match:
            informal_name = mention[: match.start()].strip() or mention[match.end():].strip() or None

        print("[lookup] author/year candidate search")
        verified = search_by_author(
            surname_match.group(1), year=int(year_match.group(0)), keyword=informal_name
        )
        print(f"[lookup] candidates: {len(verified)}")
        if verified:
            return verified[0]["arxiv_id"]
        # Nothing both author-verified and year-matched -- don't guess with
        # an unrelated top hit.

    if local:
        print(f"[reference] matched (no arXiv ID in entry): {local.get('title') or local['raw_text'][:60]}")
        return local["raw_text"]

    print("[reference] no match in current bibliography or arXiv author search")
    return None
