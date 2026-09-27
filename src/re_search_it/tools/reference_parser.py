"""Extract structured references from a paper's raw 'references' section text.

Heuristic, not a full bibliography parser -- good enough to give the citation
resolver something to match against (year + first author + an arXiv ID when
the reference happens to include one), not to perfectly parse every citation
style.

Handles two common numbering styles: numeric ("[1]", "1.") and author-year
("[Almazrouei et al., 2023]"). The author-year marker is kept as part of the
entry (needed later to extract year/author) and entries are split via a
lookahead so a reference that wraps across multiple PDF-extracted lines stays
one entry -- splitting naively per-line truncates the entry before it
reaches the title or arXiv ID, which is what silently broke citation
resolution against author-year-style bibliographies.
"""

import re

_NUMERIC_MARKER = re.compile(r"(?:^|\n)\s*\[\d+\]\s*|(?:^|\n)\s*\d{1,3}\.\s+")
_AUTHOR_YEAR_MARKER = re.compile(r"(?:^|\n)\s*\[[A-Z][^\[\]\n]{0,80}?(?:19|20)\d{2}\]")
_LEADING_MARKER = re.compile(r"^\s*(?:\[[^\]]*\]|\d{1,3}\.)\s*")
_YEAR_PATTERN = re.compile(r"\b(19|20)\d{2}\b")
_ARXIV_ID_PATTERN = re.compile(r"\d{4}\.\d{4,5}(?:v\d+)?")
_AUTHOR_PATTERN = re.compile(r"^([A-Z][a-zA-Z.\-]+)")
_VENUE_MARKER = re.compile(
    r"\b(arXiv|Proceedings|In\s|Conference|Journal|pp\.|vol\.|IEEE|ACM|NeurIPS|ICML|ICLR|CVPR)\b",
    re.IGNORECASE,
)
_NAME_TOKEN = re.compile(r"^[A-Z][a-zA-Z.\-]*$")


def _extract_authors(body: str) -> list[str]:
    """Best-effort: the author list is usually the segment before the first
    sentence break. Citation styles vary wildly, so this tolerates failure --
    an empty list is expected and fine for many entries."""
    first_segment = re.split(r"\.\s+", body, maxsplit=1)[0]
    parts = re.split(r",\s*|\s+and\s+|\s*&\s*", first_segment)
    names = [p.strip() for p in parts if p.strip() and _NAME_TOKEN.match(p.strip())]
    return names[:10]


def _extract_title(body: str) -> str | None:
    """Best-effort: the title is usually the segment right after the author
    list and before a venue marker or the trailing year. Genuinely
    heuristic -- returns None rather than guessing badly."""
    segments = [s.strip() for s in re.split(r"\.\s+", body) if len(s.strip()) > 3]
    if len(segments) < 2:
        return None
    candidates = [s for s in segments[1:] if not _VENUE_MARKER.search(s)]
    return candidates[0] if candidates else segments[1]


def parse_references(text: str) -> list[dict]:
    """Split a references-section blob into entries with {raw_text, title,
    authors, year, arxiv_id, first_author} -- any field can be None/empty if
    not found. `title`/`authors` are best-effort heuristics (citation styles
    vary too much to parse reliably); callers must tolerate them being
    missing or wrong, and prefer `arxiv_id`/`year`/`first_author` when
    precision matters.
    """
    if not text or not text.strip():
        return []

    if _AUTHOR_YEAR_MARKER.search(text):
        parts = re.split(rf"(?={_AUTHOR_YEAR_MARKER.pattern})", text)
        entries = [p.strip() for p in parts if p.strip()]
    else:
        entries = [e.strip() for e in _NUMERIC_MARKER.split(text) if e.strip()]
        if len(entries) <= 1:
            # No [N]/N. markers found either -- fall back to one entry per
            # non-trivial line (loses multi-line entries, but nothing better
            # to key off of).
            entries = [line.strip() for line in text.splitlines() if len(line.strip()) > 20]

    references = []
    for entry in entries:
        year_match = _YEAR_PATTERN.search(entry)
        arxiv_match = _ARXIV_ID_PATTERN.search(entry)
        author_match = _AUTHOR_PATTERN.match(entry)
        body = _LEADING_MARKER.sub("", entry).strip()
        references.append(
            {
                "raw_text": entry[:500],
                "title": _extract_title(body),
                "authors": _extract_authors(body),
                "year": int(year_match.group(0)) if year_match else None,
                "arxiv_id": arxiv_match.group(0) if arxiv_match else None,
                "first_author": author_match.group(1).rstrip(",") if author_match else None,
            }
        )
    return references
