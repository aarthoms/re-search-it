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
_YEAR_PATTERN = re.compile(r"\b(19|20)\d{2}\b")
_ARXIV_ID_PATTERN = re.compile(r"\d{4}\.\d{4,5}(?:v\d+)?")
_AUTHOR_PATTERN = re.compile(r"^([A-Z][a-zA-Z.\-]+)")


def parse_references(text: str) -> list[dict]:
    """Split a references-section blob into entries with {raw_text, year,
    arxiv_id, first_author} -- any field can be None if not found."""
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
        references.append(
            {
                "raw_text": entry[:500],
                "year": int(year_match.group(0)) if year_match else None,
                "arxiv_id": arxiv_match.group(0) if arxiv_match else None,
                "first_author": author_match.group(1).rstrip(",") if author_match else None,
            }
        )
    return references
