"""PDF text extraction and section-aware splitting."""

import re

from pypdf import PdfReader

SECTION_HEADERS = [
    "abstract",
    "introduction",
    "related work",
    "related work and background",
    "background",
    "method",
    "methods",
    "methodology",
    "approach",
    "experiments",
    "experimental setup",
    "experiments and results",
    "results",
    "results and discussion",
    "evaluation",
    "discussion",
    "conclusion",
    "conclusions",
    "limitations",
    "acknowledgments",
    "acknowledgements",
    "references",
    "appendix",
]

# Numeric ("3.", "3.1") or Roman-numeral ("III.") numbering, optionally
# preceding a heading.
_NUMBERED_PREFIX = r"(?:\d+\.?\d*\.?\s+|[IVXLCDM]+\.\s+)?"

# Matches a standalone heading line (longer phrases first, so "results and
# discussion" wins over a bare "results" when both are present).
_ORDERED_HEADERS = sorted(SECTION_HEADERS, key=len, reverse=True)
_HEADER_PATTERN = re.compile(
    r"^\s*" + _NUMBERED_PREFIX + r"(" + "|".join(re.escape(h) for h in _ORDERED_HEADERS) + r")\s*[:.\-—]?\s*$",
    re.IGNORECASE,
)

# "Abstract—We propose..." / "Abstract. We propose..." -- heading and body
# share one line, which _HEADER_PATTERN (whole-line match) never catches.
# Only tried while still in the preamble and before any abstract has been
# found, so it can't accidentally reclassify a later, unrelated line.
_INLINE_ABSTRACT_PATTERN = re.compile(r"^\s*abstract\b\s*[:.\-—]+\s*(\S.*)$", re.IGNORECASE)

# Fallback for custom headings not in SECTION_HEADERS, e.g. "7 Mojo for
# Financial LLMs" or "7.1 Benchmark Setup" -- any short numbered line
# starting with a capital letter and NOT ending in a full stop (an ordinary
# sentence that happens to start with a number, e.g. "3 kernels were
# tested...", almost always ends with one; a heading almost never does).
_GENERIC_NUMBERED_HEADING = re.compile(r"^\s*\d+(?:\.\d+)*\.?\s+([A-Z][A-Za-z0-9 ,\-:&']{0,79})$")


def _match_heading(line: str) -> str | None:
    match = _HEADER_PATTERN.match(line)
    if match:
        return match.group(1).strip().lower()

    stripped = line.strip()
    if stripped and not stripped.endswith("."):
        generic = _GENERIC_NUMBERED_HEADING.match(line)
        if generic:
            return generic.group(1).strip().lower()
    return None


def extract_text(pdf_path: str) -> str:
    """Extract raw text from every page of a PDF."""
    reader = PdfReader(pdf_path)
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def split_sections(text: str) -> dict[str, str]:
    """Split extracted text into sections keyed by lowercase heading name.

    Text before the first recognized heading is kept under "preamble".
    Headings that never appear in the text simply don't appear in the
    result. A heading that appears MORE THAN ONCE (e.g. "Results" in the
    main text and again in an appendix) has its occurrences appended
    together rather than the later one silently overwriting the earlier --
    losing a whole section's text to a same-named heading elsewhere in the
    paper was a real failure mode this guards against.
    """
    sections: dict[str, list[str]] = {}
    current = "preamble"
    buffer: list[str] = []

    def flush() -> None:
        if buffer:
            body = "\n".join(buffer).strip()
            if body:
                sections.setdefault(current, []).append(body)

    for line in text.splitlines():
        heading = _match_heading(line)
        if heading:
            flush()
            current = heading
            buffer = []
            continue

        if current == "preamble" and "abstract" not in sections:
            inline = _INLINE_ABSTRACT_PATTERN.match(line)
            if inline:
                flush()
                current = "abstract"
                buffer = [inline.group(1)]
                continue

        buffer.append(line)

    flush()

    return {name: "\n\n".join(parts).strip() for name, parts in sections.items() if parts}


def is_structure_degraded(sections: dict[str, str]) -> bool:
    """True if section-splitting effectively failed even though it didn't
    raise -- e.g. everything landed in "preamble" (and maybe "references")
    because the paper uses heading styles _match_heading still can't catch.
    Downstream code should treat this like a parse failure (visible
    warning, different summarization sampling strategy) even though
    parse_pdf() returned successfully.
    """
    real_sections = {k: v for k, v in sections.items() if k not in ("preamble", "references")}
    if not real_sections:
        return True

    total = sum(len(v) for v in sections.values())
    preamble_len = len(sections.get("preamble", ""))
    return total > 0 and (preamble_len / total) > 0.6


def parse_pdf(pdf_path: str) -> dict[str, str]:
    """Extract and section-split a PDF's text. Raises ValueError if no text found."""
    text = extract_text(pdf_path)
    if not text.strip():
        raise ValueError(f"No extractable text in PDF: {pdf_path}")
    return split_sections(text)
