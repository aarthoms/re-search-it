"""PDF text extraction and section-aware splitting."""

import re

from pypdf import PdfReader

SECTION_HEADERS = [
    "abstract",
    "introduction",
    "related work",
    "background",
    "method",
    "methods",
    "methodology",
    "approach",
    "experiments",
    "experimental setup",
    "results",
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

# Matches a standalone heading line, optionally numbered (e.g. "3. Related Work").
_HEADER_PATTERN = re.compile(
    r"^\s*(?:\d+\.?\d*\.?\s+)?(" + "|".join(re.escape(h) for h in SECTION_HEADERS) + r")\s*$",
    re.IGNORECASE,
)


def extract_text(pdf_path: str) -> str:
    """Extract raw text from every page of a PDF."""
    reader = PdfReader(pdf_path)
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def split_sections(text: str) -> dict[str, str]:
    """Split extracted text into sections keyed by lowercase heading name.

    Text before the first recognized heading is kept under "preamble".
    Headings that never appear in the text simply don't appear in the result.
    """
    sections: dict[str, str] = {}
    current = "preamble"
    buffer: list[str] = []

    for line in text.splitlines():
        match = _HEADER_PATTERN.match(line)
        if match:
            if buffer:
                sections[current] = "\n".join(buffer).strip()
            current = match.group(1).strip().lower()
            buffer = []
        else:
            buffer.append(line)

    if buffer:
        sections[current] = "\n".join(buffer).strip()

    return {name: body for name, body in sections.items() if body}


def parse_pdf(pdf_path: str) -> dict[str, str]:
    """Extract and section-split a PDF's text. Raises ValueError if no text found."""
    text = extract_text(pdf_path)
    if not text.strip():
        raise ValueError(f"No extractable text in PDF: {pdf_path}")
    return split_sections(text)
