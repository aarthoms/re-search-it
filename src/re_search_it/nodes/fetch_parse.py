"""Node: download the selected paper's PDF, extract sections, prep its vector collection.

Two distinct failure modes, handled differently:
  - PDF download fails (network, 404, etc.) -> hard failure, state["error"] set,
    nothing downstream can proceed without a document to chunk.
  - PDF downloads but text extraction/section-splitting fails (scanned PDF, odd
    layout) -> graceful degradation: fall back to a single "abstract" section
    built from arXiv metadata, so the pipeline can still produce a thin briefing
    and answer abstract-level questions instead of dying outright.
"""

import re

from re_search_it.state import PaperState
from re_search_it.tools.arxiv_client import download_pdf
from re_search_it.tools.pdf_parser import parse_pdf
from re_search_it.tools.reference_parser import parse_references
from re_search_it.tools.vector_store import get_or_create_collection

_SAFE_CHARS = re.compile(r"[^a-zA-Z0-9_-]")


def _collection_id(arxiv_id: str) -> str:
    """arXiv IDs contain '.' which ChromaDB collection names disallow."""
    return "paper_" + _SAFE_CHARS.sub("_", arxiv_id)


def fetch_parse(state: PaperState) -> PaperState:
    paper = state["selected_paper"]
    arxiv_id = paper["arxiv_id"]

    try:
        pdf_path = download_pdf(arxiv_id)
    except Exception as exc:
        return {**state, "error": f"Failed to download PDF for {arxiv_id}: {exc}"}

    try:
        sections = parse_pdf(pdf_path)
        parse_degraded = False
    except Exception:
        # Falls back to an abstract-only briefing instead of dying outright
        # -- but this must be VISIBLE, not silent, since a briefing built
        # from only the abstract looks identical to a full-paper one
        # otherwise.
        sections = {"abstract": paper.get("summary", "")}
        parse_degraded = True

    references = parse_references(sections.get("references", ""))

    collection_id = _collection_id(arxiv_id)
    get_or_create_collection(
        collection_id,
        metadata={
            "arxiv_id": arxiv_id,
            "title": paper["title"],
            "authors": ", ".join(paper["authors"]),
        },
    )

    return {
        **state,
        "pdf_path": pdf_path,
        "parsed_sections": sections,
        "parse_degraded": parse_degraded,
        "references": references,
        "vector_collection_id": collection_id,
    }
