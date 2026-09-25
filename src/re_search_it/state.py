"""PaperState: shared typed state passed through the LangGraph pipeline."""

from typing import TypedDict


class PaperState(TypedDict, total=False):
    query: str
    is_direct_id: bool
    arxiv_id: str
    search_terms: list[str]
    category: str | None
    candidates: list[dict]
    selected_paper: dict
    low_confidence: bool
    pdf_path: str
    parsed_sections: dict[str, str]
    vector_collection_id: str
    chunk_count: int
    briefing: dict
    conversation_history: list[dict]
    error: str

    # QA loop
    question: str
    retrieval_plan: dict
    retrieved_chunks: list[dict]
    retrieval_rounds: int
    evidence_sufficient: bool
    answer: dict
