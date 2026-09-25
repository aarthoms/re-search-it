"""PaperState: shared typed state passed through the LangGraph pipeline."""

from typing import TypedDict


class PaperState(TypedDict, total=False):
    query: str
    is_direct_id: bool
    arxiv_id: str
    candidates: list[dict]
    selected_paper: dict
    parsed_sections: dict[str, str]
    vector_collection_id: str
    chunk_count: int
    briefing: dict
    conversation_history: list[dict]
    error: str
