"""Node: generate a structured, validated executive briefing from parsed sections."""

from re_search_it.state import PaperState
from re_search_it.tools.cohere_client import summarize_paper

# Keep the prompt bounded -- full papers can run 30k+ words; a truncated but
# section-ordered excerpt (abstract/intro/method/results first) is plenty for
# a briefing and keeps cost/latency predictable.
MAX_SECTIONS_CHARS = 12000

_PRIORITY_SECTIONS = [
    "abstract",
    "introduction",
    "method",
    "methods",
    "methodology",
    "approach",
    "results",
    "evaluation",
    "discussion",
    "conclusion",
    "conclusions",
    "limitations",
]


def _build_sections_text(sections: dict[str, str]) -> str:
    ordered_names = [s for s in _PRIORITY_SECTIONS if s in sections]
    ordered_names += [s for s in sections if s not in ordered_names]

    parts = []
    remaining = MAX_SECTIONS_CHARS
    for name in ordered_names:
        if remaining <= 0:
            break
        body = sections[name][:remaining]
        parts.append(f"## {name}\n{body}")
        remaining -= len(body)

    return "\n\n".join(parts)


def summarize(state: PaperState) -> PaperState:
    paper = state["selected_paper"]
    sections_text = _build_sections_text(state["parsed_sections"])

    try:
        briefing = summarize_paper(paper["title"], paper["authors"], sections_text)
    except Exception as exc:
        return {**state, "error": f"Failed to generate briefing for {paper['arxiv_id']}: {exc}"}

    return {**state, "briefing": briefing.model_dump()}
