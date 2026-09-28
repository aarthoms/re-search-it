"""Node: generate a structured, validated executive briefing from parsed sections."""

from re_search_it.state import PaperState
from re_search_it.tools.cohere_client import summarize_paper

# Keep the prompt bounded -- full papers can run 30k+ words; a truncated but
# section-ordered excerpt (abstract/intro/method/results first) is plenty for
# a briefing and keeps cost/latency predictable.
MAX_SECTIONS_CHARS = 12000
# A per-section cap, not just an overall one -- without this, a long
# introduction alone could consume the entire budget and starve every
# section after it (the model would then write "findings"/"limitations" it
# never actually saw). Capping each section means the excerpt spans several
# sections even when one of them is long.
MAX_CHARS_PER_SECTION = 3000

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

# Bibliography text -- never useful to the summarizer (it's not paper
# content), and previously it was leaking in via the priority list's
# catch-all "append every remaining section" fallback, silently eating half
# the character budget on some papers.
_EXCLUDED_SECTIONS = {"references"}


def _sample_start_middle_end(text: str, budget: int) -> str:
    """When section-splitting failed to find real structure, everything
    landed in one giant blob (usually "preamble") -- naive truncation from
    the start only ever shows the abstract/intro. Limitations and
    conclusions typically sit near the end, just before the references, so
    sampling the beginning, middle, AND end gives the model a shot at all
    three instead of just the first ~6% of the paper.
    """
    if len(text) <= budget:
        return text

    third = budget // 3
    mid_point = len(text) // 2
    start = text[:third]
    middle = text[mid_point - third // 2 : mid_point + third // 2]
    end = text[-third:]
    return f"{start}\n\n[... middle of paper omitted ...]\n\n{middle}\n\n[... omitted ...]\n\n{end}"


def _build_sections_text(sections: dict[str, str], structure_degraded: bool = False) -> str:
    usable = {k: v for k, v in sections.items() if k not in _EXCLUDED_SECTIONS}

    if structure_degraded:
        # Section labels aren't trustworthy here (most/all text landed in
        # one bucket) -- sample across the whole body instead of trusting
        # section order to put the important parts first.
        combined = "\n\n".join(usable.values())
        return _sample_start_middle_end(combined, MAX_SECTIONS_CHARS)

    ordered_names = [s for s in _PRIORITY_SECTIONS if s in usable]
    ordered_names += [s for s in usable if s not in ordered_names]

    parts = []
    remaining = MAX_SECTIONS_CHARS
    for name in ordered_names:
        if remaining <= 0:
            break
        cap = min(MAX_CHARS_PER_SECTION, remaining)
        body = usable[name][:cap]
        parts.append(f"## {name}\n{body}")
        remaining -= len(body)

    return "\n\n".join(parts)


def summarize(state: PaperState) -> PaperState:
    paper = state["selected_paper"]
    sections_text = _build_sections_text(state["parsed_sections"], state.get("structure_degraded", False))

    try:
        briefing = summarize_paper(paper["title"], paper["authors"], sections_text)
    except Exception as exc:
        return {**state, "error": f"Failed to generate briefing for {paper['arxiv_id']}: {exc}"}

    return {**state, "briefing": briefing.model_dump()}
