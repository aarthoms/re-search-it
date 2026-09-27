"""Export a paper's briefing (metadata + generated content) to disk.

The Briefing schema itself doesn't carry paper metadata (arxiv_id/authors/
date/link) -- that already lives on the paper object, and the LLM has no
reliable way to know its own paper's ID/date anyway. This combines both at
export time instead of duplicating metadata into the LLM-generated schema.
"""

import json
import re
from pathlib import Path

_SAFE_CHARS = re.compile(r"[^a-zA-Z0-9_-]")
EXPORT_DIR = "./data/briefings"


def _combined(paper: dict, briefing: dict) -> dict:
    return {
        "arxiv_id": paper["arxiv_id"],
        "title": paper["title"],
        "authors": paper["authors"],
        "published": paper.get("published"),
        "pdf_url": paper["pdf_url"],
        **briefing,
    }


def _to_markdown(b: dict) -> str:
    lines = [
        f"# {b['title']}",
        "",
        f"**arXiv ID:** {b['arxiv_id']}  ",
        f"**Authors:** {', '.join(b['authors'])}  ",
    ]
    if b.get("published"):
        lines.append(f"**Published:** {b['published'][:10]}  ")
    lines.append(f"**Link:** {b['pdf_url']}")
    lines.append("")
    lines.append(f"## TL;DR\n{b['tldr']}")
    lines.append(f"\n## Problem\n{b['problem']}")
    lines.append(f"\n## Why it matters\n{b['significance']}")
    lines.append("\n## Approach")
    lines += [f"- {a}" for a in b["approach"]]
    lines.append("\n## Key findings")
    lines += [f"- {f}" for f in b["key_findings"]]
    lines.append("\n## Limitations")
    lines += [f"- {lim}" for lim in b["limitations"]]
    lines.append("\n## Follow-up questions")
    lines += [f"- {q}" for q in b["follow_up_questions"]]
    return "\n".join(lines) + "\n"


def save_briefing(paper: dict, briefing: dict) -> str:
    """Write the combined paper metadata + briefing to disk as both JSON and
    Markdown (same stem, same directory). Returns the JSON path."""
    Path(EXPORT_DIR).mkdir(parents=True, exist_ok=True)
    stem = _SAFE_CHARS.sub("_", paper["arxiv_id"])
    combined = _combined(paper, briefing)

    json_path = Path(EXPORT_DIR) / f"{stem}.json"
    json_path.write_text(json.dumps(combined, indent=2), encoding="utf-8")

    md_path = Path(EXPORT_DIR) / f"{stem}.md"
    md_path.write_text(_to_markdown(combined), encoding="utf-8")

    return str(json_path)
