"""Interactive CLI: one loop, three operations, routed before any retrieval runs.

Every message is classified into discovery (many papers on a topic), lookup
(one specific paper), or qa (a question about the paper already loaded).
Without this split up front, the system collapses "find me all papers on X"
and "what does this paper say about X" into the same operation -- answering
from whatever's already loaded, which is a routing bug, not a ranking one.

Discovery deliberately stops at a ranked candidate list (see discovery.py) --
it does not fetch/parse/chunk/embed every candidate, only whichever one the
user picks via a follow-up lookup.
"""

import itertools
import random
import sys
import threading
import time
from dataclasses import dataclass, field

from colorama import Fore, Style, init as colorama_init

from re_search_it.discovery import discover_papers
from re_search_it.graph import build_retrieval_graph
from re_search_it.qa_graph import build_qa_graph
from re_search_it.tools.arxiv_client import find_arxiv_id
from re_search_it.tools.cohere_client import route_top_level
from re_search_it.tools.reference_resolver import looks_like_citation, resolve_reference

colorama_init(autoreset=True)

_HEADER = Fore.CYAN + Style.BRIGHT
_TITLE = Fore.GREEN + Style.BRIGHT
_DIM = Fore.WHITE + Style.DIM
_WARN = Fore.YELLOW
_ERROR = Fore.RED + Style.BRIGHT
_PROMPT = Fore.MAGENTA + Style.BRIGHT
_LABEL = Fore.CYAN

_WAIT_MESSAGES = [
    "Digging through the paper...",
    "Consulting the abstract...",
    "Reranking evidence...",
    "Chasing citations...",
    "Reading between the lines...",
    "Asking Cohere nicely...",
    "Untangling the methodology section...",
]

RAW_LOG_WINDOW = 6
PROMPT_HINT = (
    "Ask a question, search for papers ('find all papers on X'), name a "
    "specific paper, or 'new' to reset. 'exit' to quit."
)


def _c(color: str, text: str) -> str:
    return f"{color}{text}{Style.RESET_ALL}"


class _Spinner:
    """Background-thread spinner shown while a blocking call runs."""

    def __init__(self, message: str | None = None):
        self._message = message or random.choice(_WAIT_MESSAGES)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _spin(self) -> None:
        for frame in itertools.cycle("|/-\\"):
            if self._stop.is_set():
                break
            sys.stdout.write(f"\r{_c(_DIM, self._message)} {frame} ")
            sys.stdout.flush()
            time.sleep(0.1)
        sys.stdout.write("\r" + " " * (len(self._message) + 4) + "\r")
        sys.stdout.flush()

    def __enter__(self) -> "_Spinner":
        if sys.stdout.isatty():
            self._thread = threading.Thread(target=self._spin, daemon=True)
            self._thread.start()
        return self

    def __exit__(self, *exc_info) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join()


class _ExitCLI(Exception):
    """Raised to unwind out of the loop and quit the whole program."""


@dataclass
class ResearchState:
    """One object, updated every turn, instead of loose locals threaded
    through the loop by hand. `last_query`/`last_intent` exist specifically
    so search_refinement can reuse the ORIGINAL search instead of treating
    the refinement phrasing itself ("even weaker matches") as a new topic."""

    last_query: str | None = None
    last_intent: dict | None = None
    candidates: list[dict] = field(default_factory=list)
    active_paper: dict | None = None
    conversation_history: list[dict] = field(default_factory=list)
    raw_log: list[str] = field(default_factory=list)

    def reset_paper(self) -> None:
        self.active_paper = None
        self.conversation_history = []

    def reset_all(self) -> None:
        self.last_query = None
        self.last_intent = None
        self.candidates = []
        self.reset_paper()


def _print_paper(paper: dict) -> None:
    print(f"\n  {_c(_TITLE, paper['arxiv_id'])}  {paper['title']}")
    print(f"  {_c(_DIM, 'authors:')} {', '.join(paper['authors'][:3])}"
          + (" et al." if len(paper["authors"]) > 3 else ""))
    if "relevance_score" in paper:
        print(f"  {_c(_DIM, 'relevance:')} {paper['relevance_score']:.3f}")
    print(f"  {_c(_DIM, paper['pdf_url'])}")


def _do_lookup(retrieval_graph, target: str) -> dict | None:
    """Run the full single-paper pipeline (fetch/parse/chunk/embed/summarize)."""
    with _Spinner("Fetching and indexing..."):
        result = retrieval_graph.invoke({"query": target, "conversation_history": []})

    if result.get("error"):
        print(f"\n  {_c(_ERROR, '!')} {result['error']}")
        return None

    if result.get("is_direct_id"):
        print(f"\n{_c(_HEADER, 'Found paper:')}")
        _print_paper(result["selected_paper"])
    else:
        diagnostics = (
            f"[diagnostics] {len(result['candidates'])} candidates, "
            f"search terms: {', '.join(result.get('search_terms', []))}"
        )
        category = result.get("category")
        if category:
            diagnostics += f", category: {category}"
        print(f"\n{_c(_DIM, diagnostics)}")

        if not result.get("answerable", True):
            # Top score was below ANSWERABLE_FLOOR -- e.g. 0.03. Refuse to
            # present it as "the answer"; show it as a weak candidate instead.
            top = result["candidates"][0]
            warning = (
                f"! No confident match for that (best relevance "
                f"{top.get('relevance_score', 0):.3f}). Closest candidates:"
            )
            print(f"\n  {_c(_WARN, warning)}")
            for c in result["candidates"][:5]:
                print(f"  - {c['arxiv_id']}  {c['title']}  (relevance: {c.get('relevance_score', 0):.3f})")
            discovery_hint = "Try discovery instead: 'find all papers about X'."
            print(f"\n{_c(_DIM, discovery_hint)}")
            return None

        print(f"\n{_c(_HEADER, 'Found paper:')}")
        _print_paper(result["selected_paper"])

        if result.get("low_confidence"):
            score = result["selected_paper"].get("relevance_score", 0)
            warning = (
                f'! This is a weak match (relevance {score:.2f}) -- it may not directly '
                'address your query. Try discovery instead: "find all papers about X".'
            )
            print(f"\n  {_c(_WARN, warning)}")

        others = result["candidates"][1:4]
        if others:
            print(f"\n{_c(_DIM, 'Other candidates:')}")
            for c in others:
                print(f"  - {c['arxiv_id']}  {c['title']}  (relevance: {c.get('relevance_score', 0):.3f})")

    _print_parse_summary(result)
    return result


def _print_parse_summary(result: dict) -> None:
    sections = result.get("parsed_sections")
    if not sections:
        return
    sections_line = f"Parsed sections ({len(sections)}): " + ", ".join(sections.keys())
    print(f"\n{_c(_DIM, sections_line)}")

    chunk_line = (
        f"Chunked & embedded: {result.get('chunk_count', 0)} chunks -> "
        f"{result['vector_collection_id']}"
    )
    print(_c(_DIM, chunk_line))

    _print_briefing(result.get("briefing"))


def _print_briefing(briefing: dict | None) -> None:
    if not briefing:
        return
    print(f"\n{_c(_HEADER, '--- Executive Briefing ---')}")
    print(f"{_c(_LABEL, 'TL;DR:')} {briefing['tldr']}")
    print(f"\n{_c(_LABEL, 'Problem:')} {briefing['problem']}")
    print(f"\n{_c(_LABEL, 'Approach:')} {briefing['approach']}")
    print(f"\n{_c(_LABEL, 'Key findings:')}")
    for f in briefing["key_findings"]:
        print(f"  - {f}")
    print(f"\n{_c(_LABEL, 'Limitations:')}")
    for lim in briefing["limitations"]:
        print(f"  - {lim}")


def _print_answer(answer: dict) -> None:
    print(f"\n{answer['text']}")
    if answer["sources"]:
        sections = ", ".join(sorted({s["section"] for s in answer["sources"]}))
        print(f"{_c(_DIM, f'Sources: {sections}')}")
    if answer["confidence"] == "low":
        print(f"{_c(_WARN, '(low-confidence -- evidence for this was weak)')}")
    print()


def _do_discovery(topic: str, relaxed: bool = False) -> list[dict]:
    with _Spinner("Searching broadly across arXiv..."):
        result = discover_papers(topic, relaxed=relaxed)

    diagnostics = (
        f"[diagnostics] {len(result['queries'])} query formulations, "
        f"{result['raw_count']} candidates retrieved, {result['unique_count']} unique"
    )
    print(f"\n{_c(_DIM, diagnostics)}")

    filters = result.get("filters") or {}
    if filters.get("authors") or filters.get("date_from") or filters.get("date_to"):
        filter_line = (
            f"[diagnostics] filters -- authors: {filters.get('authors') or 'none'}, "
            f"date range: {filters.get('date_from') or '-inf'} to {filters.get('date_to') or 'now'}"
        )
        print(_c(_DIM, filter_line))

    if not result["results"]:
        print(f"\n  {_c(_WARN, f'No sufficiently relevant papers found for \"{topic}\".')}")
        return []

    found_header = f'Found {len(result["results"])} potentially relevant papers on "{topic}":'
    print(f"\n{_c(_HEADER, found_header)}")
    for i, p in enumerate(result["results"], 1):
        print(f"\n  {_c(_TITLE, str(i) + '.')} {p['title']} ({p['arxiv_id']})")
        print(f"     relevance: {p['relevance_score']:.2f} -- {p.get('relation', '')}")

    hint = "Say e.g. 'read paper 2' to load one and get its briefing."
    print(f"\n{_c(_DIM, hint)}")
    return result["results"]


def main() -> None:
    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8")

    print(_c(_HEADER, "re-search-it -- arXiv paper digest & QA agent"))
    print(_c(_DIM, PROMPT_HINT + "\n"))

    retrieval_graph = build_retrieval_graph()
    qa_graph = build_qa_graph()
    state = ResearchState()

    while True:
        try:
            message = input(_c(_PROMPT, "> ")).strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not message:
            continue
        if message.lower() in {"exit", "quit"}:
            break
        if message.lower() == "new":
            state.reset_all()
            print(f"\n{_c(_DIM, 'Reset. ' + PROMPT_HINT)}")
            continue

        state.raw_log.append(message)

        try:
            direct_id = find_arxiv_id(message)
            resolved_citation = None
            if not direct_id and state.active_paper and looks_like_citation(message):
                # A bracketed citation like "Falcon [Almazrouei et al., 2023]"
                # is a bibliographic reference, not a fresh literature search --
                # try to resolve it deterministically before ever asking the
                # LLM router or touching semantic discovery.
                resolved_citation = resolve_reference(
                    message, state.active_paper.get("references", [])
                )

            if direct_id:
                mode, topic, paper_id, selection, standalone_query = (
                    "lookup", None, direct_id, None, message
                )
            elif resolved_citation:
                mode, topic, paper_id, selection, standalone_query = (
                    "lookup", resolved_citation, None, None, message
                )
            else:
                with _Spinner("Routing..."):
                    intent = route_top_level(
                        message,
                        state.raw_log[-1 - RAW_LOG_WINDOW : -1],
                        state.active_paper["selected_paper"]["title"] if state.active_paper else None,
                        state.candidates,
                        has_last_search=state.last_query is not None,
                    )
                mode, topic, paper_id, selection, standalone_query = (
                    intent.mode, intent.topic, intent.paper_id, intent.selection, intent.standalone_query
                )

            if mode == "discovery":
                # Pass the raw message, not the router's extracted `topic` --
                # discover_papers does its own author/date-filter extraction
                # on the full text, and the router's topic field may have
                # already stripped exactly that phrasing away (e.g. "Recent
                # developments in X" -> topic "X", losing "recent").
                discovery_query = message
                state.candidates = _do_discovery(discovery_query)
                state.last_query = discovery_query
                state.last_intent = intent.model_dump() if not direct_id else None
                continue

            if mode == "search_refinement":
                # Reuse the ORIGINAL search, not this message's wording --
                # "even weaker matches" is not a topic to embed.
                refine_topic = state.last_query or topic or message
                state.candidates = _do_discovery(refine_topic, relaxed=True)
                continue

            if mode == "lookup":
                if selection and 1 <= selection <= len(state.candidates):
                    target = state.candidates[selection - 1]["arxiv_id"]
                else:
                    target = paper_id or topic or message
                    state.last_query = target
                new_state = _do_lookup(retrieval_graph, target)
                if new_state is not None:
                    state.active_paper = new_state
                    state.conversation_history = []
                    print(f"\n{_c(_DIM, PROMPT_HINT)}")
                continue

            # mode == "qa" -- zero arXiv/discovery calls, only the paper
            # already loaded is touched.
            if state.active_paper is None:
                no_paper_msg = "No paper loaded yet -- search for one first."
                print(f"\n  {_c(_WARN, no_paper_msg)}")
                continue

            with _Spinner():
                result = qa_graph.invoke(
                    {
                        **state.active_paper,
                        "question": standalone_query,
                        "conversation_history": state.conversation_history,
                    }
                )
            state.conversation_history = result["conversation_history"]
            _print_answer(result["answer"])

        except Exception as exc:
            print(f"\n  {_c(_ERROR, '!')} Unexpected error: {exc}")


if __name__ == "__main__":
    main()
