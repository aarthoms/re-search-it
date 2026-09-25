"""Interactive CLI: query -> retrieval -> parse/chunk/embed -> briefing -> chat loop.

The chat loop resolves each message's intent before touching retrieval:
"new_paper" (the user named/implied a different paper -> switch and reset
conversation state) vs. "follow_up" (a question about the currently loaded
paper, with pronouns resolved against history). Without this split, a message
like "what does 2301.12345 say?" gets answered from whatever paper happens to
already be loaded -- a grounded-looking answer about the wrong paper, which is
worse than an honest "I don't have that."
"""

import itertools
import random
import sys
import threading
import time

from colorama import Fore, Style, init as colorama_init

from re_search_it.graph import build_retrieval_graph
from re_search_it.qa_graph import build_qa_graph
from re_search_it.tools.arxiv_client import find_arxiv_id
from re_search_it.tools.cohere_client import resolve_intent

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
    """Raised to unwind out of the chat loop and quit the whole program."""


def _print_paper(paper: dict) -> None:
    print(f"\n  {_c(_TITLE, paper['arxiv_id'])}  {paper['title']}")
    print(f"  {_c(_DIM, 'authors:')} {', '.join(paper['authors'][:3])}"
          + (" et al." if len(paper["authors"]) > 3 else ""))
    if "relevance_score" in paper:
        print(f"  {_c(_DIM, 'relevance:')} {paper['relevance_score']:.3f}")
    print(f"  {_c(_DIM, paper['pdf_url'])}")


def _handle_query(graph, query: str) -> dict | None:
    with _Spinner("Searching arXiv..."):
        result = graph.invoke({"query": query, "conversation_history": []})

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

        print(f"\n{_c(_HEADER, 'Found paper:')}")
        _print_paper(result["selected_paper"])

        if result.get("low_confidence"):
            score = result["selected_paper"].get("relevance_score", 0)
            print(
                f"\n  {_c(_WARN, f'! This is a weak match (relevance {score:.2f}) -- it may not '
                                  'directly address your query. Consider rephrasing.')}"
            )

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


def _chat_loop(retrieval_graph, qa_graph, paper_state: dict) -> None:
    prompt_hint = "Ask a question, name another paper to switch, 'new' for a fresh topic search, 'exit' to quit."
    print(f"\n{_c(_DIM, prompt_hint)}")

    raw_log: list[str] = []
    conversation_history: list[dict] = []

    while True:
        try:
            message = input(_c(_PROMPT, "? ")).strip()
        except (EOFError, KeyboardInterrupt):
            print()
            raise _ExitCLI

        if not message:
            continue
        if message.lower() in {"exit", "quit"}:
            raise _ExitCLI
        if message.lower() == "new":
            return

        raw_log.append(message)

        direct_id = find_arxiv_id(message)
        if direct_id:
            intent_kind, target_id, standalone_query = "new_paper", direct_id, message
        else:
            with _Spinner("Working out what you mean..."):
                intent = resolve_intent(
                    message,
                    raw_log[-1 - RAW_LOG_WINDOW : -1],
                    paper_state["selected_paper"]["title"],
                )
            intent_kind, target_id, standalone_query = (
                intent.intent, intent.paper_id, intent.standalone_query
            )

        if intent_kind == "new_paper":
            print(f"\n{_c(_DIM, f'Switching to {target_id}...')}")
            new_state = _handle_query(retrieval_graph, target_id)
            if new_state is None:
                print(f"  {_c(_WARN, 'Staying on the current paper.')}")
                continue
            paper_state = new_state
            conversation_history = []
            print(f"\n{_c(_DIM, prompt_hint)}")
            continue

        try:
            with _Spinner():
                result = qa_graph.invoke(
                    {**paper_state, "question": standalone_query, "conversation_history": conversation_history}
                )
            conversation_history = result["conversation_history"]
            _print_answer(result["answer"])
        except Exception as exc:
            print(f"\n  {_c(_ERROR, '!')} Unexpected error: {exc}")


def main() -> None:
    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8")

    print(_c(_HEADER, "re-search-it -- arXiv paper digest & QA agent"))
    print(_c(_DIM, "Enter a research topic or a direct arXiv ID (e.g. 2301.12345)."))
    print(_c(_DIM, "Type 'exit' or 'quit' to leave.\n"))

    retrieval_graph = build_retrieval_graph()
    qa_graph = build_qa_graph()

    while True:
        try:
            query = input(_c(_PROMPT, "> ")).strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not query:
            continue
        if query.lower() in {"exit", "quit"}:
            break

        try:
            paper_state = _handle_query(retrieval_graph, query)
        except Exception as exc:
            print(f"\n  {_c(_ERROR, '!')} Unexpected error: {exc}")
            continue

        if paper_state is not None:
            try:
                _chat_loop(retrieval_graph, qa_graph, paper_state)
            except _ExitCLI:
                break


if __name__ == "__main__":
    main()
