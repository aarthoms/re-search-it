"""Interactive CLI: query -> retrieval -> parse/chunk/embed -> briefing -> QA loop."""

import sys

from colorama import Fore, Style, init as colorama_init

from re_search_it.graph import build_retrieval_graph
from re_search_it.qa_graph import build_qa_graph

colorama_init(autoreset=True)

_HEADER = Fore.CYAN + Style.BRIGHT
_TITLE = Fore.GREEN + Style.BRIGHT
_DIM = Fore.WHITE + Style.DIM
_WARN = Fore.YELLOW
_ERROR = Fore.RED + Style.BRIGHT
_PROMPT = Fore.MAGENTA + Style.BRIGHT
_LABEL = Fore.CYAN


def _c(color: str, text: str) -> str:
    return f"{color}{text}{Style.RESET_ALL}"


class _ExitCLI(Exception):
    """Raised to unwind out of the QA sub-loop and quit the whole program."""


def _print_paper(paper: dict) -> None:
    print(f"\n  {_c(_TITLE, paper['arxiv_id'])}  {paper['title']}")
    print(f"  {_c(_DIM, 'authors:')} {', '.join(paper['authors'][:3])}"
          + (" et al." if len(paper["authors"]) > 3 else ""))
    if "relevance_score" in paper:
        print(f"  {_c(_DIM, 'relevance:')} {paper['relevance_score']:.3f}")
    print(f"  {_c(_DIM, paper['pdf_url'])}")


def _handle_query(graph, query: str) -> dict | None:
    result = graph.invoke({"query": query, "conversation_history": []})

    if result.get("error"):
        print(f"\n  {_c(_ERROR, '!')} {result['error']}")
        return None

    if result.get("is_direct_id"):
        print(f"\n{_c(_HEADER, 'Found paper:')}")
        _print_paper(result["selected_paper"])
    else:
        category = result.get("category")
        print(f"\n{_c(_LABEL, 'Search terms used:')} {', '.join(result.get('search_terms', []))}"
              + (f" (category: {category})" if category else ""))
        top_match_header = f"Top match ({len(result['candidates'])} candidates considered):"
        print(_c(_HEADER, top_match_header))
        _print_paper(result["selected_paper"])

        if result.get("low_confidence"):
            print(f"\n  {_c(_WARN, '! Low-confidence match -- try rephrasing your query for better results.')}")

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


def _qa_loop(qa_graph, paper_state: dict) -> None:
    prompt_hint = "Ask a question about this paper ('new' for another paper, 'exit' to quit)."
    print(f"\n{_c(_DIM, prompt_hint)}")
    conversation_history: list[dict] = []

    while True:
        try:
            question = input(_c(_PROMPT, "? ")).strip()
        except (EOFError, KeyboardInterrupt):
            print()
            raise _ExitCLI

        if not question:
            continue
        if question.lower() in {"exit", "quit"}:
            raise _ExitCLI
        if question.lower() == "new":
            return

        try:
            result = qa_graph.invoke(
                {**paper_state, "question": question, "conversation_history": conversation_history}
            )
            conversation_history = result["conversation_history"]
            print(f"\n{result['answer']}\n")
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
                _qa_loop(qa_graph, paper_state)
            except _ExitCLI:
                break


if __name__ == "__main__":
    main()
