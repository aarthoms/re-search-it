"""Interactive CLI: query -> retrieval -> parse/chunk/embed -> briefing -> QA loop."""

import sys

from re_search_it.graph import build_retrieval_graph
from re_search_it.qa_graph import build_qa_graph


class _ExitCLI(Exception):
    """Raised to unwind out of the QA sub-loop and quit the whole program."""


def _print_paper(paper: dict) -> None:
    print(f"\n  {paper['arxiv_id']}  {paper['title']}")
    print(f"  authors: {', '.join(paper['authors'][:3])}"
          + (" et al." if len(paper["authors"]) > 3 else ""))
    if "relevance_score" in paper:
        print(f"  relevance: {paper['relevance_score']:.3f}")
    print(f"  {paper['pdf_url']}")


def _handle_query(graph, query: str) -> dict | None:
    result = graph.invoke({"query": query, "conversation_history": []})

    if result.get("error"):
        print(f"\n  ! {result['error']}")
        return None

    if result.get("is_direct_id"):
        print("\nFound paper:")
        _print_paper(result["selected_paper"])
    else:
        category = result.get("category")
        print(f"\nSearch terms used: {', '.join(result.get('search_terms', []))}"
              + (f" (category: {category})" if category else ""))
        print(f"Top match ({len(result['candidates'])} candidates considered):")
        _print_paper(result["selected_paper"])

        if result.get("low_confidence"):
            print("\n  ! Low-confidence match -- try rephrasing your query for better results.")

        others = result["candidates"][1:4]
        if others:
            print("\nOther candidates:")
            for c in others:
                print(f"  - {c['arxiv_id']}  {c['title']}  (relevance: {c.get('relevance_score', 0):.3f})")

    _print_parse_summary(result)
    return result


def _print_parse_summary(result: dict) -> None:
    sections = result.get("parsed_sections")
    if not sections:
        return
    print(f"\nParsed sections ({len(sections)}): {', '.join(sections.keys())}")
    print(f"Chunked & embedded: {result.get('chunk_count', 0)} chunks -> {result['vector_collection_id']}")

    _print_briefing(result.get("briefing"))


def _print_briefing(briefing: dict | None) -> None:
    if not briefing:
        return
    print("\n--- Executive Briefing ---")
    print(f"TL;DR: {briefing['tldr']}")
    print(f"\nProblem: {briefing['problem']}")
    print(f"\nApproach: {briefing['approach']}")
    print("\nKey findings:")
    for f in briefing["key_findings"]:
        print(f"  - {f}")
    print("\nLimitations:")
    for lim in briefing["limitations"]:
        print(f"  - {lim}")


def _qa_loop(qa_graph, paper_state: dict) -> None:
    print("\nAsk a question about this paper ('new' for another paper, 'exit' to quit).")
    conversation_history: list[dict] = []

    while True:
        try:
            question = input("? ").strip()
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
            print(f"\n  ! Unexpected error: {exc}")


def main() -> None:
    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8")

    print("re-search-it -- arXiv paper digest & QA agent")
    print("Enter a research topic or a direct arXiv ID (e.g. 2301.12345).")
    print("Type 'exit' or 'quit' to leave.\n")

    retrieval_graph = build_retrieval_graph()
    qa_graph = build_qa_graph()

    while True:
        try:
            query = input("> ").strip()
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
            print(f"\n  ! Unexpected error: {exc}")
            continue

        if paper_state is not None:
            try:
                _qa_loop(qa_graph, paper_state)
            except _ExitCLI:
                break


if __name__ == "__main__":
    main()
