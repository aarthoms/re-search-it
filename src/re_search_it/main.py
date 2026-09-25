"""Interactive CLI: query -> retrieval graph -> ranked/selected paper.

Only the retrieval stage is wired so far (query understanding, arXiv fetch,
selection ranking). Parsing, chunking/embedding, summarizing, and QA land in
later build steps.
"""

import sys

from re_search_it.graph import build_retrieval_graph


def _print_paper(paper: dict) -> None:
    print(f"\n  {paper['arxiv_id']}  {paper['title']}")
    print(f"  authors: {', '.join(paper['authors'][:3])}"
          + (" et al." if len(paper["authors"]) > 3 else ""))
    if "relevance_score" in paper:
        print(f"  relevance: {paper['relevance_score']:.3f}")
    print(f"  {paper['pdf_url']}")


def _handle_query(graph, query: str) -> None:
    result = graph.invoke({"query": query, "conversation_history": []})

    if result.get("error"):
        print(f"\n  ! {result['error']}")
        return

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


def _print_parse_summary(result: dict) -> None:
    sections = result.get("parsed_sections")
    if not sections:
        return
    print(f"\nParsed sections ({len(sections)}): {', '.join(sections.keys())}")
    print(f"Vector collection ready: {result['vector_collection_id']}")


def main() -> None:
    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8")

    print("re-search-it -- arXiv paper digest & QA agent")
    print("Enter a research topic or a direct arXiv ID (e.g. 2301.12345).")
    print("Type 'exit' or 'quit' to leave.\n")

    graph = build_retrieval_graph()

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
            _handle_query(graph, query)
        except Exception as exc:
            print(f"\n  ! Unexpected error: {exc}")


if __name__ == "__main__":
    main()
