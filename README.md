# re-search-it
An Autonomous arXiv Paper Digest and QA Agent

Given a research topic or a direct arXiv ID, the agent searches/reranks candidate papers, downloads and section-splits the PDF, chunks and embeds it into a local vector store, produces a structured executive briefing, and answers follow-up questions grounded in the paper's actual text — with reasoning-aware retrieval for multi-hop questions. Built as an explicit LangGraph state graph rather than a single prompt:

**Retrieval pipeline** (`build_retrieval_graph()`): `query_understanding` (NL query -> arXiv-friendly search phrases + category, via Cohere) -> `arxiv_retrieval` (field-scoped arXiv search, or direct ID lookup) -> `selection_ranking` (Cohere `rerank-v3.5` against the original query, with a low-confidence flag below a relevance floor) -> `fetch_parse` (PDF download + heading-based section splitting, degrades gracefully to an abstract-only fallback if parsing fails) -> `chunk_embed` (overlapping section-aware chunks, embedded with `embed-english-v3.0`, stored in a per-paper ChromaDB collection alongside their original text and section/chunk metadata) -> `summarize` (structured, schema-validated executive briefing — tl;dr, problem, approach, key findings, limitations).

**QA pipeline** (`build_qa_graph()`, run per question against an already-processed paper): `retrieval_planner` (decides whether a question needs a single direct retrieval or should be decomposed into independent subquestions, plus which sections likely hold the answer) -> `retrieve_chunks` (dense retrieval per subquery, section hints as soft preference not hard filter, merged/deduped, reranked) -> an evidence-sufficiency gate that either proceeds to `answer` or loops once through `refine_query` (generates one targeted follow-up query for the specific evidence gap, capped at `MAX_RETRIEVAL_ROUNDS`) -> `answer` (grounded response citing sections, refuses rather than hallucinates when evidence is genuinely insufficient, conversation history threaded across turns).

All of this is reachable through a color-coded REPL-style CLI (`re-search-it`) as well as programmatically.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate      # Windows
source .venv/bin/activate   # macOS/Linux
pip install -r requirements.txt
pip install -e .            # registers the `re-search-it` CLI command
cp .env.example .env        # then fill in your API keys
```

## Usage

```bash
re-search-it
```

Enter a research topic or a direct arXiv ID (e.g. `2301.12345`) at the prompt. Once a paper is found and briefed, ask follow-up questions about it (`new` for another paper, `exit`/`quit` to leave).

## Tech Stack

- **Orchestration:** LangGraph
- **LLM:** Cohere `command-a-03-2025` (fallback: Gemini via `google-genai`)
- **Embeddings/Rerank:** Cohere `embed-english-v3.0`, `rerank-v3.5`
- **Vector DB:** ChromaDB (local, persisted under `data/chroma/`)
- **Paper source:** arXiv API
- **CLI:** colorama for cross-platform ANSI color output

See `memory.md` for full architecture notes and design-decision history.
