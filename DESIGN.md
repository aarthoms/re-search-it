# Autonomous arXiv Paper Digest & QA Agent — Project Overview

## What This Project Is

An agent that takes a research topic or a specific arXiv paper ID, retrieves the relevant paper(s), parses and indexes their content, produces a structured executive briefing, and answers follow-up questions grounded in the paper's actual text (RAG).

The system is built as an explicit stateful graph — a sequence of nodes operating on shared state — rather than a single monolithic prompt.

## Architecture Summary

- **Orchestration:** state graph with nodes connected by explicit edges (one conditional branch: topic search vs. direct paper lookup)
- **State:** a single typed object (`PaperState`) passed through every node, holding paper metadata, parsed sections, a vector store pointer, chunk count, briefing output, and conversation history
- **Vector store:** separate from state — state holds only a `vector_collection_id` pointer; embeddings and chunk text/metadata live in the vector DB (e.g. Chroma)
- **RAG approach:** top-k retrieval with section-aware chunking; groundedness checked via a relevance/confidence mechanism before answering
- **Failure handling:** at least one explicit graceful-degradation path (e.g. zero arXiv candidates, or PDF parse failure)

## Pipeline Stages

1. Query Understanding — detect topic search vs. direct arXiv ID
2. arXiv Retrieval — fetch candidate metadata via arXiv API
3. Selection/Ranking — pick best candidate (topic search only)
4. Fetch & Parse — download PDF, extract text/sections
5. Chunk & Embed — section-aware chunking, store embeddings
6. Summarize — generate structured executive briefing
7. QA Loop — retrieve relevant chunks, answer grounded in paper content

## Build Order

1. Define `PaperState` schema (typed, all fields explicit)
2. Build skeleton graph with no-op nodes; run end to end
3. Decide and wire one graceful failure branch (e.g. zero candidates)
4. Build and standalone-test tools: arXiv client, PDF parser, chunker, vector store wrapper
5. Implement `query_understanding` (regex/heuristic first, LLM fallback if ambiguous)
6. Implement `arxiv_retrieval` and `selection_ranking`
7. Implement `fetch_parse` with degradation on parse failure
8. Implement `chunk_embed` (section-aware chunking)
9. Implement `summarize` (structured, validated output — limitations field enforced)
10. Implement grounding/eval mechanism for QA (e.g. relevance threshold before answering)
11. Implement `qa` loop with conversation history wired into retrieval/prompt
12. Manually test QA against a deliberate question set: directly answerable, out of scope, partially covered, comparative-external, metadata, multi-turn
13. Write README: architecture description, setup instructions, sample run, design-decisions/tradeoffs section, known limitations
14. Record short video reflection

## Design Decisions (to be filled in as built)

| Decision | Choice | Reason |
|---|---|---|
| Orchestration | | |
| Vector DB | | |
| Chunking strategy | | |
| Ranking heuristic | | |
| Grounding/eval mechanism | | |
| State persistence | | |
| Failure case handled | | |

## Known Limitations / Out of Scope

- No UI beyond CLI
- No non-arXiv sources
- No fine-tuning
- No multi-user auth or deployment infra

## Status

Not yet started / in progress / complete — update as work proceeds.

# Tools and Tech Stack 
=> Cohere command-a-03-2025 (command-r-plus was deprecated by Cohere on 2025-09-15) | Rerank-v3.5 | embed-v3 or embed-4
=> Fallback: Gemini via google-genai SDK (google-generativeai is deprecated upstream)
=> Lang-Graph (chat model wired via langchain-cohere's ChatCohere)
=> ChromaDB as vectorDB

## Verified

- COHERE_API key confirmed working 2026-09-25, tested via raw `cohere` SDK (`ClientV2`) and via `langchain_cohere.ChatCohere`, both against `command-a-03-2025`.

