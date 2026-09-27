# re-search-it
An Autonomous arXiv Paper Digest and QA Agent

Given a research topic, a direct arXiv ID, or a paper title, the agent routes the request to the right operation — discover many papers on a topic, look up and fully process one specific paper, or answer a question about a paper already loaded — searches/reranks candidates, downloads and section-splits the PDF, chunks and embeds it into a local vector store, produces a structured executive briefing, and answers follow-up questions grounded in the paper's actual text, with reasoning-aware retrieval for multi-hop questions and bounded self-learning for unfamiliar terminology.

Built as explicit LangGraph state graphs — not a single monolithic prompt.

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

One prompt, three things you can do:

```
> find all papers about diffusion models for video generation      (discovery)
> Attention is all you need                                        (lookup — title, ID, or topic)
> what does this paper say about their evaluation setup?           (qa — once a paper is loaded)
```

`new` resets the session, `exit`/`quit` leaves.

## System overview

```mermaid
flowchart TB
    U["User message"] --> R["Router\nroute_top_level()"]

    R -->|discovery| DISC["Discovery pipeline\ndiscovery.py"]
    R -->|lookup| RET["Retrieval graph\nbuild_retrieval_graph()"]
    R -->|qa| QA["QA graph\nbuild_qa_graph()"]
    R -->|search_refinement| DISC

    DISC -->|user picks a result| RET
    RET -->|paper loaded| QA

    style R fill:#5b6ee1,color:#fff
    style DISC fill:#2d9d78,color:#fff
    style RET fill:#d97a3f,color:#fff
    style QA fill:#a34fb0,color:#fff
```

A citation mention like `Falcon [Almazrouei et al., 2023]` skips the router entirely — a deterministic resolver (`reference_resolver.py`) checks the loaded paper's own bibliography first, so it never gets treated as a fresh literature search.

## 1. Retrieval graph — finding and processing one paper

```mermaid
flowchart TD
    START(["query"]) --> QU["query_understanding\ndirect ID? exact title? else: LLM decomposes\nquery into 6-10 loose search formulations"]
    QU --> AR["arxiv_retrieval\nper formulation: unconstrained search\n+ category-scoped search (additive, never a filter)\ndedupe by arXiv ID"]

    AR -->|zero candidates, not yet retried| REC["research_recovery\nmemory lookup -> discovery if miss\nbroadened loose formulations, retry once"]
    AR -->|candidates found| SR["selection_ranking\nCohere rerank-v3.5 vs. ORIGINAL query"]
    REC --> SR

    SR -->|score < ANSWERABLE_FLOOR, not yet retried| REC
    SR -->|score < ANSWERABLE_FLOOR, already retried| FAIL(["no confident match"])
    SR -->|answerable| FP["fetch_parse\ndownload PDF (cached) -> extract text\n-> split into sections -> extract references"]

    FP -->|download failed| FAIL
    FP --> CE["chunk_embed\n200-word overlapping chunks per section\nembed (skipped if already embedded)\n-> store in per-paper Chroma collection"]
    CE --> SUM["summarize\nschema-validated briefing:\ntl;dr / problem / approach / findings / limitations"]
    SUM --> DONE(["paper loaded, ready for QA"])

    style QU fill:#d97a3f,color:#fff
    style AR fill:#d97a3f,color:#fff
    style REC fill:#c0473f,color:#fff
    style SR fill:#d97a3f,color:#fff
    style FP fill:#3f7fd9,color:#fff
    style CE fill:#3f7fd9,color:#fff
    style SUM fill:#2d9d78,color:#fff
    style FAIL fill:#888,color:#fff
```

**Recall vs. precision, kept deliberately separate:** `query_understanding` + `arxiv_retrieval` optimize for *recall* — several independent loose formulations (never one restrictive multi-concept phrase), category treated as an additive signal that can only add candidates, never remove them. `selection_ranking` is the only stage that optimizes for *precision*, reranking the whole candidate pool against the user's original wording and gating on two thresholds:

| Score | Outcome |
|---|---|
| ≥ 0.3 | confident match |
| 0.15 – 0.3 | shown, but flagged low-confidence |
| < 0.15 | refused — one recovery attempt, then gives up rather than presenting a non-match as an answer |

`research_recovery` is bounded to exactly one attempt per query (never loops), and never fires for a direct arXiv ID (a bad literal ID isn't a terminology problem).

## 2. QA graph — answering questions about a loaded paper

```mermaid
flowchart TD
    Q(["question"]) --> PLAN["retrieval_planner\ndirect (one retrieval) or decomposed\n(2-4 independent subquestions)?\nplus section hints"]
    PLAN --> RC["retrieve_chunks\nembed each subquery -> dense search\n(section hints are soft: hinted + whole-paper\nquery both run) -> merge -> rerank"]

    RC -->|evidence weak, round < 2| REF["refine_query\none targeted follow-up query\nfor the specific gap"]
    REF --> RC

    RC -->|evidence sufficient, or round cap hit| ANS["answer\ngrounded generation: explicit vs. inferred\nvs. not-stated, cites sections"]
    ANS --> OUT(["{text, sources, grounded, confidence}"])

    style PLAN fill:#a34fb0,color:#fff
    style RC fill:#a34fb0,color:#fff
    style REF fill:#a34fb0,color:#fff
    style ANS fill:#2d9d78,color:#fff
```

Chunk embeddings use `input_type="search_query"` vs. `"search_document"` asymmetrically (Cohere's embed model expects this), and refinement rounds *accumulate* evidence rather than replacing it.

## 3. Discovery — finding many papers, not committing to one

```mermaid
flowchart LR
    T(["request"]) --> XF["extract_discovery_filters\nauthors, date range, clean topic"]
    XF --> GQ["generate_discovery_queries\n4-6 varied formulations"]
    GQ --> S1["arXiv search per formulation\nAND'd with author/date filters"]
    S1 --> DD["dedupe by arXiv ID"]
    DD --> RR["rerank vs. original request\n(rerank-v3.5)"]
    RR --> FLT["filter by relevance floor"]
    FLT --> REL["describe_relations\none batched LLM call:\nwhy each result is relevant"]
    REL --> LIST(["ranked list — NOT fetched/parsed/embedded"])

    style XF fill:#2d9d78,color:#fff
    style GQ fill:#2d9d78,color:#fff
    style RR fill:#2d9d78,color:#fff
    style REL fill:#2d9d78,color:#fff
```

Discovery deliberately **stops at a list**. A paper only gets the expensive fetch → parse → chunk → embed → summarize treatment once the user picks one (`"read paper 2"`), which routes into the retrieval graph above.

**Author and date-range filtering** ("papers by Almazrouei and Touvron on language models", "recent developments in JEPA", "papers on X before 2005", "papers from 01/01/2026 to 03/04/2026") is extracted from the request up front and applied as a **hard** arXiv-side filter (`au:A1 OR au:A2` AND'd with `submittedDate:[...]`) — unlike the topic search itself, which stays loose for recall, and unlike the LLM-guessed category elsewhere in this project (soft/additive), author/date constraints here are things the user explicitly stated, so a query for "papers before 2005" should never surface a 2010 paper. "Recent"/vague ranges default to a 2-year window from today; an explicit `MM/DD/YYYY` range is used as-is (ambiguous day/month order defaults to `MM/DD`).

## 4. Persistent research memory (bounded, not a knowledge graph)

```mermaid
flowchart TD
    QT(["query mentions a term"]) --> LOOK{"in research_memory?\n(SQLite, canonical name + aliases)"}
    LOOK -->|HIT| USE["fold known aliases/related terms\ninto query expansion"]
    LOOK -->|MISS, and retrieval fails/weak| DISC2["discover_terminology (LLM)\ncanonical name, aliases, related terms,\nself-estimated confidence"]

    DISC2 -->|confidence too low| DROP(["discard — never persisted"])
    DISC2 -->|confidence OK| VAL["validate against REAL arXiv evidence\n(does a paper actually surface for this name?)"]
    VAL -->|no evidence| DROP
    VAL -->|validated| SAVE["persist to SQLite\n(never downgrades existing higher confidence)"]

    style LOOK fill:#c0473f,color:#fff
    style DISC2 fill:#c0473f,color:#fff
    style VAL fill:#c0473f,color:#fff
    style SAVE fill:#c0473f,color:#fff
```

No open web search is wired in (no such API is configured for this project) — "web evidence" is satisfied via real arXiv evidence instead, which this project actually has and can check. A raw LLM claim is never trusted or persisted on its own.

## Known limitations

- Section-splitting is heading-text heuristics, not real PDF layout parsing — Roman-numeral or stylized headings fall through to a coarser `"preamble"` bucket instead of crashing.
- Discovery/recovery can't validate terminology with zero arXiv footprint (no web-search fallback).
- `research_memory`'s lookup scans the full concept table per query — fine at a personal-vocabulary scale, not designed to scale to a large shared vocabulary.
- No automated eval harness for retrieval quality/answer correctness beyond the unit test suite (`pytest tests/`, mocked LLM/network calls) and manual live spot-checks.
- Multi-author date-range discovery (`generate_discovery_queries`) can paraphrase a topic into period-inappropriate terminology for historical searches — e.g. asking for pre-1998 neural network papers can yield formulations like "deep learning architectures," a term that didn't exist yet, correctly returning zero for that specific phrasing even though period-appropriate formulations (e.g. "connectionist systems") would find real results. The date filter mechanism itself is correct (live-verified against real 1990s arXiv papers); the LLM's phrasing isn't yet period-aware.
- Author matching in discovery is `au:` OR-matched and unverified against a full author list (unlike `search_by_author` in the single-paper lookup path, which does verify) — a common surname could pull in an unrelated author's papers.

## Tech Stack

- **Orchestration:** LangGraph
- **LLM:** Cohere `command-a-03-2025` (sole provider — `google-genai`/`GEMINI_API_KEY` are present in requirements/config but not yet wired into any request path; a Gemini fallback is not currently implemented)
- **Embeddings/Rerank:** Cohere `embed-english-v3.0`, `rerank-v3.5`
- **Vector DB:** ChromaDB (local, persisted under `data/chroma/`)
- **Research memory:** SQLite (local, persisted under `data/research_memory.sqlite`)
- **Paper source:** arXiv API
- **CLI:** colorama for cross-platform ANSI color output
- **Tests:** pytest

See `memory.md` for the running design-decision history.
