# re-search-it
An Autonomous arXiv Paper Digest and QA Agent

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate      # Windows
source .venv/bin/activate   # macOS/Linux
pip install -r requirements.txt
cp .env.example .env        # then fill in your API keys
```

## Tech Stack

- **Orchestration:** LangGraph
- **LLM:** Cohere `command-r-plus` (fallback: Gemini)
- **Embeddings/Rerank:** Cohere `embed-v3`/`embed-4`, `rerank-v3.5`
- **Vector DB:** ChromaDB
- **Paper source:** arXiv API

See `memory.md` for full architecture notes.
