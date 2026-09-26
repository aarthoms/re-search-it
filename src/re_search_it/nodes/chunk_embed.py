"""Node: chunk parsed sections, embed each chunk, store into the paper's vector collection.

Each chunk is stored with its original text (unmodified) as the Chroma
"document", its embedding vector, and a small metadata dict for filtering/
citation -- no separate storage layer, Chroma holds all three together.

Chroma's collection persists across runs (see vector_store.CHROMA_PERSIST_DIR),
so re-looking-up a paper already processed in a previous session would
otherwise re-embed it from scratch every time -- the expensive step (an
embedding API call per chunk) for work that's already done and sitting on
disk. Skip straight to reporting the existing chunk count instead.
"""

from re_search_it.state import PaperState
from re_search_it.tools.chunker import chunk_sections
from re_search_it.tools.cohere_client import embed
from re_search_it.tools.vector_store import add_chunks, collection_chunk_count


def chunk_embed(state: PaperState) -> PaperState:
    arxiv_id = state["selected_paper"]["arxiv_id"]
    collection_id = state["vector_collection_id"]

    existing_count = collection_chunk_count(collection_id)
    if existing_count > 0:
        return {**state, "chunk_count": existing_count}

    chunks = chunk_sections(state["parsed_sections"])
    if not chunks:
        return {**state, "error": f"No text to chunk for {arxiv_id}"}

    vectors = embed([c["text"] for c in chunks], input_type="search_document")

    ids = [f"{arxiv_id}_{c['section']}_{c['chunk_index']}" for c in chunks]
    documents = [c["text"] for c in chunks]
    metadatas = [
        {
            "arxiv_id": arxiv_id,
            "section": c["section"],
            "chunk_index": c["chunk_index"],
        }
        for c in chunks
    ]

    add_chunks(
        collection_id=collection_id,
        ids=ids,
        documents=documents,
        embeddings=vectors,
        metadatas=metadatas,
    )

    return {**state, "chunk_count": len(chunks)}
