"""Node: chunk parsed sections, embed each chunk, store into the paper's vector collection.

Each chunk is stored with its original text (unmodified) as the Chroma
"document", its embedding vector, and a small metadata dict for filtering/
citation -- no separate storage layer, Chroma holds all three together.
"""

from re_search_it.state import PaperState
from re_search_it.tools.chunker import chunk_sections
from re_search_it.tools.cohere_client import embed
from re_search_it.tools.vector_store import add_chunks


def chunk_embed(state: PaperState) -> PaperState:
    arxiv_id = state["selected_paper"]["arxiv_id"]
    collection_id = state["vector_collection_id"]

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
