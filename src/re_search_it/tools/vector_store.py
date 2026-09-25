"""ChromaDB wrapper: persistent local vector store for paper chunks."""

import chromadb
from chromadb.api.models.Collection import Collection

from re_search_it.config import CHROMA_PERSIST_DIR

_client: chromadb.ClientAPI | None = None


def get_client() -> chromadb.ClientAPI:
    global _client
    if _client is None:
        _client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
    return _client


def get_or_create_collection(collection_id: str) -> Collection:
    return get_client().get_or_create_collection(name=collection_id)


def add_chunks(
    collection_id: str,
    ids: list[str],
    documents: list[str],
    embeddings: list[list[float]],
    metadatas: list[dict],
) -> None:
    collection = get_or_create_collection(collection_id)
    collection.add(ids=ids, documents=documents, embeddings=embeddings, metadatas=metadatas)


def query_chunks(collection_id: str, query_embedding: list[float], top_k: int = 5) -> dict:
    collection = get_or_create_collection(collection_id)
    return collection.query(query_embeddings=[query_embedding], n_results=top_k)
