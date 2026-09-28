"""Section-aware chunking: split each section's text into overlapping word-bounded chunks."""

CHUNK_SIZE_WORDS = 200
CHUNK_OVERLAP_WORDS = 40

# Bump whenever parsing, chunking, or the embedding input format changes --
# folded into the Chroma collection name so a paper indexed by an older
# version gets a fresh collection instead of reusing stale chunks.
INDEX_VERSION = 2


def contextualize(section: str, text: str) -> str:
    """Prefix a chunk with its section name for embedding/reranking -- the
    embedding and rerank calls must see the exact same context, so this is
    the one place that format is built."""
    return f"[{section}] {text}"


def chunk_sections(
    sections: dict[str, str],
    chunk_size: int = CHUNK_SIZE_WORDS,
    overlap: int = CHUNK_OVERLAP_WORDS,
) -> list[dict]:
    """Split each section into overlapping chunks.

    Returns a list of {"section": str, "chunk_index": int, "text": str}, in
    section order. Chunking is per-section (never spans two sections), so a
    chunk's metadata always accurately reflects where it came from.
    """
    chunks: list[dict] = []

    for section, text in sections.items():
        words = text.split()
        if not words:
            continue

        step = chunk_size - overlap
        chunk_index = 0
        start = 0
        while start < len(words):
            piece = words[start : start + chunk_size]
            chunks.append(
                {
                    "section": section,
                    "chunk_index": chunk_index,
                    "text": " ".join(piece),
                }
            )
            chunk_index += 1
            start += step

    return chunks
