"""Section-aware chunking: split each section's text into overlapping word-bounded chunks."""

CHUNK_SIZE_WORDS = 200
CHUNK_OVERLAP_WORDS = 40


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
