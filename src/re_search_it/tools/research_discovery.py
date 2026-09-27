"""Bounded terminology discovery: resolve unfamiliar research terms and
persist validated results to research_memory, to improve first-stage
arXiv recall for terminology the keyword search alone can't find (unseen
acronyms like "JEPA", ambiguous jargon, etc.).

No open web search here -- no web-search API is configured for this
project (only COHERE_API exists). Discovery is LLM concept-reasoning
validated against REAL arXiv evidence (does a paper matching the resolved
terminology actually exist?), which is a genuine, checkable signal, unlike
trusting a raw LLM claim. This satisfies "validate through web evidence OR
arXiv evidence" using the evidence source this project actually has.
"""

from re_search_it import research_memory
from re_search_it.tools.arxiv_client import search_by_topic
from re_search_it.tools.cohere_client import discover_terminology

# Below this, don't even bother validating -- the model itself said it's guessing.
MIN_DISCOVERY_CONFIDENCE = 0.5


def _validate_against_arxiv(canonical_name: str, aliases: list[str]) -> bool:
    """Does at least one arXiv result surface for the resolved name or a top
    alias? If not, the LLM's resolution isn't trustworthy enough to persist,
    however confident it claimed to be."""
    for term in [canonical_name, *aliases[:2]]:
        if search_by_topic(f'abs:"{term}" OR ti:"{term}"', max_results=3):
            return True
    return False


def discover_and_persist(term: str, query: str) -> dict | None:
    """Resolve `term` (as it appeared in `query`) and, if it passes both the
    confidence gate and arXiv-evidence validation, persist it to
    research_memory and return the record. Returns None if discovery
    failed, was too low-confidence, or couldn't be validated -- in all
    those cases nothing is written to memory, and the caller should
    proceed without concept augmentation rather than trust a guess.
    """
    concept = discover_terminology(term, query)
    if concept is None:
        print(f"[research-discovery] failed to resolve {term!r}")
        return None

    if concept.confidence < MIN_DISCOVERY_CONFIDENCE:
        print(f"[research-discovery] low confidence ({concept.confidence:.2f}) for {term!r} -- not persisting")
        return None

    if not _validate_against_arxiv(concept.canonical_name, concept.aliases):
        print(
            f"[research-discovery] could not validate {concept.canonical_name!r} "
            "against arXiv evidence -- not persisting"
        )
        return None

    record = {
        "canonical_name": concept.canonical_name,
        "aliases": list(dict.fromkeys([term, *concept.aliases])),
        "related_terms": concept.related_terms[:6],
        "domains": concept.domains[:3],
        "source": ["llm", "arxiv_validated"],
        "confidence": concept.confidence,
    }
    research_memory.save_concept(record)
    print(f"[research-discovery] discovered: {record['canonical_name']} (aliases: {record['aliases']})")
    print(f"[research-memory] persisted: {record['canonical_name']}")
    return record
