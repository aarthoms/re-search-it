"""Regression tests for the persistent research-memory + discovery subsystem."""

from unittest.mock import patch

from re_search_it import research_memory
from re_search_it.schemas import ConceptDiscovery
from re_search_it.tools import research_discovery


def _mojo_concept() -> dict:
    return {
        "canonical_name": "Mojo",
        "aliases": ["MOJO programming language", "Mojo language"],
        "related_terms": ["MLIR", "Python superset"],
        "domains": ["programming languages"],
        "source": ["llm", "arxiv_validated"],
        "confidence": 0.9,
    }


# 1. Known term -> memory hit after it has previously been learned.
def test_known_term_memory_hit():
    research_memory.save_concept(_mojo_concept())

    matches = research_memory.find_known_concepts("MOJO programming language benchmarks")
    assert len(matches) == 1
    assert matches[0]["canonical_name"] == "Mojo"


def test_unknown_term_memory_miss():
    matches = research_memory.find_known_concepts("something entirely unrelated")
    assert matches == []


# 2. New term -> discovery -> persistence -> (arXiv retrieval happens downstream, not tested here).
def test_new_term_discovery_persists_on_success():
    fake_resolution = ConceptDiscovery(
        canonical_name="Joint Embedding Predictive Architecture",
        aliases=["JEPA", "I-JEPA", "V-JEPA"],
        related_terms=["world models", "predictive representation learning"],
        domains=["machine learning"],
        confidence=0.92,
    )

    with patch.object(research_discovery, "discover_terminology", return_value=fake_resolution), \
         patch.object(research_discovery, "search_by_topic", return_value=[{"arxiv_id": "1"}]):
        record = research_discovery.discover_and_persist("JEPA", "JEPA world models")

    assert record is not None
    assert record["canonical_name"] == "Joint Embedding Predictive Architecture"
    assert "JEPA" in record["aliases"]

    # Persisted for real -- a later lookup finds it without re-discovering.
    matches = research_memory.find_known_concepts("JEPA vs reinforcement learning")
    assert len(matches) == 1
    assert matches[0]["canonical_name"] == "Joint Embedding Predictive Architecture"


def test_discovery_does_not_persist_low_confidence():
    fake_resolution = ConceptDiscovery(
        canonical_name="Some Guess",
        aliases=["SG"],
        related_terms=[],
        domains=[],
        confidence=0.2,  # below MIN_DISCOVERY_CONFIDENCE
    )

    with patch.object(research_discovery, "discover_terminology", return_value=fake_resolution), \
         patch.object(research_discovery, "search_by_topic", return_value=[{"arxiv_id": "1"}]) as mock_search:
        record = research_discovery.discover_and_persist("SG", "SG something")

    assert record is None
    mock_search.assert_not_called()  # don't even bother validating a guess
    assert research_memory.find_known_concepts("SG something") == []


def test_discovery_does_not_persist_without_arxiv_evidence():
    fake_resolution = ConceptDiscovery(
        canonical_name="Hallucinated Concept",
        aliases=["HC"],
        related_terms=[],
        domains=[],
        confidence=0.95,  # high self-confidence, but...
    )

    with patch.object(research_discovery, "discover_terminology", return_value=fake_resolution), \
         patch.object(research_discovery, "search_by_topic", return_value=[]):  # ...no real evidence
        record = research_discovery.discover_and_persist("HC", "HC something")

    assert record is None
    assert research_memory.find_known_concepts("HC something") == []


# 3. Second query for the same concept -> memory hit, no unnecessary re-discovery.
def test_second_query_hits_memory_without_rediscovering():
    research_memory.save_concept(
        {
            "canonical_name": "Joint Embedding Predictive Architecture",
            "aliases": ["JEPA"],
            "related_terms": ["world models"],
            "domains": ["machine learning"],
            "source": ["llm", "arxiv_validated"],
            "confidence": 0.9,
        }
    )

    with patch.object(research_discovery, "discover_terminology") as mock_discover:
        matches = research_memory.find_known_concepts("JEPA vs reinforcement learning")
        assert len(matches) == 1
        mock_discover.assert_not_called()


def test_save_concept_never_downgrades_confidence():
    research_memory.save_concept({**_mojo_concept(), "confidence": 0.9})
    research_memory.save_concept({**_mojo_concept(), "confidence": 0.3, "aliases": ["a new alias"]})

    matches = research_memory.find_known_concepts("Mojo")
    assert matches[0]["confidence"] == 0.9  # kept the higher confidence
    assert "a new alias" in matches[0]["aliases"]  # but still merged the new alias
