"""Persistent research vocabulary: concept records learned across sessions.

SQLite, not a knowledge graph -- one flat table, small enough to load and
scan in Python for lookups (this is a vocabulary of concepts encountered
across a user's sessions, not a corpus; it will stay small). Each row
represents a research CONCEPT (e.g. "Joint Embedding Predictive
Architecture"), not a cached query -- the point is that "JEPA" resolved once
should stay resolved for every future query that mentions it, regardless of
the rest of the query's wording.
"""

import json
import re
import sqlite3
from datetime import datetime, timezone

from re_search_it.config import RESEARCH_MEMORY_DB_PATH

_connection: sqlite3.Connection | None = None


def _get_connection() -> sqlite3.Connection:
    global _connection
    if _connection is None:
        _connection = sqlite3.connect(RESEARCH_MEMORY_DB_PATH, check_same_thread=False)
        _connection.execute(
            """
            CREATE TABLE IF NOT EXISTS concepts (
                canonical_name TEXT PRIMARY KEY,
                aliases TEXT NOT NULL,
                related_terms TEXT NOT NULL,
                domains TEXT NOT NULL,
                source TEXT NOT NULL,
                confidence REAL NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        _connection.commit()
    return _connection


def _row_to_concept(row: tuple) -> dict:
    canonical_name, aliases, related_terms, domains, source, confidence, created_at, updated_at = row
    return {
        "canonical_name": canonical_name,
        "aliases": json.loads(aliases),
        "related_terms": json.loads(related_terms),
        "domains": json.loads(domains),
        "source": json.loads(source),
        "confidence": confidence,
        "created_at": created_at,
        "updated_at": updated_at,
    }


def all_concepts() -> list[dict]:
    rows = _get_connection().execute("SELECT * FROM concepts").fetchall()
    return [_row_to_concept(r) for r in rows]


def find_known_concepts(text: str) -> list[dict]:
    """Which known concepts does `text` mention? Matches on canonical_name
    or any alias, whole-word/case-insensitive (so an alias like "AI" doesn't
    match inside an unrelated word like "explain")."""
    matches = []
    for concept in all_concepts():
        names = [concept["canonical_name"], *concept["aliases"]]
        if any(re.search(rf"\b{re.escape(n)}\b", text, re.IGNORECASE) for n in names if n):
            matches.append(concept)
    return matches


def save_concept(concept: dict) -> None:
    """Upsert a concept record. Never downgrades an existing high-confidence
    concept with weaker incoming information -- if a concept with the same
    canonical_name already exists and the new confidence is lower, aliases/
    related_terms are still merged (more recall never hurts) but the higher
    confidence and existing source provenance are kept.
    """
    conn = _get_connection()
    now = datetime.now(timezone.utc).isoformat()
    existing_row = conn.execute(
        "SELECT * FROM concepts WHERE canonical_name = ?", (concept["canonical_name"],)
    ).fetchone()

    if existing_row is None:
        conn.execute(
            "INSERT INTO concepts VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                concept["canonical_name"],
                json.dumps(concept.get("aliases", [])),
                json.dumps(concept.get("related_terms", [])),
                json.dumps(concept.get("domains", [])),
                json.dumps(concept.get("source", [])),
                concept["confidence"],
                now,
                now,
            ),
        )
    else:
        existing = _row_to_concept(existing_row)
        merged_aliases = list(dict.fromkeys(existing["aliases"] + concept.get("aliases", [])))
        merged_related = list(dict.fromkeys(existing["related_terms"] + concept.get("related_terms", [])))
        merged_domains = list(dict.fromkeys(existing["domains"] + concept.get("domains", [])))
        merged_source = list(dict.fromkeys(existing["source"] + concept.get("source", [])))
        kept_confidence = max(existing["confidence"], concept["confidence"])

        conn.execute(
            """UPDATE concepts SET aliases=?, related_terms=?, domains=?,
               source=?, confidence=?, updated_at=? WHERE canonical_name=?""",
            (
                json.dumps(merged_aliases),
                json.dumps(merged_related),
                json.dumps(merged_domains),
                json.dumps(merged_source),
                kept_confidence,
                now,
                concept["canonical_name"],
            ),
        )
    conn.commit()
