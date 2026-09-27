"""Shared test fixtures."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pytest

from re_search_it import research_memory


@pytest.fixture(autouse=True)
def isolated_research_memory(tmp_path, monkeypatch):
    """Every test gets a fresh, temporary SQLite file -- never touches the
    real data/research_memory.sqlite, and tests never see each other's
    persisted concepts."""
    db_path = tmp_path / "test_research_memory.sqlite"
    monkeypatch.setattr(research_memory, "RESEARCH_MEMORY_DB_PATH", str(db_path))
    monkeypatch.setattr(research_memory, "_connection", None)
    yield
    research_memory._connection = None
