"""Validated output schemas."""

from typing import Literal

from pydantic import BaseModel, Field, field_validator


class Briefing(BaseModel):
    """Structured executive briefing for a paper. `limitations` is required and
    non-empty -- a briefing that omits limitations misrepresents the paper's
    claims as unconditional, so we refuse to accept one that skips it."""

    tldr: str = Field(description="1-2 sentence plain-language summary")
    problem: str = Field(description="What problem the paper addresses and why it matters")
    approach: str = Field(description="The core method/technique")
    key_findings: list[str] = Field(description="3-5 bullet-point findings/results")
    limitations: list[str] = Field(description="Limitations the paper itself acknowledges")

    @field_validator("key_findings", "limitations")
    @classmethod
    def _non_empty(cls, value: list[str]) -> list[str]:
        if not value:
            raise ValueError("must contain at least one item")
        return value


class RetrievalPlan(BaseModel):
    """A question -> retrieval strategy. `direct` treats the question as a
    single dense-retrieval query. `decomposed` breaks a multi-hop question
    into independently-retrievable subquestions (e.g. "did X outperform Y"
    needs the method, the baseline, AND the results -- one embedding search
    over the raw question rarely surfaces all three)."""

    mode: Literal["direct", "decomposed"]
    subqueries: list[str] = Field(
        description="1 item for direct mode; 2-4 independently-answerable "
        "subquestions for decomposed mode"
    )
    section_hints: list[str] = Field(
        default_factory=list,
        description="Section names (lowercase) where evidence is likely to live, "
        "e.g. ['limitations', 'conclusion']. Empty list if no section is a "
        "clearly better bet than searching the whole paper.",
    )

    @field_validator("subqueries")
    @classmethod
    def _non_empty(cls, value: list[str]) -> list[str]:
        if not value:
            raise ValueError("must contain at least one subquery")
        return value


class RetrievedChunk(BaseModel):
    text: str
    section: str
    chunk_index: int
    relevance_score: float
