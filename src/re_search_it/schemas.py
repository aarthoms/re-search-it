"""Validated output schemas."""

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
