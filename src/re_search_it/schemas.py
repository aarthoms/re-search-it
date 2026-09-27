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


class QueryExpansion(BaseModel):
    """Structured decomposition of a research query into retrieval concepts,
    used to generate several independent LOOSE arXiv search formulations
    instead of one restrictive phrase that requires every concept to
    co-occur verbatim. All the concept-extraction fields are advisory
    context for the model producing search_formulations -- only
    search_formulations and category actually drive retrieval."""

    entities: list[str] = Field(default_factory=list, description="Named things: languages, models, datasets, systems")
    topics: list[str] = Field(default_factory=list, description="Broader subject areas")
    tasks: list[str] = Field(default_factory=list, description="Actions: benchmark, train, compare, evaluate")
    comparators: list[str] = Field(default_factory=list, description="Things being compared against, if any")
    domains: list[str] = Field(default_factory=list, description="1-2 broad fields, e.g. 'high-performance computing'")
    category: str | None = Field(
        default=None,
        description="A single arXiv category if clearly applicable, else null. "
        "Advisory only -- retrieval never hard-requires it.",
    )
    search_formulations: list[str] = Field(
        description="6-10 short (2-4 word) loose arXiv search phrases. Each "
        "combines AT MOST 2 concepts (e.g. entity + task, or topic + "
        "comparator) -- never every concept in one phrase, never a "
        "Cartesian product of every combination."
    )

    @field_validator("search_formulations")
    @classmethod
    def _non_empty(cls, value: list[str]) -> list[str]:
        if not value:
            raise ValueError("must contain at least one search formulation")
        return value


class ConceptDiscovery(BaseModel):
    """LLM's resolution of an unfamiliar research term (e.g. "JEPA" ->
    "Joint Embedding Predictive Architecture"). Never persisted on trust
    alone -- `confidence` is the model's own self-estimate, and the caller
    additionally requires real arXiv evidence before writing this to
    research_memory (see tools/research_discovery.py)."""

    canonical_name: str = Field(description="The full/formal name of the concept")
    aliases: list[str] = Field(default_factory=list, description="Other short names/acronyms")
    related_terms: list[str] = Field(
        default_factory=list,
        description="Closely related concepts/terminology, for search recall",
    )
    domains: list[str] = Field(default_factory=list, description="1-3 broad fields, e.g. 'machine learning'")
    confidence: float = Field(ge=0.0, le=1.0, description="Self-estimated confidence this is a real, correct resolution")


class RetrievedChunk(BaseModel):
    text: str
    section: str
    chunk_index: int
    relevance_score: float


class TopLevelIntent(BaseModel):
    """Routes every chat message to one of three operations, BEFORE any
    retrieval happens. Without this, the system can't distinguish "find me
    all papers on X" (discovery, many results) from "find the paper about X"
    (lookup, one result) from "what does this paper say about X" (qa, the
    already-loaded paper) -- collapsing all three into "answer from whatever
    paper is loaded" is a routing bug, not a ranking one."""

    mode: Literal["discovery", "lookup", "qa", "search_refinement"] = Field(
        description="discovery: the user wants MULTIPLE papers on a NEW topic "
        "(plural language: 'sources', 'papers', 'all', 'any other'). "
        "lookup: the user wants ONE specific paper (names it, gives an "
        "arXiv ID, or is picking an item from the most recent discovery "
        "list). qa: a question about the paper already loaded -- prefer this "
        "when a paper is loaded and the message is a vague/pronoun follow-up "
        "with no new topic or plural language (e.g. 'tell me what it says', "
        "'what do you mean by that'). search_refinement: the user wants "
        "broader/weaker/more results for the SAME search they just asked for "
        "(e.g. 'even weak matches', 'show me more', 'loosen it up') -- only "
        "valid if a previous search actually happened."
    )
    topic: str | None = Field(
        default=None,
        description="What to search for, for discovery/lookup. Infer from "
        "context if the message doesn't restate it (e.g. 'any other papers "
        "about that?').",
    )
    paper_id: str | None = Field(
        default=None, description="arXiv ID if the message names one directly"
    )
    selection: int | None = Field(
        default=None,
        description="1-based index into the most recent discovery list, if "
        "the user is picking a paper from it (e.g. 'read the second one')",
    )
    standalone_query: str = Field(
        description="The message rewritten as self-contained, with pronouns "
        "resolved via history. Only used for mode='qa'."
    )
