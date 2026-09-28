"""Cohere wrapper: query expansion (chat), candidate reranking, embedding, summarization."""

import json
from datetime import date

import cohere
from pydantic import ValidationError

from re_search_it.config import COHERE_API_KEY
from re_search_it.schemas import Briefing, ConceptDiscovery, DiscoveryFilters, QueryExpansion, RetrievalPlan, TopLevelIntent

CHAT_MODEL = "command-a-03-2025"
# All chat calls in this file are extraction, planning, routing,
# summarizing, or answering -- none benefit from randomness.
CHAT_TEMPERATURE = 0.0
RERANK_MODEL = "rerank-v3.5"
EMBED_MODEL = "embed-english-v3.0"
EMBED_BATCH_SIZE = 96

_client = cohere.ClientV2(api_key=COHERE_API_KEY)

_EXPAND_PROMPT = """Decompose this research question into retrieval concepts
for arXiv search. arXiv keyword search needs SHORT, LOOSE phrases -- it does
not understand natural-language intent, and one phrase combining every
concept in the query is too restrictive to find anything (this is the
actual bug being fixed: legitimate questions were producing zero candidates
because the generated search was one long multi-concept phrase).

Extract:
- entities: named things (languages, models, datasets, systems), e.g. "Mojo"
- topics: broader subject areas, e.g. "GPU programming"
- tasks: actions, e.g. "benchmark", "train", "compare"
- comparators: things being compared against, if any, e.g. "CUDA", "HIP"
- domains: 1-2 broad fields, e.g. "high-performance computing"
- category: a single arXiv category code (cs.CL, cs.LG, cs.CV, cs.AI, cs.DC,
  cs.PL, stat.ML, ...) if clearly applicable, else null. This is ADVISORY
  ONLY -- it will never be used as a hard requirement.
- authors: author surname(s) if the query names one (e.g. "Almazrouei 2023
  Falcon" or "paper by Almazrouei in 2023"), else empty list.
- year: a publication year if the query names one, else null.
- search_formulations: 6-10 SHORT (2-4 word) loose search phrases. Each
  phrase combines AT MOST 2 concepts (e.g. an entity + a task, or a topic +
  a comparator) -- never combine every concept into one phrase, and never
  produce a Cartesian product of every combination.

Example, for "benchmark Mojo against similar languages for GPU acceleration":
{{"entities": ["Mojo"], "topics": ["GPU programming", "performance benchmarking"],
  "tasks": ["benchmark", "performance comparison"], "comparators": ["CUDA", "HIP"],
  "domains": ["high-performance computing"], "category": "cs.PL",
  "authors": [], "year": null,
  "search_formulations": ["Mojo GPU benchmark", "Mojo GPU performance",
  "Mojo CUDA performance", "Mojo HIP performance", "Mojo GPU programming",
  "Mojo performance portability"]}}

Respond with ONLY JSON matching that shape.

User query: {query}"""


def expand_query(query: str, known_concepts: list[dict] | None = None) -> dict:
    """Turn a natural-language query into several loose arXiv search
    formulations + an advisory category, via structured concept extraction.

    `known_concepts` (from research_memory) are folded in as extra context
    so the LLM's phrasing benefits from previously-learned aliases/related
    terms for anything the query mentions -- e.g. once "JEPA" has been
    resolved to "Joint Embedding Predictive Architecture", later queries
    mentioning JEPA get that fuller vocabulary for free instead of
    re-deriving it (or failing to) from scratch every time.

    Returns {"terms": list[str], "category": str | None, "authors":
    list[str], "year": int | None}. "terms" now holds 6-10 loose
    formulations rather than 2-3 phrases; "authors"/"year" (new) let
    arxiv_retrieval additively run a precise au: field search alongside the
    normal keyword search when the query names an author -- e.g. "Almazrouei
    2023 Falcon" typed with no paper loaded still gets a structured author
    lookup, not just a generic keyword search that never uses arXiv's au:
    field at all.
    """
    prompt = _EXPAND_PROMPT.format(query=query)
    if known_concepts:
        lines = [
            f"- {c['canonical_name']}: {', '.join([*c['aliases'], *c['related_terms']][:6])}"
            for c in known_concepts
        ]
        prompt += "\n\nKnown terminology for concepts mentioned in this query:\n" + "\n".join(lines)

    response = _client.chat(
        model=CHAT_MODEL,
        temperature=CHAT_TEMPERATURE,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
    )
    text = response.message.content[0].text
    try:
        expansion = QueryExpansion.model_validate(json.loads(text))
    except (json.JSONDecodeError, ValidationError):
        return {"terms": [query], "category": None, "authors": [], "year": None}

    return {
        # Capped at 6 (not 10): each formulation costs up to 2 rate-limited
        # arXiv calls (with/without category), so 10 could mean 20+ calls at
        # ~3s apart per lookup -- a paper title lookup shouldn't take a
        # minute. 6 is still well within the recall gains this expansion
        # exists for.
        "terms": expansion.search_formulations[:6],
        "category": expansion.category,
        "authors": expansion.authors[:2],
        "year": expansion.year,
    }


def rerank(query: str, documents: list[str], top_n: int | None = None) -> list[dict]:
    """Rerank documents (e.g. paper abstracts) against a query.

    Returns a list of {"index": int, "relevance_score": float}, sorted by
    relevance_score descending, indices referring to positions in `documents`.
    """
    if not documents:
        return []
    response = _client.rerank(
        model=RERANK_MODEL,
        query=query,
        documents=documents,
        top_n=top_n or len(documents),
    )
    return [{"index": r.index, "relevance_score": r.relevance_score} for r in response.results]


def embed(texts: list[str], input_type: str = "search_document") -> list[list[float]]:
    """Embed a batch of texts.

    input_type="search_document" for chunks going into the vector store,
    "search_query" for a user question at retrieval time -- Cohere embeds
    these two asymmetrically, so passing the wrong one degrades relevance.
    """
    if not texts:
        return []

    vectors: list[list[float]] = []
    for i in range(0, len(texts), EMBED_BATCH_SIZE):
        batch = texts[i : i + EMBED_BATCH_SIZE]
        response = _client.embed(
            model=EMBED_MODEL,
            texts=batch,
            input_type=input_type,
            embedding_types=["float"],
        )
        vectors.extend(response.embeddings.float_)

    return vectors


_BRIEFING_SCHEMA_HINT = """Respond with ONLY a JSON object matching this exact shape:
{
  "tldr": "1-2 sentence plain-language summary",
  "problem": "what problem the paper addresses",
  "significance": "one paragraph on why this work matters",
  "approach": ["method step/aspect 1", "method step/aspect 2", "..."],
  "key_findings": ["finding 1", "finding 2", "..."],
  "limitations": ["limitation 1 the paper itself states", "..."],
  "follow_up_questions": ["question 1 a reader might ask", "..."]
}
approach, key_findings, limitations, and follow_up_questions must each
contain at least one item. Base every field ONLY on the paper text given
below -- if a section you'd need wasn't included in the excerpt, don't
guess at its content.

follow_up_questions specifically: each one MUST be answerable from THIS
paper's own text (e.g. about its method, dataset, results, or stated
limitations) -- never a question that requires comparing against a
different paper/tool the excerpt doesn't mention (e.g. "how does this
compare to Julia?" when Julia is never discussed). A question this paper
can't answer is a bad demo of the QA feature, not a good one.

For limitations specifically: use ONLY what the paper itself states. If (and
only if) the paper states none at all, you may add at most 1-2 defensible
ones inferred from its scope (e.g. dataset size, domain, evaluation setup) --
but each such item MUST be prefixed literally with "(inferred)" so it's
never mistaken for something the authors actually wrote."""

_SUMMARIZE_PROMPT = """Write a structured executive briefing for this paper.

Title: {title}
Authors: {authors}

Paper text (sections may be truncated):
{sections_text}

{schema_hint}"""

MAX_SUMMARIZE_RETRIES = 2


def summarize_paper(title: str, authors: list[str], sections_text: str) -> Briefing:
    """Generate a validated Briefing for a paper. Retries on schema validation failure."""
    prompt = _SUMMARIZE_PROMPT.format(
        title=title,
        authors=", ".join(authors),
        sections_text=sections_text,
        schema_hint=_BRIEFING_SCHEMA_HINT,
    )
    messages = [{"role": "user", "content": prompt}]

    last_error: Exception | None = None
    for _ in range(MAX_SUMMARIZE_RETRIES):
        response = _client.chat(
            model=CHAT_MODEL,
            temperature=CHAT_TEMPERATURE,
            messages=messages,
            response_format={"type": "json_object"},
        )
        text = response.message.content[0].text
        try:
            return Briefing.model_validate(json.loads(text))
        except (json.JSONDecodeError, ValidationError) as exc:
            last_error = exc
            messages.append({"role": "assistant", "content": text})
            messages.append(
                {
                    "role": "user",
                    "content": f"That JSON was invalid: {exc}. Return corrected JSON only.",
                }
            )

    raise ValueError(f"Failed to produce a valid briefing after retries: {last_error}")


_PLAN_PROMPT = """You plan how to retrieve evidence from a paper's vector store
to answer a user's question.

Decide:
- "direct" if the question is answerable from one focused piece of evidence
  (e.g. "what dataset did they use?", "what model architecture?").
- "decomposed" if the question requires combining multiple independent facts
  (e.g. "did X outperform Y, and why?" needs the method, the baseline, AND
  the results/explanation -- one search rarely surfaces all of that).

For "decomposed", write 2-4 subquestions, each independently answerable by
a single retrieval.

Also name up to 3 section names (lowercase) where the evidence is most
likely to live, from this paper's actual sections: {available_sections}.
Leave section_hints empty if no section is a clearly better bet than
searching the whole paper.

Respond with ONLY JSON matching this shape:
{{
  "mode": "direct" or "decomposed",
  "subqueries": ["..."],
  "section_hints": ["..."]
}}

Question: {question}"""


def plan_retrieval(question: str, available_sections: list[str]) -> RetrievalPlan:
    """Decide direct-vs-decomposed retrieval and which sections to prioritize."""
    response = _client.chat(
        model=CHAT_MODEL,
        temperature=CHAT_TEMPERATURE,
        messages=[
            {
                "role": "user",
                "content": _PLAN_PROMPT.format(
                    question=question, available_sections=", ".join(available_sections)
                ),
            }
        ],
        response_format={"type": "json_object"},
    )
    text = response.message.content[0].text
    try:
        return RetrievalPlan.model_validate(json.loads(text))
    except (json.JSONDecodeError, ValidationError):
        # Fall back to the simplest safe plan rather than failing the whole turn.
        return RetrievalPlan(mode="direct", subqueries=[question], section_hints=[])


_REFINE_PROMPT = """You're gathering evidence from a paper to answer a question.
Here's what you've retrieved so far, and it isn't enough.

Question: {question}

Evidence retrieved so far:
{evidence}

What's the single most important piece of evidence still missing? Write ONE
short, focused search query (not a restatement of the original question) that
would retrieve it. Output ONLY the query text, no commentary."""


def refine_query(question: str, evidence_so_far: list[str]) -> str:
    """Generate one follow-up retrieval query targeting the evidence gap."""
    evidence_text = "\n---\n".join(evidence_so_far) or "(nothing relevant found yet)"
    response = _client.chat(
        model=CHAT_MODEL,
        temperature=CHAT_TEMPERATURE,
        messages=[
            {
                "role": "user",
                "content": _REFINE_PROMPT.format(question=question, evidence=evidence_text),
            }
        ],
    )
    return response.message.content[0].text.strip()


_ANSWER_PROMPT = """Answer the user's question about this paper using ONLY the
evidence excerpts below. If the evidence doesn't fully answer the question,
say so explicitly rather than filling gaps with outside knowledge. Cite which
section(s) you drew from inline, e.g. "(Results)".

Distinguish, in your wording, between:
- facts the evidence EXPLICITLY states (state plainly)
- reasonable INFERENCES from the evidence, e.g. architectural interpretation
  the paper doesn't spell out (flag as "the paper implies..." or "this
  suggests..., though it isn't stated directly")
- things the evidence does NOT cover (say so plainly rather than guessing)

Do not blur these together -- an inference presented as a stated fact is a
grounding failure even if the inference is reasonable.

Two additional rules, because getting these wrong is worse than an
unanswered question:
- Only attribute a number/result to a specific workload, model, or
  experiment if the SAME excerpt actually names that workload. A paper-wide
  or different-experiment's figure ("20x-180x speedup on kernel X") must
  never be presented as if it were the result for whatever the question
  asked about, even if it's the only number in the evidence.
- Always say whether a number is something the authors MEASURED
  themselves, something they PROJECTED/ESTIMATED (e.g. "calibrated from
  published benchmarks"), or something CITED from another source -- if the
  evidence doesn't make this clear, say the provenance is unclear rather
  than presenting it as a measured result by default.

Evidence:
{evidence}

Question: {question}"""


def answer_question(question: str, evidence_chunks: list[dict]) -> str:
    """Generate a grounded answer from reranked evidence chunks.

    Only the evidence + question go to the model -- pronoun/follow-up
    resolution already happened upstream (the router's standalone_query),
    so the answer should rely on retrieved evidence, not accumulated chat
    history.
    """
    evidence_text = "\n---\n".join(
        f"[{c['section']}] {c['text']}" for c in evidence_chunks
    )
    messages = [
        {
            "role": "user",
            "content": _ANSWER_PROMPT.format(evidence=evidence_text, question=question),
        }
    ]
    response = _client.chat(model=CHAT_MODEL, messages=messages, temperature=CHAT_TEMPERATURE)
    return response.message.content[0].text.strip()


_ROUTER_PROMPT = """You route messages for a research-paper assistant with
four operations:

- "discovery": the user wants MULTIPLE papers on a NEW topic -- plural
  language ("sources", "papers", "all", "any other") naming or clearly
  implying a topic that ISN'T just "the ones we already found/loaded".
  IMPORTANT: "sources"/"papers"/"results" referring back to papers already
  found this conversation (e.g. "what does your sources tell?", "what do
  these papers say?") is NOT a new discovery request -- it's a reference
  to existing results. If a paper is loaded, treat that case as "qa"; the
  word "sources" alone is not a topic to search arXiv for.
  ALSO IMPORTANT, the opposite direction: a paper being loaded does NOT
  mean every follow-up is about that paper. Phrases like "has anyone",
  "other papers", "other researchers", "else", "similar work",
  "benchmarked", "trained", "compared" signal the user wants to know what
  the wider LITERATURE says, not just this one document -- e.g. "has
  anyone benchmarked Mojo against CUDA?" or "has anyone else trained a
  model this way?" are discovery requests even with a paper loaded. The
  test is whether the question is about THIS paper's content (qa) or
  about what OTHER work exists (discovery), not whether a paper happens
  to be loaded.
- "lookup": the user wants ONE specific paper -- names it directly, gives an
  arXiv ID, or is picking an item from the most recent discovery list below
  (e.g. "read the second one", "load paper 3", "the MojoBench one").
- "qa": a question about the paper ALREADY LOADED (below). PREFER THIS when
  a paper is loaded and the message is a vague/pronoun follow-up with no new
  topic and no plural language naming something NEW (e.g. "tell me what it
  says", "what do you mean by that", "what is it about", "what does your
  sources tell?" -- referring to what's already loaded/found, not a fresh
  search). Do NOT treat a generic follow-up as a new search just because
  it's phrased as a question or uses a plural word.
- "search_refinement": the user wants BROADER/WEAKER/MORE results for the
  SAME search they just ran, not a new topic (e.g. "even weak matches",
  "bring me the slightest match for the query I asked above", "show me
  more", "loosen it up"). Only valid if `has_last_search` is true below --
  if false, treat it as "discovery" instead (there's nothing to refine).

has_last_search: {has_last_search}
Currently loaded paper: {active_paper}
Most recent discovery list:
{discovery_list}

Recent conversation:
{history}

New message: {message}

Examples:
"MOJO PROGRAMMING LANGUAGE" (nothing loaded yet) -> discovery or lookup depending on phrasing/plurality
"Tell me what it says?" (paper loaded) -> qa
"find all papers about X" -> discovery
"bring me the slightest matches" (has_last_search=true) -> search_refinement
"What do you mean by code smells?" (paper loaded, discussing its content) -> qa
"What does your sources tell?" (paper loaded, referring to what's already found) -> qa
"Has anyone benchmarked Mojo against similar languages for GPU acceleration?" (paper loaded or not) -> discovery
"Has anyone else trained a model this way?" (paper loaded, asking about OTHER work) -> discovery
"What did this paper find about GPU performance?" (paper loaded, asking about THIS document) -> qa
"new" -> (handled before you see it, ignore)

For "discovery" or "lookup" (except when picking from the discovery list),
extract "topic": what to search for. If the message doesn't restate the
topic explicitly, infer it from the conversation or the loaded paper's
subject. If the message directly names an arXiv ID, put it in "paper_id"
instead (forces "lookup"). If picking from the discovery list, set mode
"lookup", "selection" to its 1-based index, and leave topic/paper_id null.
For "search_refinement", leave topic/paper_id/selection null -- the caller
reuses the ORIGINAL search, not this message's wording.

Also produce "standalone_query": the message rewritten as self-contained,
with pronouns ("they", "it", "what do they do") resolved using history.
Used only for "qa"; for other modes just echo the message. If genuinely too
ambiguous to resolve, keep the ambiguous wording rather than guessing.

Respond with ONLY JSON:
{{"mode": "discovery"|"lookup"|"qa"|"search_refinement", "topic": "<topic or null>", "paper_id": "<id or null>", "selection": <int or null>, "standalone_query": "..."}}"""


def route_top_level(
    message: str,
    recent_messages: list[str],
    active_paper_title: str | None,
    discovery_list: list[dict] | None,
    has_last_search: bool = False,
) -> TopLevelIntent:
    """Classify a chat message into discovery/lookup/qa/search_refinement
    before any retrieval runs. A cheap regex check for an arXiv ID directly
    in `message` happens in the caller before this is invoked as a fast
    path; this handles the harder cases: topic inference from context,
    discovery-list selection, refinement-vs-new-search, and pronoun
    resolution for qa follow-ups.
    """
    history_text = "\n".join(recent_messages) or "(no prior messages)"
    if discovery_list:
        list_text = "\n".join(
            f"{i}. {p['title']} ({p['arxiv_id']})" for i, p in enumerate(discovery_list, 1)
        )
    else:
        list_text = "(none)"

    response = _client.chat(
        model=CHAT_MODEL,
        temperature=CHAT_TEMPERATURE,
        messages=[
            {
                "role": "user",
                "content": _ROUTER_PROMPT.format(
                    has_last_search=has_last_search,
                    active_paper=active_paper_title or "(none loaded)",
                    discovery_list=list_text,
                    history=history_text,
                    message=message,
                ),
            }
        ],
        response_format={"type": "json_object"},
    )
    text = response.message.content[0].text
    try:
        parsed = TopLevelIntent.model_validate(json.loads(text))
    except (json.JSONDecodeError, ValidationError):
        # Fail toward the safest option: qa on whatever's loaded, or a plain
        # lookup on the raw message if nothing is loaded yet.
        fallback_mode = "qa" if active_paper_title else "lookup"
        return TopLevelIntent(mode=fallback_mode, topic=message, standalone_query=message)

    if parsed.mode == "qa" and not active_paper_title:
        # Can't answer from a paper that isn't loaded -- treat as a lookup instead.
        return TopLevelIntent(mode="lookup", topic=parsed.standalone_query or message, standalone_query=message)

    if parsed.mode == "search_refinement" and not has_last_search:
        # Nothing to refine -- don't hallucinate a refinement of nothing.
        return TopLevelIntent(mode="discovery", topic=parsed.topic or message, standalone_query=message)

    return parsed


_DISCOVERY_FILTERS_PROMPT = """Extract search filters from this research
discovery request. Today's date is {today}.

Identify:
- topic: the core subject, with author names and date phrasing stripped
  out (e.g. "Recent developments in XYZ" -> topic "XYZ")
- authors: author surname(s) explicitly named, if any, else empty list
- date_from / date_to: an inclusive ISO date range (YYYY-MM-DD), resolved
  relative to today's date, or null if unbounded on that side. Examples:
  "papers published in the past year" -> date_from = one year before
  today, date_to = today. "before 2005" -> date_from = null, date_to =
  "2004-12-31". "after 2020" -> date_from = "2021-01-01", date_to = null.
  "from 01/01/2026 to 03/04/2026" -> date_from = "2026-01-01", date_to =
  "2026-04-03" (assume MM/DD/YYYY when the format is ambiguous).
  "recent developments" with no explicit range -> date_from = 2 years
  before today, date_to = today. If there's no date constraint implied at
  all, both null.

Respond with ONLY JSON:
{{"topic": "...", "authors": ["..."], "date_from": "<date or null>", "date_to": "<date or null>"}}

Request: {query}"""


def extract_discovery_filters(query: str) -> DiscoveryFilters:
    """Pull explicit author/date constraints out of a discovery request so
    they can be applied as hard arXiv filters (search_with_filters),
    distinct from the topic itself which still goes through the normal
    loose multi-formulation search."""
    response = _client.chat(
        model=CHAT_MODEL,
        temperature=CHAT_TEMPERATURE,
        messages=[
            {
                "role": "user",
                "content": _DISCOVERY_FILTERS_PROMPT.format(today=date.today().isoformat(), query=query),
            }
        ],
        response_format={"type": "json_object"},
    )
    text = response.message.content[0].text
    try:
        return DiscoveryFilters.model_validate(json.loads(text))
    except (json.JSONDecodeError, ValidationError):
        return DiscoveryFilters(topic=query)


_DISCOVERY_EXPAND_PROMPT = """Generate 4-6 independent, broad search
formulations for this research topic, each a short (2-4 word) phrase --
different ways of reaching the same literature, not paraphrases of one
long sentence.

Rules:
- Preserve the core research concept.
- Do NOT include temporal constraints (no "recent", "latest", "2024",
  "past year", specific years, etc.) -- date filtering is handled
  separately and deterministically; injecting date language into a search
  phrase only restricts recall for no benefit.
- Do NOT include author constraints.
- Do NOT invent specific papers, methods, datasets, or terminology not
  implied by the topic itself.
- Do NOT combine every concept into one query -- each formulation must be
  independently searchable, not one long restrictive phrase.
- Vary terminology, acronyms, and abstraction level so different
  formulations reach the same literature via different words.
- Output plain phrases only, no surrounding quote marks.

Output ONLY the phrases, one per line, no numbering, no quote marks.

Topic: {topic}"""


def generate_discovery_queries(topic: str) -> list[str]:
    """Expand a topic into several independent, temporal/author-free
    search formulations for broad recall."""
    response = _client.chat(
        model=CHAT_MODEL,
        temperature=CHAT_TEMPERATURE,
        messages=[{"role": "user", "content": _DISCOVERY_EXPAND_PROMPT.format(topic=topic)}],
    )
    text = response.message.content[0].text.strip()
    phrases = [line.strip("-*0123456789. \"'").strip() for line in text.splitlines() if line.strip()]
    phrases = [p for p in phrases if p]
    return phrases or [topic]


_RELATION_PROMPT = """For each paper below, write ONE short phrase (5-10
words) describing its relation to the topic "{topic}" -- e.g. "directly
studies the topic", "uses it in experiments", "evaluates its performance".

Papers:
{papers}

Respond with ONLY JSON: {{"relations": ["...", "...", ...]}}, same order and
length as the papers."""


def describe_relations(topic: str, papers: list[dict]) -> list[str]:
    """One short relation blurb per paper, in one batched call."""
    papers_text = "\n".join(
        f"{i}. {p['title']}: {p['summary'][:300]}" for i, p in enumerate(papers, 1)
    )
    response = _client.chat(
        model=CHAT_MODEL,
        temperature=CHAT_TEMPERATURE,
        messages=[
            {"role": "user", "content": _RELATION_PROMPT.format(topic=topic, papers=papers_text)}
        ],
        response_format={"type": "json_object"},
    )
    text = response.message.content[0].text.strip()
    try:
        relations = json.loads(text).get("relations", [])
        if len(relations) == len(papers):
            return [str(r) for r in relations]
    except json.JSONDecodeError:
        pass
    return ["relevant to the topic"] * len(papers)


_DISCOVERY_PROMPT = """A research query used terminology that isn't yet in
our research vocabulary memory. Resolve it.

Query: {query}
Unfamiliar term to resolve: {term}

Identify:
- canonical_name: the full/formal name of the concept (e.g. "Joint Embedding
  Predictive Architecture" for "JEPA")
- aliases: other short names/acronyms for the same concept
- related_terms: closely related concepts/terminology useful for arXiv
  search recall (e.g. "world models", "predictive representation learning")
- domains: 1-3 broad fields this belongs to (e.g. "machine learning")
- confidence: your honest confidence (0.0-1.0) that this is a real,
  established term and not a guess -- use a LOW value if unsure.

Respond with ONLY JSON:
{{"canonical_name": "...", "aliases": ["..."], "related_terms": ["..."], "domains": ["..."], "confidence": 0.0}}"""


def discover_terminology(term: str, query: str) -> ConceptDiscovery | None:
    """Resolve one unfamiliar term via LLM reasoning. Returns None on
    malformed output -- the caller (research_discovery.py) still requires
    real arXiv evidence before trusting/persisting this, so a parse failure
    here just means "no resolution attempted", not a data-quality problem.
    """
    response = _client.chat(
        model=CHAT_MODEL,
        temperature=CHAT_TEMPERATURE,
        messages=[{"role": "user", "content": _DISCOVERY_PROMPT.format(term=term, query=query)}],
        response_format={"type": "json_object"},
    )
    text = response.message.content[0].text
    try:
        return ConceptDiscovery.model_validate(json.loads(text))
    except (json.JSONDecodeError, ValidationError):
        return None


_RECOVERY_PROMPT = """arXiv search for this query returned too few or too
weak results:

Original query: {query}
{concept_context}

Generate up to 6 SHORT, LOOSE arXiv search phrases (2-5 words each) to
maximize recall. Each phrase should combine the core concept with at most
ONE additional angle/modifier -- do NOT combine many concepts into one long
quoted phrase (that's overly restrictive, and is exactly what already
failed). Vary terminology, acronyms, and related angles across the phrases.

Example style: for "graph neural networks adversarial attacks cybersecurity"
-> "graph neural network adversarial", "GNN adversarial attack", "graph
neural network poisoning", "GNN intrusion detection".

Output ONLY the phrases, one per line, no numbering."""


def generate_recovery_queries(query: str, concept: dict | None = None) -> list[str]:
    """Broadened, loose search formulations for the arXiv-recovery retry
    path (see nodes/research_recovery.py) -- only called on retrieval
    failure, never on the normal path, since it's an extra LLM call.
    """
    concept_context = ""
    if concept:
        known = ", ".join(
            dict.fromkeys([concept["canonical_name"], *concept["aliases"], *concept["related_terms"]])
        )
        concept_context = f"Known terminology for this concept: {known}"

    response = _client.chat(
        model=CHAT_MODEL,
        temperature=CHAT_TEMPERATURE,
        messages=[
            {
                "role": "user",
                "content": _RECOVERY_PROMPT.format(query=query, concept_context=concept_context),
            }
        ],
    )
    text = response.message.content[0].text.strip()
    phrases = [line.lstrip("-* ").strip() for line in text.splitlines() if line.strip()]
    return phrases[:6] or [query]
