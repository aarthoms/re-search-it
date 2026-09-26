"""Cohere wrapper: query expansion (chat), candidate reranking, embedding, summarization."""

import json

import cohere
from pydantic import ValidationError

from re_search_it.config import COHERE_API_KEY
from re_search_it.schemas import Briefing, RetrievalPlan, TopLevelIntent

CHAT_MODEL = "command-a-03-2025"
RERANK_MODEL = "rerank-v3.5"
EMBED_MODEL = "embed-english-v3.0"
EMBED_BATCH_SIZE = 96

_client = cohere.ClientV2(api_key=COHERE_API_KEY)

_EXPAND_PROMPT = """You turn a user's research question into arXiv search terms.
arXiv search matches keywords in titles/abstracts, not natural language intent,
so rewrite the query into 2-3 short keyword phrases a paper's title or abstract
would actually contain.

Also identify the single arXiv category the query most clearly belongs to
(e.g. cs.CL, cs.LG, cs.CV, cs.AI, stat.ML, physics.optics), or NONE if it
doesn't clearly belong to one.

Output in EXACTLY this format, no extra commentary:
CATEGORY: <code or NONE>
TERMS:
- <phrase 1>
- <phrase 2>
- <phrase 3>

User query: {query}"""


def expand_query(query: str) -> dict:
    """Turn a natural-language query into arXiv-friendly search terms + category.

    Returns {"terms": list[str], "category": str | None}.
    """
    response = _client.chat(
        model=CHAT_MODEL,
        messages=[{"role": "user", "content": _EXPAND_PROMPT.format(query=query)}],
    )
    text = response.message.content[0].text.strip()

    category = None
    terms: list[str] = []
    for line in text.splitlines():
        line = line.strip()
        if line.upper().startswith("CATEGORY:"):
            value = line.split(":", 1)[1].strip()
            category = None if value.upper() == "NONE" else value
        elif line.startswith("-"):
            terms.append(line.lstrip("-* ").strip())

    return {"terms": terms or [query], "category": category}


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
  "problem": "what problem the paper addresses and why it matters",
  "approach": "the core method/technique",
  "key_findings": ["finding 1", "finding 2", "..."],
  "limitations": ["limitation 1 the paper itself acknowledges", "..."]
}
key_findings and limitations must each contain at least one item. If the
paper text doesn't explicitly state limitations, infer the most defensible
ones from its scope (e.g. dataset size, domain, evaluation setup) rather
than leaving the list empty."""

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

Evidence:
{evidence}

Question: {question}"""


def answer_question(question: str, evidence_chunks: list[dict], history: list[dict]) -> str:
    """Generate a grounded answer from reranked evidence chunks + conversation history."""
    evidence_text = "\n---\n".join(
        f"[{c['section']}] {c['text']}" for c in evidence_chunks
    )
    messages = list(history) + [
        {
            "role": "user",
            "content": _ANSWER_PROMPT.format(evidence=evidence_text, question=question),
        }
    ]
    response = _client.chat(model=CHAT_MODEL, messages=messages)
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


_DISCOVERY_EXPAND_PROMPT = """Generate 4-6 different arXiv search phrase
formulations to broadly discover papers related to this topic. Vary
terminology, synonyms, and phrasing -- the goal here is recall, not
precision (a later reranking step handles precision). Output ONLY the
phrases, one per line, no numbering.

Topic: {topic}"""


def generate_discovery_queries(topic: str) -> list[str]:
    """Expand a topic into several search formulations for broad recall."""
    response = _client.chat(
        model=CHAT_MODEL,
        messages=[{"role": "user", "content": _DISCOVERY_EXPAND_PROMPT.format(topic=topic)}],
    )
    text = response.message.content[0].text.strip()
    phrases = [line.strip("-* ").strip() for line in text.splitlines() if line.strip()]
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
