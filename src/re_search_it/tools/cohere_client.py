"""Cohere wrapper: query expansion (chat), candidate reranking, embedding, summarization."""

import json

import cohere
from pydantic import ValidationError

from re_search_it.config import COHERE_API_KEY
from re_search_it.schemas import Briefing, IntentResolution, RetrievalPlan

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


_INTENT_PROMPT = """You route chat messages for a paper-QA assistant. The
currently loaded paper is: {active_paper}

Recent conversation:
{history}

New user message: {message}

Decide:
- "new_paper" if the user names or clearly implies a DIFFERENT paper than the
  one loaded (mentions an arXiv ID, asks to "fetch"/"look up"/"switch to"
  another paper, or references a paper ID mentioned earlier in the
  conversation that was never actually loaded).
- "follow_up" if the user is asking about the currently loaded paper,
  including a brand new question about it (not just a continuation of the
  last answer).

If intent is "new_paper", extract the arXiv ID (format like 2301.12345 or
2301.12345v2) from the message or from the conversation history if the
message itself doesn't contain one (e.g. "can you fetch that" referring back
to an ID mentioned earlier). If you truly cannot find one, use "follow_up"
instead -- never invent an ID.

Also produce "standalone_query": rewrite the message as a self-contained
question with pronouns/vague references ("they", "it", "what do they do")
resolved using the conversation history. If the message is already
self-contained, standalone_query is just the message. If it's genuinely too
ambiguous to resolve confidently (e.g. could mean two different things),
keep the ambiguous wording as-is rather than guessing.

Respond with ONLY JSON:
{{"intent": "new_paper" or "follow_up", "paper_id": "<id or null>", "standalone_query": "..."}}"""


def resolve_intent(
    message: str, recent_messages: list[str], active_paper_title: str | None
) -> IntentResolution:
    """Classify a chat message as switching papers vs. a follow-up, and
    resolve pronouns/referents into a standalone retrieval query.

    A cheap regex check for an arXiv ID directly in `message` happens in the
    caller before this is invoked -- this handles the harder cases: an ID
    mentioned in an earlier turn ("can you fetch that"), and pronoun
    resolution for ambiguous follow-ups ("what do they do?").
    """
    history_text = "\n".join(recent_messages) or "(no prior messages)"
    response = _client.chat(
        model=CHAT_MODEL,
        messages=[
            {
                "role": "user",
                "content": _INTENT_PROMPT.format(
                    active_paper=active_paper_title or "(none loaded)",
                    history=history_text,
                    message=message,
                ),
            }
        ],
        response_format={"type": "json_object"},
    )
    text = response.message.content[0].text
    try:
        parsed = IntentResolution.model_validate(json.loads(text))
    except (json.JSONDecodeError, ValidationError):
        return IntentResolution(intent="follow_up", standalone_query=message)

    if parsed.intent == "new_paper" and not parsed.paper_id:
        # Model claimed a paper switch but couldn't name one -- don't act on
        # a hallucinated intent, fall back to treating it as a question.
        return IntentResolution(intent="follow_up", standalone_query=parsed.standalone_query)

    return parsed
