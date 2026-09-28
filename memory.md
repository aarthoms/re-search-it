# Design Notes

These are the decisions I'd want to explain in a review: what I tried first, what went
wrong, what I changed, and what it cost.

## Finding the right paper

**Loose searches first, precision later.** My first version turned the user's topic into
one arXiv query. `GNN adversarial attacks cybersecurity` returned nothing, even though the
literature exists; one long phrase required every concept to appear together. Now an LLM
writes up to six short phrasings, each run separately, and one reranker scores the combined
pool against the user's original wording. The arXiv category the LLM guesses only ever adds
a second search; it never filters, because a wrong guess used to hide the right paper. Cost:
more API calls. I capped it at six phrasings to keep a lookup under a minute.

**Identity before similarity.** Typing "Attention Is All You Need" loaded *Is Attention All
What You Need?*, which is semantically close and the wrong paper. Finding a specific paper
isn't a similarity problem, so lookups now try, in order: an arXiv ID (in a URL or on its
own), an exact title match, then title-word matches added to the candidate pool, and only
then semantic search. Citations such as `Falcon [Almazrouei et al., 2023]` are resolved
against the loaded paper's own bibliography first, then by an author-and-year arXiv search.

**Relevance isn't answerability.** A paper can be loosely related without being what the
user asked for. Below a reranker score of 0.15 the system refuses and shows the closest
candidates; between 0.15 and 0.3 it proceeds with a warning. I'd rather return "no confident
match" than confidently summarize the wrong paper.

**One recovery attempt, never a loop.** If nothing is found, the system tries once more with
broadened phrasings (and resolves unfamiliar terms first, see below), then gives up.

## Parsing

**Fail visibly.** My demo paper came back as `Parsed sections: preamble, references`:
every heading was missed, so the briefing was built from roughly 6% of the paper and nothing
said so. The parser now also recognises numbered custom headings, merges repeated headings
instead of overwriting them, and flags a split that found no real structure. When that flag
is set, the CLI shows a warning and the summarizer samples the start, middle and end of the
paper instead of just the beginning. This is still heuristic; arXiv's HTML version is the
real fix.

**Chunks stay inside sections.** Chunks are 200 words with a 40-word overlap and never cross
a section boundary, so every piece of evidence has a location I can cite. The reference list
is excluded from both search and summaries; it was taking up to half of the summary input.

## The briefing

It's a validated schema, so fields can't be skipped. Two rules came from bad output:
limitations must come from the paper, or be marked `(inferred)`, because the model filled a
required list with invented limitations. And suggested follow-up questions must be
answerable from the paper, because "How does Mojo compare to Julia?" just produced
"not in the paper".

## Answering questions

**Plan, then retrieve.** A planner decides between a single search and 2–4 sub-questions;
decomposition costs an extra call, so simple questions skip it. Section hints are soft:
hinted and whole-paper results are merged and reranked, because my section labels aren't
always right. If evidence is weak, the system refines the query once, and still reranks
against the original question so it doesn't drift.

**Grounding in code, not only in the prompt.** When I asked about the Mojo paper's LLM
sentiment benchmark, the answer attributed a paper-wide 20–180x speedup to that workload and
presented a projected number as if it had been measured. The answer prompt now requires tying
a number only to the experiment its excerpt names, and saying whether it was measured,
projected or cited. Below the evidence threshold, a disclaimer is added in code and the turn
is kept out of chat history, so a weak answer can't anchor the next one.

## Routing

Each message is classified as discovery (many papers), lookup (one paper), a question about
the loaded paper, or "show me more". Without this, "find papers on X" and "what does this
paper say about X" were handled the same way. Discovery only lists and ranks papers; a PDF
is downloaded and indexed only when the user picks one.

## Unfamiliar terminology

Searches for acronyms like JEPA found nothing. Now, on a miss, the LLM proposes the full name
and aliases, and I only save them (in a small SQLite table) if arXiv actually returns a paper
for that name. It's a weak check, but the model can't make a term "true" just by stating it.
I chose this over web search to keep arXiv as the only source of evidence.

## Stack and state

LangGraph, with two graphs: one to find and index a paper, one to answer questions. They have
different state and different failure modes. Session state lives in memory; embeddings are
stored per paper in Chroma so a paper is only indexed once; SQLite holds only the terminology
table. I chose Cohere because one free key gives embeddings, a strong reranker and chat. The
cost is that reviewers need a trial key, which is rate-limited.

## Known weaknesses
- PDF parsing is heuristic: tables come out flattened and unusual headings can be missed (flagged, not silent).
- No page numbers in citations.
- Nothing checks an answer's claims against its evidence after it's written.
- The evaluation set is small and uses keyword matching.
