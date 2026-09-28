# Evaluation Results

Live run against the real pipeline (real arXiv fetch, real Cohere calls, real Chroma retrieval — no mocks) after a full cache clear (`data/chroma`, `data/briefings`, `data/papers`, `data/research_memory.sqlite` all deleted first, so the paper was fetched, parsed, chunked, and embedded from scratch). This run uses the fixed scorer (`eval/scoring.py`) — see "Scorer fixes" below for what changed and why.

- **Paper:** `1706.03762` (Attention Is All You Need)
- **Run:** `2026-09-28T07:46:03Z`
- **Questions:** 24 single questions + 2 conversations (5 turns) = 29 scored items
- **Raw data:** [`eval/results/20260928T074603Z.json`](eval/results/20260928T074603Z.json) (full answer text, sources, router output) · [`eval/results/latest.csv`](eval/results/latest.csv) (flat scoring rows) · [`eval/results/latest.md`](eval/results/latest.md) (machine-generated tables, same numbers as below)

## Summary

| Tier | n | pass_rate | refusal_accuracy | false_refusal_rate | hallucination_rate | mean_keyword_score | section_hit_rate | evidence_above_threshold_rate | mean_latency_s |
|---|---|---|---|---|---|---|---|---|---|
| CORE (easy+medium) | 16 | 1.000 | 1.000 | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 4.225 |
| HARD (stress tests) | 13 | 1.000 | 0.846 | 0.200 | 0.000 | 1.000 | 1.000 | 1.000 | 4.269 |
| OVERALL | 29 | 1.000 | 0.931 | 0.083 | 0.000 | 1.000 | 1.000 | 1.000 | 4.245 |

`routing_ok` (multi-turn: router mode == "qa" on follow-up turns): **1.000**

`evidence_above_threshold_rate` measures the pipeline's own relevance-floor check on retrieved evidence, not independently-verified factual grounding of the generated text — see Design Decisions & Tradeoffs in the README.

### Per-stress-tag pass rate (hard tier only)

| tag | pass rate |
|---|---|
| table | 1.000 (2/2 straightforward + 1 partial-but-scored-as-pass, see below) |
| multi-hop | 1.000 (2/2) |
| false-premise | 1.000 (2/2) |
| near-miss-oos | 1.000 (2/2) |
| numeric-reasoning | 1.000 (2/2) |
| multi-turn | 1.000 (3/3) |

## Full results

| id | difficulty | stress | refused | pass | kw | sec | conf | secs |
|---|---|---|---|---|---|---|---|---|
| e-layers | easy |  | False | True | 1.00 | y | high | 3.1 |
| e-heads | easy |  | False | True | 1.00 | y | high | 2.9 |
| e-optimizer | easy |  | False | True | 1.00 | y | high | 2.9 |
| e-bleu-ende | easy |  | False | True | 1.00 | y | high | 3.3 |
| e-dropout | easy |  | False | True | 1.00 | y | high | 5.3 |
| e-oos-imagenet | easy |  | True | True | - | - | low | 5.3 |
| conv-easy-optim#0 | easy |  | False | True | 1.00 | - | high | 2.6 |
| conv-easy-optim#1 | easy |  | False | True | 1.00 | - | high | 3.3 |
| m-schedule | medium |  | False | True | 1.00 | y | medium | 4.5 |
| m-train-time | medium |  | False | True | 1.00 | y | high | 5.1 |
| m-scaling | medium |  | False | True | 1.00 | y | high | 3.9 |
| m-enc-dec-attn | medium |  | False | True | 1.00 | y | high | 3.0 |
| m-beam | medium |  | False | True | 1.00 | y | high | 4.2 |
| m-pos-enc | medium |  | False | True | 1.00 | y | high | 5.3 |
| m-data | medium |  | False | True | 1.00 | y | high | 8.7 |
| m-oos-glue | medium |  | True | True | - | - | low | 4.2 |
| h-table-heads | hard | table | False | True | 1.00 | y | high | 3.4 |
| h-table-params | hard | table | **True** | True | 0.50 | y | high | 3.8 |
| h-table-flops | hard | table,numeric-reasoning | False | True | 1.00 | y | high | 2.7 |
| h-when-faster | hard | multi-hop | False | True | 1.00 | y | high | 6.6 |
| h-margin | hard | numeric-reasoning | False | True | 1.00 | y | high | 4.8 |
| h-parsing-setup | hard | multi-hop | False | True | 1.00 | y | high | 5.9 |
| h-false-premise-learned | hard | false-premise | True | True | 1.00 | y | high | 4.0 |
| h-false-premise-gpt4 | hard | false-premise | True | True | - | - | medium | 6.4 |
| h-near-miss-enro | hard | near-miss-oos | True | True | - | - | high | 4.3 |
| h-near-miss-memory | hard | near-miss-oos | True | True | - | - | high | 3.2 |
| conv-hard-bigmodel#0 | hard | multi-turn | False | True | 1.00 | - | medium | 2.9 |
| conv-hard-bigmodel#1 | hard | multi-turn | False | True | 1.00 | - | medium | 3.4 |
| conv-hard-bigmodel#2 | hard | multi-turn | False | True | 1.00 | - | high | 4.1 |

All 29 items pass. Zero `forbidden_hit`s anywhere in this run.

## Scorer fixes (from the previous run's review)

The prior run flagged 3 hard-tier "failures" that, on reading the actual answer text, were really the scoring harness misreading correct answers — not pipeline mistakes. Two specific, narrow fixes went into `eval/scoring.py`:

1. **`forbidden_hit` no longer triggers inside the phrase's own negation.** It used to be a flat substring search across the whole answer, so `"...does not state that they chose learned positional embeddings..."` tripped the forbidden phrase `"chose learned positional"` even though the sentence is *denying* it. The fix splits text into clauses (on sentence boundaries and `,`/`though`/`but`/`however`/`although`/`yet`) and only counts a hit if the clause containing the forbidden phrase has no negation word (`not`, `never`, `no`, `without`, or an `n't` contraction) in it.
2. **`is_refusal`'s patterns now cover `do`/`does`/`did` (not just `does`/`doesn't`), and allow up to two words between the negation and the verb.** The prior regex required `does not specify` with zero gap; the real answer read `"they do not specify the GPU memory capacity"` — plural subject, and it was never going to match `does(?:n't| not)`.

Both are covered by new regression tests in `tests/test_eval_scoring.py` (`test_forbidden_hit_not_triggered_by_the_phrase_inside_its_own_negation`, `test_is_refusal_true_for_plural_subject_do_not_phrasing`), reproducing the exact real-run text that exposed each bug.

## What the fix changed, and one new thing it surfaced

`h-near-miss-memory` (a genuinely unanswerable near-miss item) and `h-false-premise-learned` (a false-premise item, where declining the premise counts as a pass by design) both now correctly register as refusals — that's the fix working as intended.

`h-table-params` also flipped to `refused=True`, and it's worth being honest about why: its answer states the base model's 65M parameters correctly, then says in a second sentence that the big model's parameter count "is not explicitly stated in the provided excerpts." That second sentence is a genuine, correct refusal phrase — but `is_refusal` only looks at whether *any* refusal-shaped language appears in the first two sentences, not whether the whole answer is a refusal. So a **partial** answer (`keyword_score=0.50`, only found 65M, not 213M) gets `refused=True`, and the scorer's pass rule (`refused OR keyword_score==1.0`) — written for false-premise items, where declining is the correct response — ends up crediting this as a pass too. It's also excluded from `mean_keyword_score` (which only averages over non-refused answerable items), which is why that metric reads a clean 1.000 despite this 0.50 sitting in the raw data.

This is not one of the two bugs described for this round, so it wasn't fixed here — flagging it instead of patching it quietly. The real pipeline behavior is: a correct partial answer (right on the base model, silent on the big model, matching the documented pypdf table-extraction limitation) that a binary refused/answered scorer can't cleanly represent. A future fix would need to distinguish "fully refused" from "partially refused" rather than checking the whole answer for any refusal-shaped sentence, and probably restrict the "refusal-counts-as-pass" rule to items actually tagged `false-premise` rather than every answerable item.

## Analysis

**Core (easy + medium) is clean: 16/16 pass, everything grounded, every retrieval hit the expected section.** Unchanged from the prior run — the scorer fixes only affected hard-tier refusal/forbidden-phrase edge cases, and core never touched those paths.

**Hard tier: every item now scores as a pass**, but as explained above, one of those (`h-table-params`) is a partial answer that the binary refused/answered scorer credits generously rather than a fully correct one. Reading the raw data rather than the summary number: 12 of 13 hard items are unambiguously correct (full keyword match or a genuinely clean refusal/correction), and 1 (`h-table-params`) got half the table lookup right and half wrong — consistent with the already-documented pypdf table-extraction limitation, not a new problem.

**Zero hallucinations, zero forbidden-phrase hits.** `false_refusal_rate` (0.083 overall, 0.200 hard) is non-zero for the first time, but by design: it's counting `h-false-premise-learned` (correctly declining a false premise, which the item spec explicitly allows) and `h-table-params` (the partial-answer case above) as "refused when it was answerable" — neither is the failure mode `false_refusal_rate` exists to catch (the pipeline confidently refusing something it could actually answer in full).

**Routing remains solid:** all 3 multi-turn follow-ups correctly routed to `qa` and resolved to the right standalone query.

**Net read:** the pipeline's actual behavior across this run is strong and matches the prior run's real (non-scoring-artifact) results — the only substantive miss anywhere is the same one both runs found: `h-table-params`'s big-model parameter count, attributable to table-extraction, not retrieval or generation. The scorer is more accurate than before but still imperfect, and that imperfection is documented above rather than smoothed over.
