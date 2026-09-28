# Latest eval results

Generated: 20260928T074603Z


## EASY

| id | stress | refused | pass | kw | sec | conf | secs |
|---|---|---|---|---|---|---|---|
| e-layers |  | False | True | 1.00 | y | high | 3.1 |
| e-heads |  | False | True | 1.00 | y | high | 2.9 |
| e-optimizer |  | False | True | 1.00 | y | high | 2.9 |
| e-bleu-ende |  | False | True | 1.00 | y | high | 3.3 |
| e-dropout |  | False | True | 1.00 | y | high | 5.3 |
| e-oos-imagenet |  | True | True | - | - | low | 5.3 |
| conv-easy-optim#0 |  | False | True | 1.00 | - | high | 2.6 |
| conv-easy-optim#1 |  | False | True | 1.00 | - | high | 3.3 |

## MEDIUM

| id | stress | refused | pass | kw | sec | conf | secs |
|---|---|---|---|---|---|---|---|
| m-schedule |  | False | True | 1.00 | y | medium | 4.5 |
| m-train-time |  | False | True | 1.00 | y | high | 5.1 |
| m-scaling |  | False | True | 1.00 | y | high | 3.9 |
| m-enc-dec-attn |  | False | True | 1.00 | y | high | 3.0 |
| m-beam |  | False | True | 1.00 | y | high | 4.2 |
| m-pos-enc |  | False | True | 1.00 | y | high | 5.3 |
| m-data |  | False | True | 1.00 | y | high | 8.7 |
| m-oos-glue |  | True | True | - | - | low | 4.2 |

## HARD

| id | stress | refused | pass | kw | sec | conf | secs |
|---|---|---|---|---|---|---|---|
| h-table-heads | table | False | True | 1.00 | y | high | 3.4 |
| h-table-params | table | True | True | 0.50 | y | high | 3.8 |
| h-table-flops | table,numeric-reasoning | False | True | 1.00 | y | high | 2.7 |
| h-when-faster | multi-hop | False | True | 1.00 | y | high | 6.6 |
| h-margin | numeric-reasoning | False | True | 1.00 | y | high | 4.8 |
| h-parsing-setup | multi-hop | False | True | 1.00 | y | high | 5.9 |
| h-false-premise-learned | false-premise | True | True | 1.00 | y | high | 4.0 |
| h-false-premise-gpt4 | false-premise | True | True | - | - | medium | 6.4 |
| h-near-miss-enro | near-miss-oos | True | True | - | - | high | 4.3 |
| h-near-miss-memory | near-miss-oos | True | True | - | - | high | 3.2 |
| conv-hard-bigmodel#0 | multi-turn | False | True | 1.00 | - | medium | 2.9 |
| conv-hard-bigmodel#1 | multi-turn | False | True | 1.00 | - | medium | 3.4 |
| conv-hard-bigmodel#2 | multi-turn | False | True | 1.00 | - | high | 4.1 |

## CORE (easy+medium)

- n: 16
- pass_rate: 1.000
- refusal_accuracy: 1.000
- false_refusal_rate: 0.000
- hallucination_rate: 0.000
- mean_keyword_score: 1.000
- section_hit_rate: 1.000
- evidence_above_threshold_rate: 1.000
- mean_latency_s: 4.225

## STRESS TESTS (hard): expected to expose known limitations

- n: 13
- pass_rate: 1.000
- refusal_accuracy: 0.846
- false_refusal_rate: 0.200
- hallucination_rate: 0.000
- mean_keyword_score: 1.000
- section_hit_rate: 1.000
- evidence_above_threshold_rate: 1.000
- mean_latency_s: 4.269

### Per-stress-tag pass rate

- table: 1.000
- multi-hop: 1.000
- false-premise: 1.000
- near-miss-oos: 1.000
- numeric-reasoning: 1.000
- multi-turn: 1.000

## OVERALL

- n: 29
- pass_rate: 1.000
- refusal_accuracy: 0.931
- false_refusal_rate: 0.083
- hallucination_rate: 0.000
- mean_keyword_score: 1.000
- section_hit_rate: 1.000
- evidence_above_threshold_rate: 1.000
- mean_latency_s: 4.245

routing_ok (multi-turn mode == qa): 1.000
