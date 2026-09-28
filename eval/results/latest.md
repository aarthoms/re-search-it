# Latest eval results

Generated: 20260928T065446Z


## EASY

| id | stress | refused | pass | kw | sec | conf | secs |
|---|---|---|---|---|---|---|---|
| e-layers |  | False | True | 1.00 | y | high | 3.2 |
| e-heads |  | False | True | 1.00 | y | high | 11.3 |
| e-optimizer |  | False | True | 1.00 | y | high | 2.8 |
| e-bleu-ende |  | False | True | 1.00 | y | high | 3.6 |
| e-dropout |  | False | True | 1.00 | y | high | 3.0 |
| e-oos-imagenet |  | True | True | - | - | low | 7.2 |
| conv-easy-optim#0 |  | False | True | 1.00 | - | high | 5.2 |
| conv-easy-optim#1 |  | False | True | 1.00 | - | high | 4.5 |

## MEDIUM

| id | stress | refused | pass | kw | sec | conf | secs |
|---|---|---|---|---|---|---|---|
| m-schedule |  | False | True | 1.00 | y | medium | 6.2 |
| m-train-time |  | False | True | 1.00 | y | high | 6.2 |
| m-scaling |  | False | True | 1.00 | y | high | 7.8 |
| m-enc-dec-attn |  | False | True | 1.00 | y | high | 15.2 |
| m-beam |  | False | True | 1.00 | y | high | 3.5 |
| m-pos-enc |  | False | True | 1.00 | y | high | 4.5 |
| m-data |  | False | True | 1.00 | y | high | 7.5 |
| m-oos-glue |  | True | True | - | - | low | 4.2 |

## HARD

| id | stress | refused | pass | kw | sec | conf | secs |
|---|---|---|---|---|---|---|---|
| h-table-heads | table | False | True | 1.00 | y | high | 16.6 |
| h-table-params | table | False | False | 0.50 | y | high | 10.6 |
| h-table-flops | table,numeric-reasoning | False | True | 1.00 | y | high | 3.6 |
| h-when-faster | multi-hop | False | True | 1.00 | y | high | 7.6 |
| h-margin | numeric-reasoning | False | True | 1.00 | y | high | 7.4 |
| h-parsing-setup | multi-hop | False | True | 1.00 | y | high | 6.5 |
| h-false-premise-learned | false-premise | False | False | 1.00 | y | high | 6.1 |
| h-false-premise-gpt4 | false-premise | True | True | - | - | medium | 5.8 |
| h-near-miss-enro | near-miss-oos | True | True | - | - | high | 4.0 |
| h-near-miss-memory | near-miss-oos | False | False | - | - | high | 5.4 |
| conv-hard-bigmodel#0 | multi-turn | False | True | 1.00 | - | medium | 3.3 |
| conv-hard-bigmodel#1 | multi-turn | False | True | 1.00 | - | medium | 13.6 |
| conv-hard-bigmodel#2 | multi-turn | False | True | 1.00 | - | high | 4.3 |

## CORE (easy+medium)

- n: 16
- pass_rate: 1.000
- refusal_accuracy: 1.000
- false_refusal_rate: 0.000
- hallucination_rate: 0.000
- mean_keyword_score: 1.000
- section_hit_rate: 1.000
- grounded_rate: 1.000
- mean_latency_s: 6.011

## STRESS TESTS (hard): expected to expose known limitations

- n: 13
- pass_rate: 0.769
- refusal_accuracy: 0.923
- false_refusal_rate: 0.000
- hallucination_rate: 0.154
- mean_keyword_score: 0.950
- section_hit_rate: 1.000
- grounded_rate: 1.000
- mean_latency_s: 7.294

### Per-stress-tag pass rate

- table: 0.667
- multi-hop: 1.000
- false-premise: 0.500
- near-miss-oos: 0.500
- numeric-reasoning: 1.000
- multi-turn: 1.000

## OVERALL

- n: 29
- pass_rate: 0.897
- refusal_accuracy: 0.966
- false_refusal_rate: 0.000
- hallucination_rate: 0.069
- mean_keyword_score: 0.979
- section_hit_rate: 1.000
- grounded_rate: 1.000
- mean_latency_s: 6.586

routing_ok (multi-turn mode == qa): 1.000
