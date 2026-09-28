"""Offline unit tests for eval/scoring.py -- no network, no LLM."""

from eval.scoring import forbidden_hit, is_refusal, keyword_score, score_item, summarize


def test_is_refusal_true_for_explicit_refusal_phrasing():
    answer = {"text": "The paper does not mention ImageNet.", "grounded": True, "sources": [{"section": "results"}]}
    assert is_refusal(answer) is True


def test_is_refusal_true_for_ungrounded_with_no_sources():
    answer = {"text": "It's probably around 90% accuracy.", "grounded": False, "sources": []}
    assert is_refusal(answer) is True


def test_is_refusal_false_for_answer_with_trailing_caveat():
    answer = {
        "text": "They used the Adam optimizer with beta1=0.9, beta2=0.98 (Optimizer). "
                "It works well in practice. Though the paper doesn't discuss X, this detail is well established elsewhere.",
        "grounded": True,
        "sources": [{"section": "optimizer"}],
    }
    assert is_refusal(answer) is False


def test_keyword_score_variant_group_match():
    assert keyword_score("They used 4,000 warmup steps.", [["4000", "4,000"]]) == 1.0


def test_keyword_score_partial_and_zero():
    assert keyword_score("no numbers here", [["4000", "4,000"], ["warmup"]]) == 0.0
    assert keyword_score("warmup phase", [["4000", "4,000"], ["warmup"]]) == 0.5


def test_forbidden_hit_fails_otherwise_correct_refusal():
    text = "The paper doesn't discuss WMT 2016 English-to-Romanian, though it does report 28.4 BLEU elsewhere."
    assert forbidden_hit(text, ["28.4", "41.8", "41.0"]) is True


def test_forbidden_hit_absent():
    assert forbidden_hit("no relevant numbers here", ["28.4"]) is False


def test_false_premise_item_passes_when_it_corrects_the_premise():
    item = {
        "id": "h-false-premise-learned", "difficulty": "hard", "answerable": True,
        "keywords": [["sinusoid"]], "forbidden": ["they chose learned", "chose learned positional"],
    }
    answer = {
        "text": "Actually the final model uses sinusoidal positional encodings, not learned ones.",
        "grounded": True, "sources": [{"section": "positional encoding"}],
    }
    result = score_item(item, answer)
    assert result["passed"] is True


def test_false_premise_item_passes_when_it_refuses_cleanly():
    item = {
        "id": "h-false-premise-learned", "difficulty": "hard", "answerable": True,
        "keywords": [["sinusoid"]], "forbidden": ["they chose learned", "chose learned positional"],
    }
    answer = {"text": "The paper does not state a reason -- it uses sinusoidal, not learned, positional encodings.", "grounded": False, "sources": []}
    result = score_item(item, answer)
    assert result["passed"] is True


def test_score_item_unanswerable_passes_only_on_clean_refusal():
    item = {"id": "e-oos", "difficulty": "easy", "answerable": False}
    refusal = {"text": "The paper does not report ImageNet accuracy.", "grounded": False, "sources": []}
    confident_wrong = {"text": "It achieved 76.5% top-1 accuracy.", "grounded": True, "sources": [{"section": "results"}]}
    assert score_item(item, refusal)["passed"] is True
    assert score_item(item, confident_wrong)["passed"] is False


def test_summarize_excludes_hard_from_core_and_computes_per_tag_pass_rate():
    results = [
        score_item({"id": "e1", "difficulty": "easy", "answerable": True, "keywords": [["x"]]},
                   {"text": "x is here", "grounded": True, "sources": []}),
        score_item({"id": "m1", "difficulty": "medium", "answerable": True, "keywords": [["y"]]},
                   {"text": "no match", "grounded": True, "sources": []}),
        score_item({"id": "h1", "difficulty": "hard", "answerable": True, "keywords": [["z"]], "stress": ["table"]},
                   {"text": "z present", "grounded": True, "sources": []}),
        score_item({"id": "h2", "difficulty": "hard", "answerable": True, "keywords": [["w"]], "stress": ["table"]},
                   {"text": "nothing relevant", "grounded": True, "sources": []}),
    ]
    metrics = summarize(results)

    assert metrics["core"]["n"] == 2
    assert metrics["by_difficulty"]["hard"]["n"] == 2
    assert metrics["stress_by_tag"]["table"] == 0.5
    assert metrics["stress_by_tag"]["multi-hop"] is None
    assert metrics["overall"]["n"] == 4
