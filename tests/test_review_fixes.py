"""Regression tests for the correctness bugs found in the code review."""

from unittest.mock import patch

from re_search_it.nodes.answer import answer
from re_search_it.nodes.summarize import MAX_CHARS_PER_SECTION, _build_sections_text
from re_search_it.tools.arxiv_client import is_arxiv_id, find_arxiv_id
from re_search_it.tools.pdf_parser import split_sections


# Bug: a repeated heading (e.g. "Results" in the main text and again in an
# appendix) overwrote the earlier occurrence instead of merging both.
def test_repeated_heading_merges_instead_of_overwriting():
    text = "Results\nFirst results text.\nDiscussion\nSome discussion.\nResults\nAppendix results text."
    sections = split_sections(text)
    assert "First results text." in sections["results"]
    assert "Appendix results text." in sections["results"]


# Bug: "Abstract—We propose..." (heading + body on one line) was never
# detected -- the header pattern required the WHOLE line to be the heading.
def test_inline_abstract_is_detected():
    text = "Some Paper Title\nAbstract—We propose a new method for X. It works well.\nIntroduction\nBody text."
    sections = split_sections(text)
    assert "abstract" in sections
    assert "We propose a new method for X" in sections["abstract"]


def test_inline_abstract_with_period_separator():
    text = "Abstract. We propose a new method.\nIntroduction\nBody."
    sections = split_sections(text)
    assert "abstract" in sections
    assert "We propose a new method." in sections["abstract"]


# Roman-numeral headings ("I. INTRODUCTION") weren't recognized at all.
def test_roman_numeral_heading_is_recognized():
    text = "I. Introduction\nSome intro text.\nII. Related Work\nSome related work text."
    sections = split_sections(text)
    assert "introduction" in sections
    assert "related work" in sections


# Multi-word compound headings ("Experiments and Results") weren't recognized.
def test_multiword_heading_is_recognized():
    text = "Introduction\nIntro text.\nExperiments and Results\nExperiment details."
    sections = split_sections(text)
    assert "experiments and results" in sections


# Bug: the briefing summarizer filled its whole 12k-char budget from the
# FIRST section in priority order (usually the introduction), so method/
# results/limitations never reached the model at all.
def test_summarize_gives_every_section_a_share_of_the_budget():
    sections = {
        "abstract": "A" * 10000,
        "introduction": "B" * 10000,
        "results": "C" * 10000,
        "limitations": "D" * 10000,
    }
    text = _build_sections_text(sections)
    assert "## abstract" in text
    assert "## introduction" in text
    assert "## results" in text
    assert "## limitations" in text
    # No single section should have consumed more than its per-section cap.
    for marker in ("A", "B", "C", "D"):
        assert text.count(marker) <= MAX_CHARS_PER_SECTION


# Bug: nothing in the code enforced "not in the paper" -- a weak-but-nonzero
# evidence set could still produce a confident-sounding, un-flagged answer.
def test_ungrounded_answer_gets_a_code_level_disclaimer_and_is_not_persisted_to_history():
    state = {
        "question": "What dataset did they use?",
        "retrieved_chunks": [
            {"id": "c1", "text": "some weak evidence", "section": "intro", "relevance_score": 0.05}
        ],
        "evidence_sufficient": False,
        "conversation_history": [],
    }
    with patch("re_search_it.nodes.answer.answer_question", return_value="They used dataset X."):
        result = answer(state)

    assert result["answer"]["grounded"] is False
    assert "Low-confidence" in result["answer"]["text"]
    assert "They used dataset X." in result["answer"]["text"]
    # Ungrounded turns are NOT added to history, so they can't anchor later
    # turns as if they were an established fact.
    assert result["conversation_history"] == []


def test_grounded_answer_has_no_disclaimer_and_is_persisted_to_history():
    state = {
        "question": "What dataset did they use?",
        "retrieved_chunks": [
            {"id": "c1", "text": "strong evidence", "section": "results", "relevance_score": 0.6}
        ],
        "evidence_sufficient": True,
        "conversation_history": [],
    }
    with patch("re_search_it.nodes.answer.answer_question", return_value="They used dataset X."):
        result = answer(state)

    assert result["answer"]["grounded"] is True
    assert result["answer"]["text"] == "They used dataset X."
    assert len(result["conversation_history"]) == 2


# Bug: only new-style IDs (\d{4}\.\d{4,5}) were recognized -- old-style IDs
# like "cs/0112017" failed both typed directly and inside a URL.
def test_old_style_arxiv_id_is_recognized():
    assert is_arxiv_id("cs/0112017") is True
    assert is_arxiv_id("hep-th/9901001") is True
    assert is_arxiv_id("math.CO/0409461") is True


def test_old_style_arxiv_id_found_inside_text_and_url():
    assert find_arxiv_id("what does cs/0112017 say?") == "cs/0112017"
    assert find_arxiv_id("https://arxiv.org/abs/hep-th/9901001") == "hep-th/9901001"


def test_new_style_id_still_works_unaffected():
    assert is_arxiv_id("2301.12345") is True
    assert is_arxiv_id("2301.12345v2") is True
    assert find_arxiv_id("see 2301.12345 for details") == "2301.12345"
