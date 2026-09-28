"""Live evaluation harness: runs the REAL pipeline (arXiv, Cohere, Chroma) --
no mocks -- against eval/questions.json and reports honest metrics.

Usage: python -m eval.run_eval [--questions PATH] [--only id,id,...]
       [--difficulty easy,medium,hard] [--delay 3.0] [--repeat-check ID]

Costs real Cohere calls (roughly one route + one plan + one-to-several
rerank/embed + one answer per question, more for decomposed/refined
questions) -- --delay exists because trial keys are rate-limited.
"""

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import cohere
from colorama import Fore, Style, init as colorama_init

from eval.scoring import STRESS_TAGS, score_item, summarize
from re_search_it.graph import build_retrieval_graph
from re_search_it.qa_graph import build_qa_graph
from re_search_it.tools.cohere_client import route_top_level

colorama_init(autoreset=True)

_HEADER = Fore.CYAN + Style.BRIGHT
_DIM = Fore.WHITE + Style.DIM
_GREEN = Fore.GREEN
_RED = Fore.RED + Style.BRIGHT
_WARN = Fore.YELLOW

RESULTS_DIR = Path(__file__).parent / "results"
RETRY_BACKOFFS = [5, 10, 20]


def _c(color: str, text: str) -> str:
    return f"{color}{text}{Style.RESET_ALL}"


def _is_rate_limit(exc: Exception) -> bool:
    if hasattr(cohere, "TooManyRequestsError") and isinstance(exc, cohere.TooManyRequestsError):
        return True
    return "429" in str(exc)


def _with_retry(fn, *, delay: float):
    """Run fn() once, retrying with exponential backoff on rate-limit errors."""
    for attempt, backoff in enumerate([0, *RETRY_BACKOFFS]):
        if backoff:
            print(_c(_WARN, f"  ! rate limited, retrying in {backoff}s..."))
            time.sleep(backoff)
        try:
            return fn()
        except Exception as exc:
            if attempt < len(RETRY_BACKOFFS) and _is_rate_limit(exc):
                continue
            raise


def _load_questions(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _flatten_items(paper_block: dict, only: set[str] | None, difficulties: set[str] | None) -> list[dict]:
    """Single questions plus each conversation turn, each as its own scorable item."""
    items = []
    for q in paper_block.get("questions", []):
        items.append({**q, "kind": "question"})
    for conv in paper_block.get("conversations", []):
        for i, turn in enumerate(conv["turns"]):
            items.append({
                **turn,
                "id": f"{conv['id']}#{i}",
                "difficulty": conv["difficulty"],
                "stress": conv.get("stress", []),
                "kind": "turn",
                "conv_id": conv["id"],
                "turn_index": i,
            })

    if only:
        items = [it for it in items if it["id"] in only or it.get("conv_id") in only]
    if difficulties:
        items = [it for it in items if it["difficulty"] in difficulties]
    return items


def _run_paper(paper_id: str) -> dict | None:
    retrieval_graph = build_retrieval_graph()
    result = retrieval_graph.invoke({"query": paper_id, "conversation_history": []})
    if result.get("error"):
        print(_c(_RED, f"! Failed to load {paper_id}: {result['error']}"))
        return None

    sections = result.get("parsed_sections", {})
    print(_c(_DIM, f"Parsed sections ({len(sections)}): {', '.join(sections.keys())}"))
    print(_c(_DIM, f"Chunk count: {result.get('chunk_count', 0)}  structure_degraded: {result.get('structure_degraded')}"))
    return result


def _run_qa(qa_graph, paper_state: dict, question: str, history: list[dict]) -> tuple[dict, float]:
    start = time.perf_counter()
    result = qa_graph.invoke({**paper_state, "question": question, "conversation_history": history})
    elapsed = time.perf_counter() - start
    return result, elapsed


def _run_single_question(qa_graph, paper_state: dict, item: dict, delay: float) -> dict:
    def _invoke():
        return _run_qa(qa_graph, paper_state, item["q"], [])

    try:
        result, elapsed = _with_retry(_invoke, delay=delay)
        row = score_item(item, result["answer"])
        row.update({"latency_s": elapsed, "answer_text": result["answer"]["text"], "sources": result["answer"]["sources"]})
    except Exception as exc:
        row = {
            "id": item["id"], "difficulty": item["difficulty"], "stress": item.get("stress", []),
            "answerable": item["answerable"], "passed": False, "error": str(exc),
            "refused": False, "forbidden_hit": False, "keyword_score": None, "section_hit": None,
            "grounded": None, "confidence": None, "latency_s": None, "answer_text": "", "sources": [],
        }
    time.sleep(delay)
    return row


def _run_conversation(qa_graph, paper_state: dict, active_paper_title: str, conv: dict, delay: float) -> list[dict]:
    history: list[dict] = []
    raw_log: list[str] = []
    rows = []
    for i, turn in enumerate(conv["turns"]):
        raw_log.append(turn["q"])
        if i == 0:
            standalone_query, mode = turn["q"], "qa"
        else:
            def _route():
                return route_top_level(turn["q"], raw_log[:-1], active_paper_title, [])

            try:
                intent = _with_retry(_route, delay=delay)
                standalone_query, mode = intent.standalone_query, intent.mode
            except Exception as exc:
                item = {**turn, "id": f"{conv['id']}#{i}", "difficulty": conv["difficulty"], "stress": conv.get("stress", [])}
                rows.append({
                    "id": item["id"], "difficulty": item["difficulty"], "stress": item["stress"],
                    "answerable": item["answerable"], "passed": False, "error": f"routing failed: {exc}",
                    "refused": False, "forbidden_hit": False, "keyword_score": None, "section_hit": None,
                    "grounded": None, "confidence": None, "latency_s": None, "answer_text": "", "sources": [],
                    "routing_mode": None, "standalone_query": None,
                })
                continue

        item = {**turn, "id": f"{conv['id']}#{i}", "difficulty": conv["difficulty"], "stress": conv.get("stress", [])}

        def _invoke():
            return _run_qa(qa_graph, paper_state, standalone_query, history)

        try:
            result, elapsed = _with_retry(_invoke, delay=delay)
            history = result["conversation_history"]
            row = score_item(item, result["answer"])
            row.update({
                "latency_s": elapsed, "answer_text": result["answer"]["text"], "sources": result["answer"]["sources"],
                "routing_mode": mode, "standalone_query": standalone_query,
            })
        except Exception as exc:
            row = {
                "id": item["id"], "difficulty": item["difficulty"], "stress": item["stress"],
                "answerable": item["answerable"], "passed": False, "error": str(exc),
                "refused": False, "forbidden_hit": False, "keyword_score": None, "section_hit": None,
                "grounded": None, "confidence": None, "latency_s": None, "answer_text": "", "sources": [],
                "routing_mode": mode, "standalone_query": standalone_query,
            }
        rows.append(row)
        time.sleep(delay)
    return rows


def _print_table(rows: list[dict], title: str) -> None:
    print(f"\n{_c(_HEADER, title)}")
    print(f"{'id':<26}{'stress':<28}{'refused':<9}{'pass':<6}{'kw':<6}{'sec':<6}{'conf':<8}{'secs':<6}")
    for r in rows:
        pass_color = _GREEN if r["passed"] else _RED
        kw = "-" if r["keyword_score"] is None else f"{r['keyword_score']:.2f}"
        sec = "-" if r["section_hit"] is None else ("y" if r["section_hit"] else "n")
        secs = "-" if r["latency_s"] is None else f"{r['latency_s']:.1f}"
        stress = ",".join(r.get("stress", []))
        line = (
            f"{r['id']:<26}{stress:<28}{str(r['refused']):<9}{str(r['passed']):<6}"
            f"{kw:<6}{sec:<6}{str(r['confidence']):<8}{secs:<6}"
        )
        print(_c(pass_color, line) if not r["passed"] else line)
        if not r["passed"] and r["difficulty"] == "hard":
            snippet = (r.get("answer_text") or r.get("error") or "")[:160]
            print(_c(_DIM, f"    -> {snippet}"))


def _print_metrics_block(title: str, metrics: dict) -> None:
    if not metrics:
        return
    print(f"\n{_c(_HEADER, title)}")
    for key in ("n", "pass_rate", "refusal_accuracy", "false_refusal_rate", "hallucination_rate",
                "mean_keyword_score", "section_hit_rate", "grounded_rate", "mean_latency_s"):
        if key in metrics and metrics[key] is not None:
            val = metrics[key]
            print(f"  {key}: {val:.3f}" if isinstance(val, float) else f"  {key}: {val}")


def _write_reports(all_rows: list[dict], metrics: dict, routing_rows: list[dict]) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    (RESULTS_DIR / f"{ts}.json").write_text(
        json.dumps({"results": all_rows, "metrics": metrics}, indent=2), encoding="utf-8"
    )

    routing_ok = None
    if routing_rows:
        oks = [1.0 if r.get("routing_mode") == "qa" else 0.0 for r in routing_rows]
        routing_ok = sum(oks) / len(oks)

    lines = ["# Latest eval results\n", f"Generated: {ts}\n"]
    for diff in ("easy", "medium", "hard"):
        rows = [r for r in all_rows if r["difficulty"] == diff]
        if not rows:
            continue
        lines.append(f"\n## {diff.upper()}\n")
        lines.append("| id | stress | refused | pass | kw | sec | conf | secs |")
        lines.append("|---|---|---|---|---|---|---|---|")
        for r in rows:
            kw = "-" if r["keyword_score"] is None else f"{r['keyword_score']:.2f}"
            sec = "-" if r["section_hit"] is None else ("y" if r["section_hit"] else "n")
            secs = "-" if r["latency_s"] is None else f"{r['latency_s']:.1f}"
            stress = ",".join(r.get("stress", []))
            lines.append(f"| {r['id']} | {stress} | {r['refused']} | {r['passed']} | {kw} | {sec} | {r['confidence']} | {secs} |")

    lines.append("\n## CORE (easy+medium)\n")
    for k, v in metrics["core"].items():
        lines.append(f"- {k}: {v:.3f}" if isinstance(v, float) else f"- {k}: {v}")

    lines.append("\n## STRESS TESTS (hard): expected to expose known limitations\n")
    for k, v in metrics["by_difficulty"]["hard"].items():
        lines.append(f"- {k}: {v:.3f}" if isinstance(v, float) else f"- {k}: {v}")
    lines.append("\n### Per-stress-tag pass rate\n")
    for tag in STRESS_TAGS:
        v = metrics["stress_by_tag"].get(tag)
        lines.append(f"- {tag}: {v:.3f}" if v is not None else f"- {tag}: (no items)")

    lines.append("\n## OVERALL\n")
    for k, v in metrics["overall"].items():
        lines.append(f"- {k}: {v:.3f}" if isinstance(v, float) else f"- {k}: {v}")

    if routing_ok is not None:
        lines.append(f"\nrouting_ok (multi-turn mode == qa): {routing_ok:.3f}\n")

    (RESULTS_DIR / "latest.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", default=str(Path(__file__).parent / "questions.json"))
    parser.add_argument("--only", default=None, help="comma-separated question/conversation ids")
    parser.add_argument("--difficulty", default=None, help="comma-separated: easy,medium,hard")
    parser.add_argument("--delay", type=float, default=3.0)
    parser.add_argument("--repeat-check", default=None, help="question id to run twice and compare")
    args = parser.parse_args()

    only = set(args.only.split(",")) if args.only else None
    difficulties = set(args.difficulty.split(",")) if args.difficulty else None

    data = _load_questions(Path(args.questions))
    qa_graph = build_qa_graph()

    all_rows: list[dict] = []
    routing_rows: list[dict] = []

    for paper_block in data["papers"]:
        paper_id = paper_block["paper"]
        print(_c(_HEADER, f"\n=== {paper_id} ==="))
        paper_state = _run_paper(paper_id)
        if paper_state is None:
            continue
        active_paper_title = paper_state["selected_paper"]["title"]

        items = _flatten_items(paper_block, only, difficulties)
        single_items = [it for it in items if it["kind"] == "question"]
        conv_ids = {it["conv_id"] for it in items if it["kind"] == "turn"}
        convs = [c for c in paper_block.get("conversations", []) if c["id"] in conv_ids]

        for item in single_items:
            print(_c(_DIM, f"  {item['id']}: {item['q']}"))
            all_rows.append(_run_single_question(qa_graph, paper_state, item, args.delay))

        for conv in convs:
            print(_c(_DIM, f"  {conv['id']} ({len(conv['turns'])} turns)"))
            conv_rows = _run_conversation(qa_graph, paper_state, active_paper_title, conv, args.delay)
            all_rows.extend(conv_rows)
            routing_rows.extend([r for r in conv_rows if "routing_mode" in r])

        if args.repeat_check:
            match = next((it for it in single_items if it["id"] == args.repeat_check), None)
            if match:
                print(_c(_DIM, f"\nDeterminism check on {args.repeat_check}..."))
                r1, _ = _with_retry(lambda: _run_qa(qa_graph, paper_state, match["q"], []), delay=args.delay)
                time.sleep(args.delay)
                r2, _ = _with_retry(lambda: _run_qa(qa_graph, paper_state, match["q"], []), delay=args.delay)
                identical = r1["answer"]["text"] == r2["answer"]["text"]
                print(_c(_GREEN if identical else _WARN, f"  identical: {identical}"))

    for diff in ("easy", "medium", "hard"):
        rows = [r for r in all_rows if r["difficulty"] == diff]
        if rows:
            label = f"{diff.upper()} -- STRESS TESTS" if diff == "hard" else diff.upper()
            _print_table(rows, label)

    latencies = {r["id"]: r["latency_s"] for r in all_rows if r["latency_s"] is not None}
    metrics = summarize(all_rows, latencies)

    _print_metrics_block("CORE (easy+medium)", metrics["core"])
    _print_metrics_block("STRESS TESTS (hard): expected to expose known limitations", metrics["by_difficulty"]["hard"])
    print(f"\n{_c(_HEADER, 'Per-stress-tag pass rate')}")
    for tag in STRESS_TAGS:
        v = metrics["stress_by_tag"].get(tag)
        print(f"  {tag}: {v:.3f}" if v is not None else f"  {tag}: (no items)")
    _print_metrics_block("OVERALL", metrics["overall"])

    if routing_rows:
        ok = sum(1.0 for r in routing_rows if r.get("routing_mode") == "qa") / len(routing_rows)
        print(f"\n  routing_ok (multi-turn mode == qa): {ok:.3f}")

    _write_reports(all_rows, metrics, routing_rows)
    print(_c(_DIM, f"\nFull results written to {RESULTS_DIR}/ (latest.md + timestamped JSON)"))


if __name__ == "__main__":
    sys.exit(main())
