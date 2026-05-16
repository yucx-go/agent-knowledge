"""Analyze LongMemEval-S failures: dump per-question details for failed cases.

Outputs:
  - failures_top5.json   — questions where gold not in top-5
  - failures_top10.json  — questions where gold not in top-10
  - failure_summary.md   — human-readable analysis
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from agent_knowledge.core.vault import Vault
from agent_knowledge.core.models import Source
from agent_knowledge.search.engine import SearchEngine


def format_session_content(session: list[dict]) -> str:
    parts = []
    for msg in session:
        role = msg.get("role", "unknown")
        content = msg.get("content", "")
        parts.append(f"{role}: {content}")
    return "\n".join(parts)


def evaluate_one(question_data: dict, top_k: int, vault_base: str) -> dict:
    qid = question_data["question_id"]
    qtype = question_data["question_type"]
    question = question_data["question"]
    sessions = question_data["haystack_sessions"]
    session_ids = question_data["haystack_session_ids"]
    gold_ids = set(question_data["answer_session_ids"])

    vault_path = os.path.join(vault_base, f"vault_{qid}")
    try:
        vault = Vault(vault_path)
        vault.init()

        for sid, session in zip(session_ids, sessions):
            content = format_session_content(session)
            source = Source(
                id=sid,
                title=f"Session {sid}",
                content=content,
                source_type="conversation",
            )
            vault.save_source(source)

        engine = SearchEngine(vault, enable_graph=False, enable_reranker=False)
        engine.build_index()

        results = engine.search(question, top_k=top_k)

        hit_rank = None
        result_summary = []
        for rank, r in enumerate(results, 1):
            is_gold = r.id in gold_ids
            if is_gold and hit_rank is None:
                hit_rank = rank
            result_summary.append({
                "rank": rank,
                "id": r.id,
                "is_gold": is_gold,
                "score": r.score,
                "snippet": r.snippet[:200],
            })

        # Get gold session content (first matching gold)
        gold_content = ""
        if gold_ids:
            first_gold = next(iter(gold_ids))
            try:
                idx = session_ids.index(first_gold)
                gold_content = format_session_content(sessions[idx])[:600]
            except ValueError:
                pass

        return {
            "question_id": qid,
            "question_type": qtype,
            "question": question,
            "answer": question_data.get("answer", ""),
            "gold_session_ids": list(gold_ids),
            "gold_session_preview": gold_content,
            "hit_rank": hit_rank,
            "results_top5": result_summary[:5],
            "n_sessions": len(session_ids),
        }
    finally:
        if os.path.exists(vault_path):
            shutil.rmtree(vault_path, ignore_errors=True)


def categorize_failure(case: dict) -> str:
    """Heuristic categorization of failure root cause."""
    question = case["question"].lower()
    qtype = case["question_type"]
    gold_preview = case["gold_session_preview"].lower()
    hit_rank = case["hit_rank"]

    # Check keyword overlap between question and gold
    q_terms = set(w for w in question.split() if len(w) > 3)
    g_terms = set(w for w in gold_preview.split() if len(w) > 3)
    overlap = len(q_terms & g_terms)

    if hit_rank is None:
        if overlap < 2:
            return "no_keyword_overlap"
        return "very_low_rank"

    if hit_rank > 5:
        if qtype == "single-session-preference":
            return "preference_abstract_query"
        if "when" in question or "date" in question or "time" in question:
            return "temporal_keyword_dilution"
        if overlap < 3:
            return "low_keyword_overlap"
        return "competing_distractors"

    if hit_rank > 1:
        return f"close_miss_rank_{hit_rank}"

    return "ok"


def main():
    data_path = "benchmark/data/longmemeval_s_cleaned.json"
    print(f"Loading {data_path}...")
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    questions = [q for q in data if q.get("question_type") != "abstention"]
    print(f"Re-running benchmark on {len(questions)} questions to capture failure details...")

    vault_base = tempfile.mkdtemp(prefix="lme_analyze_")
    all_results = []
    try:
        for i, q in enumerate(questions):
            r = evaluate_one(q, top_k=20, vault_base=vault_base)
            all_results.append(r)
            if (i + 1) % 50 == 0:
                print(f"  Progress: {i+1}/{len(questions)}")
    finally:
        shutil.rmtree(vault_base, ignore_errors=True)

    # Categorize
    failures_top5 = [r for r in all_results if r["hit_rank"] is None or r["hit_rank"] > 5]
    failures_top10 = [r for r in all_results if r["hit_rank"] is None or r["hit_rank"] > 10]
    failures_top20 = [r for r in all_results if r["hit_rank"] is None]

    # Categorize each failure
    for f in failures_top5:
        f["category"] = categorize_failure(f)

    # Stats
    print(f"\n{'='*60}")
    print(f"Failures @5 : {len(failures_top5)} ({len(failures_top5)/len(all_results)*100:.1f}%)")
    print(f"Failures @10: {len(failures_top10)}")
    print(f"Failures @20: {len(failures_top20)}")

    cat_counter = Counter(f["category"] for f in failures_top5)
    print(f"\nFailure categories (top-5 misses):")
    for cat, n in cat_counter.most_common():
        print(f"  {cat:<35}: {n}")

    # Per-type failure rates
    print(f"\nFailures @5 by question type:")
    type_counter = Counter(f["question_type"] for f in failures_top5)
    type_total = Counter(r["question_type"] for r in all_results)
    for qtype, n in type_counter.most_common():
        total = type_total[qtype]
        print(f"  {qtype:<30}: {n}/{total} ({n/total*100:.1f}%)")

    # Save
    with open("benchmark/failures_top5.json", "w", encoding="utf-8") as f:
        json.dump(failures_top5, f, indent=2, ensure_ascii=False)
    with open("benchmark/failures_top10.json", "w", encoding="utf-8") as f:
        json.dump(failures_top10, f, indent=2, ensure_ascii=False)
    with open("benchmark/all_results.json", "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    print(f"\nSaved: benchmark/failures_top5.json ({len(failures_top5)} cases)")
    print(f"Saved: benchmark/failures_top10.json ({len(failures_top10)} cases)")
    print(f"Saved: benchmark/all_results.json ({len(all_results)} cases)")


if __name__ == "__main__":
    main()
