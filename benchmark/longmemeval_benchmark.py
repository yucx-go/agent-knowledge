"""LongMemEval-S Benchmark for agent-knowledge.

LongMemEval (ICLR 2025) evaluates long-term memory for conversational agents.
500 questions across 5 capabilities:
  - Information Extraction (single-session-user)
  - Multi-Session Reasoning (multi-session)
  - Temporal Reasoning (temporal-reasoning)
  - Knowledge Updates (knowledge-update)
  - Abstention (abstention) — skipped

Each question has ~48 history sessions (~115K tokens).
We build an independent vault per question, ingest sessions as sources,
then evaluate retrieval: does the gold session appear in top-K results?

Metrics: R@5, R@10, R@20, MRR, NDCG@10
Comparison target: AgentMemory R@5=95.2%, R@10=98.6%, MRR=0.882

Usage:
    python benchmark/longmemeval_benchmark.py [--data PATH] [--sample N] [--workers N]
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


@dataclass
class QuestionResult:
    """Result for a single question."""
    question_id: str
    question_type: str
    hit_rank: int | None  # rank where gold session found, or None
    top_k_ids: list[str] = field(default_factory=list)
    elapsed_s: float = 0.0


def format_session_content(session: list[dict]) -> str:
    """Format a session (list of messages) into plain text for ingestion."""
    parts = []
    for msg in session:
        role = msg.get("role", "unknown")
        content = msg.get("content", "")
        parts.append(f"{role}: {content}")
    return "\n".join(parts)


def evaluate_single_question(args: tuple) -> QuestionResult:
    """Evaluate a single question. Designed to run in a subprocess.

    Args is a tuple: (question_data, top_k, vault_base_dir, enable_facts)
    """
    if len(args) == 4:
        question_data, top_k, vault_base_dir, enable_facts = args
    else:
        # Back-compat for older callers
        question_data, top_k, vault_base_dir = args
        enable_facts = False

    # Import inside subprocess to avoid pickling issues
    from agent_knowledge.core.vault import Vault
    from agent_knowledge.core.models import Source
    from agent_knowledge.search.engine import SearchEngine

    qid = question_data["question_id"]
    qtype = question_data["question_type"]
    question = question_data["question"]
    sessions = question_data["haystack_sessions"]
    session_ids = question_data["haystack_session_ids"]
    gold_ids = set(question_data["answer_session_ids"])

    # Create a temporary vault for this question
    vault_path = os.path.join(vault_base_dir, f"vault_{qid}")
    try:
        vault = Vault(vault_path)
        vault.init()

        # Ingest each session as a source
        for sid, session in zip(session_ids, sessions):
            content = format_session_content(session)
            source = Source(
                id=sid,
                title=f"Session {sid}",
                content=content,
                source_type="conversation",
            )
            vault.save_source(source)

        # Build search index and query
        engine = SearchEngine(
            vault,
            enable_graph=False,
            enable_reranker=False,
            enable_fact_extraction=enable_facts,
        )
        engine.build_index()

        t0 = time.time()
        results = engine.search(question, top_k=top_k)
        elapsed = time.time() - t0

        # Check if any gold session appears in results
        hit_rank = None
        result_ids = []
        for rank, r in enumerate(results, 1):
            result_ids.append(r.id)
            if r.id in gold_ids and hit_rank is None:
                hit_rank = rank

        return QuestionResult(
            question_id=qid,
            question_type=qtype,
            hit_rank=hit_rank,
            top_k_ids=result_ids,
            elapsed_s=elapsed,
        )
    finally:
        # Cleanup vault
        if os.path.exists(vault_path):
            shutil.rmtree(vault_path, ignore_errors=True)


def compute_metrics(results: list[QuestionResult], ks: list[int] = None) -> dict:
    """Compute retrieval metrics from question results."""
    if ks is None:
        ks = [5, 10, 20]

    total = len(results)
    if total == 0:
        return {}

    hits_at = {k: 0 for k in ks}
    reciprocal_ranks = []
    ndcg_scores = []  # NDCG@10

    for r in results:
        if r.hit_rank is not None:
            for k in ks:
                if r.hit_rank <= k:
                    hits_at[k] += 1
            reciprocal_ranks.append(1.0 / r.hit_rank)
            # NDCG@10: single relevant doc, relevance=1 at hit_rank
            if r.hit_rank <= 10:
                ndcg_scores.append(1.0 / math.log2(r.hit_rank + 1))
            else:
                ndcg_scores.append(0.0)
        else:
            reciprocal_ranks.append(0.0)
            ndcg_scores.append(0.0)

    recall_at = {k: round(v / total, 4) for k, v in hits_at.items()}
    mrr = round(sum(reciprocal_ranks) / total, 4)
    ndcg10 = round(sum(ndcg_scores) / total, 4)
    avg_latency = round(sum(r.elapsed_s for r in results) / total, 3)

    return {
        "total": total,
        "hits_at": hits_at,
        "recall_at": recall_at,
        "mrr": mrr,
        "ndcg@10": ndcg10,
        "avg_latency_s": avg_latency,
    }


def run_benchmark(
    data_path: str,
    sample_n: int | None = None,
    workers: int = 1,
    top_k: int = 20,
    enable_facts: bool = False,
) -> dict:
    """Run the full LongMemEval-S benchmark.

    Args:
        data_path: Path to longmemeval_s_cleaned.json
        sample_n: If set, only evaluate first N questions (for quick testing)
        workers: Number of parallel workers
        top_k: Maximum K for retrieval (we compute R@5, R@10, R@20)

    Returns:
        Dict with overall + per-type metrics
    """
    print(f"Loading dataset from {data_path}...")
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Filter out abstention questions
    questions = [q for q in data if q.get("question_type") != "abstention"]
    abstention_count = len(data) - len(questions)
    print(f"Total questions: {len(data)}, Abstention (skipped): {abstention_count}, Evaluating: {len(questions)}")

    if sample_n and sample_n < len(questions):
        questions = questions[:sample_n]
        print(f"Sampling first {sample_n} questions")

    # Create temp dir for vaults
    vault_base = tempfile.mkdtemp(prefix="longmemeval_")
    print(f"Vault temp dir: {vault_base}")

    # Prepare args for each question
    eval_args = [(q, top_k, vault_base, enable_facts) for q in questions]

    results: list[QuestionResult] = []
    t_start = time.time()

    try:
        if workers <= 1:
            # Sequential execution
            for i, args in enumerate(eval_args):
                r = evaluate_single_question(args)
                results.append(r)
                marker = "✅" if r.hit_rank is not None else "❌"
                rank_str = str(r.hit_rank) if r.hit_rank else "-"
                print(f"  {marker} [{i+1:3d}/{len(questions)}] {r.question_type:<25} rank={rank_str:<4} {r.elapsed_s:.2f}s")
        else:
            # Parallel execution
            print(f"Running with {workers} workers...")
            completed = 0
            with ProcessPoolExecutor(max_workers=workers) as executor:
                futures = {executor.submit(evaluate_single_question, args): i for i, args in enumerate(eval_args)}
                for future in as_completed(futures):
                    r = future.result()
                    results.append(r)
                    completed += 1
                    marker = "✅" if r.hit_rank is not None else "❌"
                    rank_str = str(r.hit_rank) if r.hit_rank else "-"
                    if completed % 10 == 0 or completed == len(questions):
                        print(f"  {marker} [{completed:3d}/{len(questions)}] {r.question_type:<25} rank={rank_str:<4}")
    finally:
        # Cleanup
        shutil.rmtree(vault_base, ignore_errors=True)

    total_time = time.time() - t_start

    # Overall metrics
    overall = compute_metrics(results)
    overall["wall_time_s"] = round(total_time, 1)

    # Per-type metrics
    type_groups: dict[str, list[QuestionResult]] = {}
    for r in results:
        type_groups.setdefault(r.question_type, []).append(r)

    per_type = {}
    for qtype, type_results in sorted(type_groups.items()):
        per_type[qtype] = compute_metrics(type_results)

    # Print results
    print(f"\n{'='*70}")
    print(f"LongMemEval-S BENCHMARK RESULTS")
    print(f"{'='*70}")
    print(f"  Questions evaluated: {overall['total']}")
    print(f"  Wall time: {total_time:.1f}s ({total_time/60:.1f}min)")
    print(f"  Avg latency/question: {overall['avg_latency_s']:.3f}s")
    print()
    print(f"  {'Metric':<12} {'agent-knowledge':>18} {'AgentMemory':>18}")
    print(f"  {'-'*12} {'-'*18} {'-'*18}")
    print(f"  {'R@5':<12} {overall['recall_at'].get(5, 0):.1%}{'':<12} {'95.2%':>18}")
    print(f"  {'R@10':<12} {overall['recall_at'].get(10, 0):.1%}{'':<12} {'98.6%':>18}")
    print(f"  {'R@20':<12} {overall['recall_at'].get(20, 0):.1%}{'':<12} {'—':>18}")
    print(f"  {'MRR':<12} {overall['mrr']:.4f}{'':<12} {'0.8820':>18}")
    print(f"  {'NDCG@10':<12} {overall['ndcg@10']:.4f}{'':<12} {'—':>18}")

    print(f"\n  Per-type breakdown:")
    print(f"  {'Type':<30} {'N':>4} {'R@5':>8} {'R@10':>8} {'MRR':>8}")
    print(f"  {'-'*30} {'-'*4} {'-'*8} {'-'*8} {'-'*8}")
    for qtype, metrics in per_type.items():
        print(f"  {qtype:<30} {metrics['total']:>4} {metrics['recall_at'].get(5, 0):>7.1%} {metrics['recall_at'].get(10, 0):>7.1%} {metrics['mrr']:>7.4f}")

    return {
        "overall": overall,
        "per_type": per_type,
        "config": {
            "top_k": top_k,
            "workers": workers,
            "sample_n": sample_n,
            "enable_facts": enable_facts,
            "search_engine": (
                "BM25+Exact+RRF" + (" +Facts" if enable_facts else "")
                + " (no vector, no graph, no reranker)"
            ),
        },
    }


def save_results(results: dict, output_path: str) -> None:
    """Save benchmark results as JSON."""
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="LongMemEval-S Benchmark for agent-knowledge")
    parser.add_argument("--data", default="benchmark/data/longmemeval_s_cleaned.json",
                        help="Path to longmemeval_s_cleaned.json")
    parser.add_argument("--sample", type=int, default=None,
                        help="Only evaluate first N questions (for quick testing)")
    parser.add_argument("--workers", type=int, default=1,
                        help="Number of parallel workers")
    parser.add_argument("--top-k", type=int, default=20,
                        help="Maximum K for retrieval")
    parser.add_argument("--output", default="benchmark/longmemeval_results.json",
                        help="Output JSON path")
    parser.add_argument("--enable-facts", action="store_true",
                        help="Enable by-the-way fact extraction (opt-in; "
                             "see benchmark/FAILURE_ANALYSIS.md)")
    args = parser.parse_args()

    if not os.path.exists(args.data):
        print(f"ERROR: Dataset not found at {args.data}")
        print(f"Download it first:")
        print(f"  pip install huggingface_hub")
        print(f"  python3 -c \"from huggingface_hub import hf_hub_download; hf_hub_download(repo_id='xiaowu0162/longmemeval-cleaned', filename='longmemeval_s_cleaned.json', repo_type='dataset', local_dir='benchmark/data')\"")
        sys.exit(1)

    results = run_benchmark(
        data_path=args.data,
        sample_n=args.sample,
        workers=args.workers,
        top_k=args.top_k,
        enable_facts=args.enable_facts,
    )
    save_results(results, args.output)


if __name__ == "__main__":
    main()
