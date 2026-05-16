"""Run retrieval benchmark on the synthetic corpus: R@K, MRR.

Usage:
  python benchmark/run_benchmark.py            # build & evaluate a fresh demo vault
  python benchmark/run_benchmark.py VAULT_PATH # evaluate against an existing vault
"""

from __future__ import annotations

import os
import re
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from agent_knowledge.core.compiler import Compiler
from agent_knowledge.core.vault import Vault
from agent_knowledge.search.engine import SearchEngine
from qa_pairs import QA_PAIRS, SYNTHETIC_SOURCES


def _cjk_normalize(text: str) -> str:
    """Collapse optional spaces between CJK and latin chars.

    '不能用 tag' and '不能用tag' should match each other.
    """
    text = re.sub(r"([\u4e00-\u9fff])\s+([a-zA-Z0-9_])", r"\1\2", text)
    text = re.sub(r"([a-zA-Z0-9_])\s+([\u4e00-\u9fff])", r"\1\2", text)
    return text


def build_demo_vault(vault_path: str) -> Vault:
    """Build a fresh demo vault from SYNTHETIC_SOURCES.

    Source IDs are derived from the synthetic source slugs (e.g. ``frontend-
    decision``) so the QA pairs can verify the *correct* source surfaces.
    """
    if os.path.exists(vault_path):
        shutil.rmtree(vault_path)
    vault = Vault(vault_path)
    vault.init(lang="en")
    compiler = Compiler(vault)
    for sid, title, body in SYNTHETIC_SOURCES:
        source = compiler.ingest(body, title=title, source_type="text")
        # Rewrite the source ID to the slug we declared so QA pairs are
        # human-readable when debugging mismatches.
        if source.id != sid:
            source.id = sid
            vault.save_source(source)
    return vault


def run_benchmark(vault_path: str, top_k: int = 10) -> dict:
    vault = Vault(vault_path)
    engine = SearchEngine(vault)
    engine.build_index()

    total = len(QA_PAIRS)
    hits_at = {1: 0, 3: 0, 5: 0, 10: 0}
    reciprocal_ranks: list[float] = []

    print(f"Running benchmark: {total} questions, top_k={top_k}\n")

    for i, (question, keywords, _expected_source_id) in enumerate(QA_PAIRS):
        results = engine.search(question, top_k=top_k)

        hit_rank: int | None = None
        for rank, r in enumerate(results, 1):
            result_text = (r.title + " " + r.snippet).lower()
            full_source = vault.load_source(r.id)
            if full_source:
                result_text += " " + full_source.content.lower()
            if any(
                _cjk_normalize(kw.lower()) in _cjk_normalize(result_text)
                for kw in keywords
            ):
                hit_rank = rank
                break

        if hit_rank:
            for k in hits_at:
                if hit_rank <= k:
                    hits_at[k] += 1
            reciprocal_ranks.append(1.0 / hit_rank)
            marker = "✅"
        else:
            reciprocal_ranks.append(0.0)
            marker = "❌"

        print(f"  {marker} [{i+1:2d}] Q: {question[:60]:<60} rank: {hit_rank or '-'}")

    recall_at = {k: round(v / total, 4) for k, v in hits_at.items()}
    mrr = round(sum(reciprocal_ranks) / total, 4) if total > 0 else 0.0

    print(f"\n{'='*60}")
    print(f"BENCHMARK RESULTS ({total} questions on synthetic corpus)")
    print(f"{'='*60}")
    print(f"  R@1  = {recall_at[1]:.1%}  ({hits_at[1]}/{total})")
    print(f"  R@3  = {recall_at[3]:.1%}  ({hits_at[3]}/{total})")
    print(f"  R@5  = {recall_at[5]:.1%}  ({hits_at[5]}/{total})")
    print(f"  R@10 = {recall_at[10]:.1%}  ({hits_at[10]}/{total})")
    print(f"  MRR  = {mrr:.4f}")
    print(
        "\nFor the academic LongMemEval-S benchmark see "
        "longmemeval_benchmark.py."
    )

    return {
        "total": total,
        "hits_at": hits_at,
        "recall_at": recall_at,
        "mrr": mrr,
    }


if __name__ == "__main__":
    if len(sys.argv) > 1:
        vault_path = sys.argv[1]
    else:
        vault_path = os.path.join(tempfile.gettempdir(), "ak-demo-bench-vault")
        print(f"[i] Building demo vault at {vault_path}\n")
        build_demo_vault(vault_path)
    run_benchmark(vault_path)
