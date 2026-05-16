"""Run retrieval benchmark with BM25 + Vector (RRF fusion).

Set ``AK_EMBEDDING_URL``, ``AK_EMBEDDING_KEY`` and ``AK_EMBEDDING_MODEL``
to point at any OpenAI-compatible embedding endpoint
(e.g. an OpenAI endpoint, a self-hosted vLLM server, an Ollama bridge).
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from agent_knowledge.core.vault import Vault
from agent_knowledge.search.embedder import APIEmbedder
from agent_knowledge.search.engine import SearchEngine
from qa_pairs import QA_PAIRS

# OpenAI-compatible embedding endpoint. The defaults are placeholders;
# override via env vars to point at your own deployment.
EMBEDDING_BASE_URL = os.environ.get(
    "AK_EMBEDDING_URL",
    "https://api.openai.com/v1",
)
EMBEDDING_API_KEY = os.environ.get("AK_EMBEDDING_KEY", "")
EMBEDDING_MODEL = os.environ.get(
    "AK_EMBEDDING_MODEL",
    "text-embedding-3-small",
)


def run_benchmark(vault_path: str, top_k: int = 10) -> dict:
    vault = Vault(vault_path)

    print(f"Embedding: {EMBEDDING_MODEL} @ {EMBEDDING_BASE_URL}")
    embedder = APIEmbedder(
        base_url=EMBEDDING_BASE_URL,
        api_key=EMBEDDING_API_KEY,
        model=EMBEDDING_MODEL,
        timeout=30.0,
    )

    print("Probing embedding endpoint...")
    test_vec = embedder.embed(["test"])
    dim = len(test_vec[0]) if test_vec else 0
    print(f"Embedding dim: {dim}")
    if dim == 0 or all(v == 0.0 for v in test_vec[0]):
        print("⚠️  Embedding endpoint returned zero vector, falling back to BM25 only")
        embedder = None

    engine = SearchEngine(vault, embedder=embedder)
    print("Building index (BM25 + Vector)...")
    doc_count = engine.build_index()
    print(f"Indexed {doc_count} documents\n")

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
            if any(kw.lower() in result_text for kw in keywords):
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

    mode = "BM25 + Vector (RRF)" if embedder else "BM25 only (embedding failed)"
    print(f"\n{'='*60}")
    print(f"BENCHMARK RESULTS — {mode} ({total} questions)")
    print(f"{'='*60}")
    print(f"  R@1  = {recall_at[1]:.1%}  ({hits_at[1]}/{total})")
    print(f"  R@3  = {recall_at[3]:.1%}  ({hits_at[3]}/{total})")
    print(f"  R@5  = {recall_at[5]:.1%}  ({hits_at[5]}/{total})")
    print(f"  R@10 = {recall_at[10]:.1%}  ({hits_at[10]}/{total})")
    print(f"  MRR  = {mrr:.4f}")

    return {"recall_at": recall_at, "mrr": mrr}


if __name__ == "__main__":
    vault_path = sys.argv[1] if len(sys.argv) > 1 else "/tmp/test-ak-vault"
    run_benchmark(vault_path)
