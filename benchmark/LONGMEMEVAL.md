# LongMemEval-S — Methodology

For the headline numbers see [`../BENCHMARK.md`](../BENCHMARK.md).
This document covers methodology, design choices, and full reproduction
options.

## Dataset

[LongMemEval](https://arxiv.org/abs/2410.10813) (ICLR 2025) tests 5
memory capabilities across 500 questions:

| Capability | Count | What it tests |
|:-----------|------:|:--------------|
| Information Extraction | ~200 | Retrieve specific facts from a single session |
| Multi-Session Reasoning | ~100 | Combine info across multiple sessions |
| Temporal Reasoning | ~70 | Time-ordered events |
| Knowledge Updates | ~100 | Evolving / conflicting information |
| Abstention | 30 | Know when information is unavailable |

We evaluate on **LongMemEval-S** (retrieval-only): the full 500
questions, each with ~48 history sessions (~115K tokens). The task is
pure retrieval — does the system surface the correct session(s) in its
top-K results?

## Setup

- Each question gets an independent vault (no cross-question leakage).
- Each history session is ingested as a separate source (`session_id` →
  `source_id`).
- Query = the question text. Hit = any gold session appears in top-K.

## Search configuration

- BM25 + Exact substring match + RRF fusion
- No vector, no graph, no reranker — lightweight zero-dependency config
- Top-K reported: 5 / 10 / 20

## Metrics

- **R@K** — fraction of questions where any gold session is in top-K
- **MRR** — mean of `1/rank` for the first gold hit
- **NDCG@10** — normalized discounted cumulative gain at 10

Wall time: 221s (3.7 min) single core, ~10 ms / question.

## Reproduction

```bash
# 1. Get the dataset
pip install huggingface_hub
python3 -c "
from huggingface_hub import hf_hub_download
hf_hub_download(
    repo_id='xiaowu0162/longmemeval-cleaned',
    filename='longmemeval_s_cleaned.json',
    repo_type='dataset',
    local_dir='benchmark/data',
)"

# 2. Run
python benchmark/longmemeval_benchmark.py                    # full
python benchmark/longmemeval_benchmark.py --sample 50        # quick
python benchmark/longmemeval_benchmark.py --workers 4        # parallel
python benchmark/longmemeval_benchmark.py --semantic         # +vector
```

## Design choices

- **No vector by default** — establishes a zero-dependency baseline.
  Enable with `--semantic` (requires `sentence-transformers`) for
  hybrid retrieval.
- **No graph / no reranker** — both are designed for compiled
  knowledge (entities, claims) rather than raw session retrieval.
- **Independent vaults per question** — matches the LongMemEval
  protocol; eliminates cross-question information leakage.
