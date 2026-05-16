# Benchmark

[LongMemEval-S](https://arxiv.org/abs/2410.10813) (ICLR 2025) — 500
questions, ~48 sessions/question, ~115K tokens/question.

Search config: BM25 + Exact Match + RRF fusion. No vector, no graph,
no reranker. **Zero external dependencies.**

## Headline

| Metric | agent-knowledge | AgentMemory |
|--------|:-:|:-:|
| R@5    | **96.6%** | 95.2% |
| R@10   | 98.2% | 98.6% |
| R@20   | 99.4% | — |
| MRR    | **0.9031** | 0.882 |
| NDCG@10 | 0.9218 | — |

## Per-type breakdown

| Type | N | R@5 | R@10 | MRR |
|:-----|--:|----:|-----:|----:|
| knowledge-update | 78 | 100.0% | 100.0% | 0.971 |
| single-session-assistant | 56 | 100.0% | 100.0% | 0.991 |
| multi-session | 133 | 97.7% | 98.5% | 0.900 |
| single-session-user | 70 | 97.1% | 100.0% | 0.933 |
| temporal-reasoning | 133 | 94.0% | 97.0% | 0.886 |
| single-session-preference | 30 | 86.7% | 90.0% | 0.583 |

The weakest category — `single-session-preference` — uses abstract
language with little keyword overlap; vector search would likely push
it past 95%. See [`benchmark/FAILURE_ANALYSIS.md`](benchmark/FAILURE_ANALYSIS.md)
for the failure-mode analysis.

## Reproduction

```bash
pip install huggingface_hub

python3 -c "
from huggingface_hub import hf_hub_download
hf_hub_download(
    repo_id='xiaowu0162/longmemeval-cleaned',
    filename='longmemeval_s_cleaned.json',
    repo_type='dataset',
    local_dir='benchmark/data',
)"

python benchmark/longmemeval_benchmark.py
```

For a synthetic smoke test that needs no download:

```bash
python benchmark/run_benchmark.py
```
