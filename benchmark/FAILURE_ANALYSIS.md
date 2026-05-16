# LongMemEval-S Failure Analysis

The 3.4% R@5 misses (17 / 500) are almost entirely an inherent limit of
BM25 keyword retrieval — near-zero token overlap between query and gold
session.

## Failure coverage

| Threshold | Misses | Share |
|-----------|-------:|------:|
| Top-5  | 17 / 500 | 3.4% |
| Top-10 |  9 / 500 | 1.8% |
| Top-20 |  3 / 500 | 0.6% |

Top-1 and top-20 scores on the failing cases are barely distinguishable
(0.054–0.057); successful cases score 0.08–0.15 at top-1.
**This is a recall problem, not a ranking problem.**

## Failure modes

### 1. "By the way" mentions (~10/17)

The user asks about A, and casually mentions answer B with phrasings
like "by the way / I just / today".

| Question | Where the answer appears |
|----------|--------------------------|
| What music streaming service? → Spotify | While asking for concert recs: *"by the way I've been using Spotify"* |
| Kitchen appliance bought 10 days ago? → smoker | While asking about BBQ sauce: *"I just got a smoker today"* |
| Cooked for friend? → chocolate cake | While asking for dessert ideas: *"I made a chocolate cake last week"* |

### 2. Cross-session counting / aggregation (~3/17)

| Question | Gold |
|----------|------|
| How many doctors did I visit? → 3 | 3 sessions, each mentioning one doctor |
| How many siblings? → 4 | "siblings" only appears as context |

The query subject is incidental in gold sessions, never the topic.

### 3. Abstract preference queries (4/17)

| Question | Gold topic |
|----------|------------|
| Recommend recent publications? | Previously discussed medical AI |
| What should I serve for dinner? | Earlier mention of growing basil and mint |
| Battery life tips? | Previously bought a power bank |

The query contains none of the keywords the answer would carry.

### 4. Temporal-adverb dilution (2/17)

Phrases like "two weeks ago" appear in every session and carry no
discriminative power.

## Why "by the way" extraction does not help

Intuitively, surfacing incidental mentions as standalone short docs
(`fact_extractor`) should boost BM25 hits. In practice it slightly
regresses:

| Configuration | R@5 | R@10 | MRR | NDCG@10 |
|---------------|----:|-----:|----:|--------:|
| Default            | **96.6%** | 98.2% | **0.9031** | 0.9218 |
| `--enable-facts`   | 96.0%     | 98.0% |   0.8847   | 0.9073 |

Two reasons:

1. **The failure is a semantic gap, not buried keywords.**
   `"music streaming service"` vs `"using Spotify"` share no surface
   tokens — fact extraction still cannot bridge them.
2. **Short docs cause spurious matches.** Preference queries (*"what
   should I serve for dinner"*) pull any fact snippet containing
   "dinner" to the top, displacing the real source.
   `single-session-preference` R@5 drops from 86.7% to 80.0%.

The module stays in the repo (`fact_extractor.py`, 21 unit tests),
toggled via `enable_fact_extraction=True`. It is a good fit for
production workloads where users bury information in long documents
and queries share concrete keywords with the answer.

## Two paths to R@5 → 99%

The remaining 3.4% requires semantic understanding:

1. Vector retrieval (`--semantic`, needs `sentence-transformers` or any
   OpenAI-compatible embedding API)
2. LLM-assisted indexing (rewrite sessions into retrievable fact lists
   at ingest time)

Both require external dependencies. The current 96.6% should be read
as **the ceiling of the zero-dependency configuration**.
