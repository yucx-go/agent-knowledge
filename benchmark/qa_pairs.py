"""Benchmark QA pairs over a synthetic, project-internal corpus.

Each pair is ``(question, expected_answer_keywords, expected_source_id)``.
A search result is counted as a hit when any of the keywords appears in
the result's title, snippet, or full source content.

The corpus is built from agent-knowledge's own public documentation plus
a handful of synthetic sources (defined in :data:`SYNTHETIC_SOURCES`)
that exercise the typical claim shapes: decision, fact, risk, todo.

For a richer / academic benchmark see ``longmemeval_benchmark.py``,
which evaluates against the public LongMemEval-S dataset.
"""

from __future__ import annotations

# ─────────────────────────────────────────────────────────────────────────────
# Synthetic sources — small, self-contained, copyright-clean.
# Each tuple: (source_id, title, body). ``source_id`` is referenced by the
# QA pairs below so the benchmark can verify the *correct* source ranks first.
# ─────────────────────────────────────────────────────────────────────────────

SYNTHETIC_SOURCES: list[tuple[str, str, str]] = [
    (
        "frontend-decision",
        "Frontend framework decision",
        "## Decisions\n"
        "- We decided to use React instead of Vue for the dashboard project.\n"
        "- The team has more React experience and the component library "
        "ecosystem is richer.\n"
        "## Risks\n"
        "- Migration from the existing Vue codebase will take 2 sprints.",
    ),
    (
        "mcp-migration",
        "API gateway MCP migration",
        "## Decisions\n"
        "- The API gateway was migrated from REST to the MCP protocol.\n"
        "## Learnings\n"
        "- MCP provides better agent interoperability and standardized "
        "tool calling.\n"
        "- Latency increased by 15ms but agent compatibility improved "
        "significantly.",
    ),
    (
        "mcp-timeout-bug",
        "MCP timeout bug postmortem",
        "## Risks\n"
        "- The MCP protocol exposed a critical bug: timeout handling causes "
        "silent failures.\n"
        "- When the MCP server does not respond within 30s, the agent retries "
        "indefinitely.\n"
        "## Actions\n"
        "- Fix: added circuit breaker with 3-retry limit and exponential "
        "backoff.",
    ),
    (
        "q1-financials",
        "Q1 synthetic financial results",
        "## Facts\n"
        "- Q1 revenue hit $4.2M, exceeding the demo budget by 3.6%.\n"
        "- International markets contributed 62% of total revenue.\n"
        "- Gross margin at 11.3%, 0.9pp above the demo budget target.",
    ),
    (
        "search-stack",
        "agent-knowledge search stack",
        "## Decisions\n"
        "- Default retrieval is BM25 + Exact + Graph + Reranker with RRF "
        "fusion (k=60).\n"
        "- Vector search is optional and disabled by default — zero external "
        "dependencies.\n"
        "## Learnings\n"
        "- The reranker uses 7 TF-IDF signals: keyword match, entity "
        "coverage, exact phrase, title match, position, proximity, and the "
        "original retrieval score.",
    ),
    (
        "umsf-boundary",
        "UMSF universal boundary format",
        "## Decisions\n"
        "- Every adapter produces a UMSFDocument; the Compiler reads only "
        "UMSF.\n"
        "## Facts\n"
        "- UMSF defines 8 typed events: message, tool_use, tool_result, "
        "decision, file_change, error, session_start, session_end.\n"
        "- It defines 7 source types: conversation, curated_memory, "
        "skill_artifact, tool_trace, decision, file_change, document.",
    ),
    (
        "dream-cycle",
        "Dream cycle scoring dimensions",
        "## Facts\n"
        "- The Dream Cycle scores candidates on six dimensions: frequency, "
        "relevance, query diversity, recency, consolidation, concept "
        "richness.\n"
        "- Default weights are 0.24 / 0.30 / 0.15 / 0.15 / 0.10 / 0.06.\n"
        "## Decisions\n"
        "- Candidates scoring above 0.45 are promoted to compiled truth.",
    ),
    (
        "vault-storage",
        "Vault storage layout",
        "## Facts\n"
        "- The vault uses YAML files for sources/entities/concepts and a "
        "SQLite event index at .ak-events.db.\n"
        "- Pull adapter dedup state lives at .ak-pull-state.json, namespaced "
        "by adapter name.",
    ),
    (
        "entity-extraction",
        "Reverse-filter entity extraction",
        "## Decisions\n"
        "- Entity extraction switched from positive patterns to a reverse "
        "filter in v0.3.0.\n"
        "## Learnings\n"
        "- Wide-net candidate generation is followed by stopword filtering.\n"
        "- About 300 Chinese stopwords and 120 English stopwords cover the "
        "language-level filtering, leaving domain vocabulary intact.",
    ),
    (
        "claim-model",
        "Claim and evidence model",
        "## Facts\n"
        "- A Claim has confidence, polarity, durability, status, and a list "
        "of Evidence entries.\n"
        "- Confidence is recomputed as total_weight / (total_weight + 1.0), "
        "capped at 0.95.\n"
        "- Evidence weights: PRIMARY=1.0, SECONDARY=0.7, TERTIARY=0.4, "
        "PRIOR=0.2.",
    ),
]


# ─────────────────────────────────────────────────────────────────────────────
# QA pairs over the synthetic corpus
# ─────────────────────────────────────────────────────────────────────────────

QA_PAIRS = [
    # frontend-decision
    (
        "Which frontend framework did the team pick for the dashboard?",
        ["react", "instead of vue"],
        "frontend-decision",
    ),
    (
        "Why did the team prefer React over Vue?",
        ["team", "experience", "ecosystem"],
        "frontend-decision",
    ),
    (
        "How long will the Vue-to-React migration take?",
        ["2 sprints"],
        "frontend-decision",
    ),
    # mcp-migration
    (
        "What protocol did the API gateway migrate to?",
        ["mcp"],
        "mcp-migration",
    ),
    (
        "What was the latency impact of the MCP migration?",
        ["15ms"],
        "mcp-migration",
    ),
    # mcp-timeout-bug
    (
        "What MCP bug was discovered around timeouts?",
        ["silent failure", "retries indefinitely"],
        "mcp-timeout-bug",
    ),
    (
        "How was the MCP retry loop fixed?",
        ["circuit breaker", "3-retry", "exponential"],
        "mcp-timeout-bug",
    ),
    # q1-financials
    (
        "What was Q1 demo revenue?",
        ["$4.2m", "4.2m"],
        "q1-financials",
    ),
    (
        "How much of Q1 revenue came from international markets?",
        ["62%"],
        "q1-financials",
    ),
    (
        "What was the Q1 gross margin in the demo?",
        ["11.3%"],
        "q1-financials",
    ),
    # search-stack
    (
        "Which engines participate in the default search stack?",
        ["bm25", "exact", "graph", "reranker"],
        "search-stack",
    ),
    (
        "Is vector search required by default?",
        ["optional", "disabled by default", "zero external"],
        "search-stack",
    ),
    (
        "How many signals does the reranker use?",
        ["7", "seven"],
        "search-stack",
    ),
    # umsf-boundary
    (
        "How many UMSF event types are there?",
        ["8", "eight"],
        "umsf-boundary",
    ),
    (
        "How many UMSF source types are there?",
        ["7", "seven"],
        "umsf-boundary",
    ),
    (
        "What format do adapters produce?",
        ["umsfdocument", "umsf"],
        "umsf-boundary",
    ),
    # dream-cycle
    (
        "How many dimensions does the Dream Cycle score on?",
        ["6", "six"],
        "dream-cycle",
    ),
    (
        "What is the Dream Cycle promotion threshold?",
        ["0.45"],
        "dream-cycle",
    ),
    (
        "What are the six dream-cycle scoring dimensions?",
        ["frequency", "relevance", "recency"],
        "dream-cycle",
    ),
    # vault-storage
    (
        "What storage format does the vault use for sources and entities?",
        ["yaml"],
        "vault-storage",
    ),
    (
        "Where does the event index live?",
        [".ak-events.db", "sqlite"],
        "vault-storage",
    ),
    (
        "How is pull-adapter dedup state namespaced?",
        ["adapter name", "namespaced"],
        "vault-storage",
    ),
    # entity-extraction
    (
        "When did entity extraction switch to reverse filtering?",
        ["v0.3.0", "0.3.0"],
        "entity-extraction",
    ),
    (
        "Roughly how many stopwords cover language-level filtering?",
        ["300", "120"],
        "entity-extraction",
    ),
    # claim-model
    (
        "How is claim confidence recomputed?",
        ["total_weight", "weight + 1", "0.95"],
        "claim-model",
    ),
    (
        "What weight does primary evidence carry?",
        ["1.0", "primary"],
        "claim-model",
    ),
    (
        "Which fields are tracked on a Claim?",
        ["confidence", "polarity", "durability"],
        "claim-model",
    ),
]
