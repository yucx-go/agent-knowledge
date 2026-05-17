# agent-knowledge

> **Long-term memory for AI agents — without vector embeddings.**

[![PyPI](https://img.shields.io/pypi/v/compiled-memory.svg)](https://pypi.org/project/compiled-memory/)
[![Downloads](https://static.pepy.tech/badge/compiled-memory)](https://pepy.tech/project/compiled-memory)
[![Python](https://img.shields.io/pypi/pyversions/compiled-memory.svg)](https://pypi.org/project/compiled-memory/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![CI](https://github.com/yucx-go/agent-knowledge/actions/workflows/ci.yml/badge.svg)](https://github.com/yucx-go/agent-knowledge/actions/workflows/ci.yml)
[![MCP](https://img.shields.io/badge/MCP-compatible-blue.svg)](docs/mcp-integration.md)
[![GitHub stars](https://img.shields.io/github/stars/yucx-go/agent-knowledge?style=social)](https://github.com/yucx-go/agent-knowledge/stargazers)

**English** | [简体中文](README.zh-CN.md)

A persistent **knowledge base and long-term memory layer** for AI agents. Conversations, documents, and decisions are auto-compiled into structured knowledge with claim/evidence provenance, append-only timeline, and contradiction detection. Pure Python, local-first, MIT licensed.

Ships as an **MCP server** (stdio JSON-RPC 2.0) for [Claude Code](examples/mcp/claude-code.json), [Cursor](examples/mcp/cursor.json), [Codex](examples/mcp/codex.toml), and any MCP-aware client. **96.6% R@5 on [LongMemEval-S](https://arxiv.org/abs/2410.10813)** with zero vector dependencies — BM25 + Knowledge Graph + RRF only.

## Why

LLM memory today is either flat RAG chunks or a key-value preference cache. Neither answers *"how did we get here?"*.

agent-knowledge adds a **knowledge-compilation layer**: raw material is decomposed into Claims and Evidence, claims about the same entity are merged into a Compiled Truth, and a timeline is kept append-only — rewritten holistically when new evidence arrives. Every fact is traceable to its source, timestamp, and confidence.

## Features

- 🧠 **Compiled long-term memory** — Claim / Evidence / Compiled Truth / append-only Timeline
- 🔍 **Multi-path retrieval** — Exact + BM25 + Knowledge Graph + weighted RRF + TF-IDF reranker
- 🚫 **Zero vector dependencies** — no embeddings, no vector database, no external services required
- 🔌 **MCP server** — stdio JSON-RPC 2.0, 8 tools + 2 resource URIs, works with Claude Code / Cursor / Codex / any MCP client
- 🪶 **Local-first storage** — human-readable YAML vault + SQLite event index, sync-friendly with git
- 🪝 **Auto-capture hooks** — 7 built-in hooks for messages, tool calls, decisions, file changes, errors
- ⚔️ **Contradiction detection** — polarity-based; surfaces "we used to say X, now we say not-X"
- 💤 **Dream cycle** — offline memory consolidation, deduplication, supersession
- 📊 **Benchmarked** — 96.6% R@5 / 0.9031 MRR on LongMemEval-S (ICLR 2025)
- 🪪 **MIT licensed**, Python 3.10–3.13, pure standard library + PyYAML

## Before / After

**Scenario**: over 10 weeks the team revisits its frontend stack three times — Week 1 picks Vue, Week 6 evaluates React, Week 10 switches to React. All three meeting notes are in the agent's conversation history.

A new teammate asks the agent *"what frontend are we on, and why?"*

**Without** — RAG over chat history

```
The team decided to use Vue as the frontend framework.
```

Vector search returns the highest-similarity chunk (the earliest meeting note). No temporal awareness → **stale answer**.

**With agent-knowledge** — `ak_query` returns the entity (real vault YAML)

```yaml
name: Frontend stack
entity_type: concept
compiled_truth:
  claims:
    - text: use React for the dashboard
      status: active
      confidence: 0.85
      evidence:
        - source_id: c3a9d1f2
          weight: 1.0
    - text: use Vue for the dashboard
      status: superseded
      confidence: 0.65
      evidence:
        - source_id: a1b2c3d4
          weight: 0.7
timeline:
  - date: 2026-02-01
    title: Decided on Vue
    source_id: a1b2c3d4
  - date: 2026-03-12
    title: Evaluated React
    source_id: b5e6f7a8
  - date: 2026-04-09
    title: Switched to React; Vue ecosystem limits
    source_id: c3a9d1f2
```

The agent now sees **the current fact, the timeline, the sources, and the superseded prior claim** in one call. Its answer naturally becomes "we're on React — switched from Vue in April due to ecosystem limits," with every fact traceable to a source.

## Quick Start

```bash
pip install compiled-memory   # PyPI package; the Python module is `agent_knowledge`

ak init   ~/my-knowledge
ak ingest ~/my-knowledge --file ./meeting-notes.md
ak query  ~/my-knowledge "why did we pick React?"
ak dream  ~/my-knowledge       # offline consolidation
ak lint   ~/my-knowledge       # health check
```

Or run it as an MCP server, plugged into any MCP-aware client:

```bash
ak mcp ~/my-knowledge
```

Copy-paste configs for Claude Code / Cursor / Codex live in [`examples/mcp/`](examples/mcp/); the full tool list is in [`docs/mcp-integration.md`](docs/mcp-integration.md). Agents picking up this repo should read [`AGENTS.md`](AGENTS.md) first.

## Architecture

```
┌─────────────────────────────────┐
│  Adapter Layer                  │  CLI · MCP · pull adapters
├─────────────────────────────────┤
│  UMSF Boundary                  │  unified data contract
├─────────────────────────────────┤
│  Knowledge Layer (core)         │  Compiler · Compiled Truth · Hooks · Dream
├─────────────────────────────────┤
│  Storage                        │  Vault (YAML) + EventIndex (SQLite)
├─────────────────────────────────┤
│  Search Layer                   │  Exact + BM25 + Graph + RRF + Reranker
└─────────────────────────────────┘
```

- **Knowledge Layer** — pure Python over local files, no external services
- **Search Layer** — zero dependencies by default; optional embedding model for stronger semantic recall
- **Adapter Layer** — UMSF unifies the boundary; a new agent adapter is ~80 lines

See [`docs/architecture.md`](docs/architecture.md).

## Benchmark

[LongMemEval-S (ICLR 2025)](https://arxiv.org/abs/2410.10813) — 500 questions, ~48 sessions/question, ~115K tokens/question:

| Metric  | Score |
|:--------|:-----:|
| R@5     | **96.6%** |
| R@10    | 98.2% |
| MRR     | **0.9031** |
| NDCG@10 | 0.9218 |

Zero vector dependencies — BM25 + Exact Match + RRF only. Full per-type breakdown in [`BENCHMARK.md`](BENCHMARK.md).

## Documentation

- [`AGENTS.md`](AGENTS.md) — project guide for AI agents
- [`docs/architecture.md`](docs/architecture.md) — five-layer architecture and data flow
- [`docs/adapters.md`](docs/adapters.md) — writing a new adapter
- [`docs/mcp-integration.md`](docs/mcp-integration.md) — MCP server integration
- [`BENCHMARK.md`](BENCHMARK.md) — benchmark reproduction
- [`examples/mcp/`](examples/mcp/) — ready-to-use MCP client configs

## FAQ

<details>
<summary><strong>How is this different from a vector database / RAG?</strong></summary>

Vector RAG retrieves text chunks by embedding similarity. It cannot tell you whether a fact is current, has been superseded, or contradicts another fact in the corpus. agent-knowledge compiles raw input into structured Claims, merges claims per entity into a Compiled Truth with explicit `active` / `superseded` / `disputed` status, and keeps an append-only timeline — so the agent retrieves *the current answer plus its lineage* in one call.

You can still bring a vector model in as an optional reranking signal; it is not required.

</details>

<details>
<summary><strong>Do I need a vector database (Chroma, Qdrant, Pinecone, …)?</strong></summary>

No. Default retrieval is Exact Match + BM25 + Knowledge Graph + weighted RRF + TF-IDF reranker — all pure Python, all local. agent-knowledge has only **one** runtime dependency (`PyYAML`). On LongMemEval-S the vector-free path reaches **96.6% R@5**, matching strong embedding-based baselines.

</details>

<details>
<summary><strong>How is this different from mem0 / Letta / Zep / LangChain memory?</strong></summary>

Most memory frameworks store fragments (messages, summaries, or embeddings) and retrieve by similarity. agent-knowledge is built around **knowledge compilation** instead: every claim has source provenance, every entity has a Compiled Truth, and the timeline is rewritten holistically when new evidence arrives. The output is *traceable structured knowledge*, not a bag of remembered turns.

Different optimization target — both are valid; pick by what you need.

</details>

<details>
<summary><strong>Can I plug it into Claude Code / Cursor / Codex?</strong></summary>

Yes — it ships as an MCP server (stdio JSON-RPC 2.0). Copy-paste configs are in [`examples/mcp/`](examples/mcp/):
- Claude Code: [`examples/mcp/claude-code.json`](examples/mcp/claude-code.json)
- Cursor: [`examples/mcp/cursor.json`](examples/mcp/cursor.json)
- Codex: [`examples/mcp/codex.toml`](examples/mcp/codex.toml)

Start the server with `compiled-memory-mcp` (default vault at `~/.agent-knowledge/vault`) or `ak mcp /path/to/your/vault`.

</details>

<details>
<summary><strong>Where does the data live? Is anything sent to the cloud?</strong></summary>

Nothing leaves your machine by default. The vault is a directory of human-readable YAML files plus a SQLite event index — both git-friendly. No telemetry, no calls home, no required API keys.

</details>

<details>
<summary><strong>What is UMSF?</strong></summary>

Universal Memory Source Format — a small JSON schema that unifies how conversations, tool traces, decisions, and file changes are submitted to the vault. Eight event types, seven source types. It is what lets adapters for different agents (Claude Code, Codex, custom hermes/openclaw slots, …) share the same ingest pipeline. See [`docs/architecture.md`](docs/architecture.md).

</details>

## Citation

If you use agent-knowledge in research or a publication, please cite:

```bibtex
@software{agent_knowledge_2026,
  author  = {Yu, Chengxin},
  title   = {agent-knowledge: long-term memory and knowledge compilation for AI agents},
  year    = {2026},
  url     = {https://github.com/yucx-go/agent-knowledge},
  version = {0.3.1}
}
```

A [`CITATION.cff`](CITATION.cff) is included for GitHub's automatic citation widget.

## License

MIT — see [`LICENSE`](LICENSE).
