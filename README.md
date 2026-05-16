# agent-knowledge

[![PyPI](https://img.shields.io/pypi/v/agent-knowledge.svg)](https://pypi.org/project/agent-knowledge/)
[![Python](https://img.shields.io/pypi/pyversions/agent-knowledge.svg)](https://pypi.org/project/agent-knowledge/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![CI](https://github.com/yucx-go/agent-knowledge/actions/workflows/ci.yml/badge.svg)](https://github.com/yucx-go/agent-knowledge/actions/workflows/ci.yml)
[![MCP](https://img.shields.io/badge/MCP-compatible-blue.svg)](docs/mcp-integration.md)

**English** | [简体中文](README.zh-CN.md)

Long-term knowledge for AI agents. Conversations, documents, and decisions are auto-compiled into a structured knowledge base with source provenance, confidence tracking, and contradiction detection. Drop-in MCP server — Claude Code, Cursor, Codex, and any MCP-aware client connect with one config line.

## Why

LLM memory today is either flat RAG chunks or a key-value preference cache. Neither answers *"how did we get here?"*.

agent-knowledge adds a **knowledge-compilation layer**: raw material is decomposed into Claims and Evidence, claims about the same entity are merged into a Compiled Truth, and a timeline is kept append-only — rewritten holistically when new evidence arrives. Every fact is traceable to its source, timestamp, and confidence.

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
pip install agent-knowledge

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

## License

MIT
