# AGENTS.md

Project guide for AI agents (Claude Code, Codex, Cursor, custom agents).
Human readers should start at [`README.md`](README.md).

## What this project is

`agent-knowledge` is **long-term memory and a knowledge base for AI agents**:
raw conversations and documents → Claim + Evidence extraction → merged into
Compiled Truth → append-only timeline. Every fact is traceable to source,
timestamp, and confidence, with contradiction detection on top.

Position: **it is *compiled* memory — not RAG, not a KV preference cache.**

## When you (an agent) should use it

- The user asks *"why did we decide X back then?"* — needs timeline and
  decision context
- Multiple sources about the same entity need to be merged and reconciled
- Facts need to persist across sessions (user preferences, project
  conventions, prior decisions)
- The answer must be explainable with source provenance, not just
  "vector similarity"

Skip it for:

- One-shot retrieval (plain RAG is lighter)
- Real-time event streams (this is the knowledge layer, not an event bus)

## Quickest path to use

### Option 1: MCP server (recommended for agent integration)

```json
{
  "mcpServers": {
    "agent-knowledge": {
      "command": "ak",
      "args": ["mcp", "/absolute/path/to/vault"]
    }
  }
}
```

Drop this into `.claude/settings.json`, `.cursor/mcp.json`, or any
MCP-aware client config. Full tool list is in
[`docs/mcp-integration.md`](docs/mcp-integration.md).

### Option 2: CLI

```bash
pip install agent-knowledge
ak init   /path/to/vault
ak ingest /path/to/vault --file notes.md
ak query  /path/to/vault "why did we pick React?"
```

### Option 3: Python API

```python
from agent_knowledge import Vault, Compiler, SearchEngine

vault = Vault("/path/to/vault")
vault.init()

compiler = Compiler(vault)
compiler.ingest("We picked React over Vue.", title="Frontend decision")

engine = SearchEngine(vault)
for hit in engine.search("why React?", top_k=5):
    print(hit.title, hit.score)
```

## MCP tools cheat-sheet

| Tool | Purpose |
|------|---------|
| `ak_query` | Search the vault; returns hits with source provenance |
| `ak_ingest` | Ingest a single text snippet |
| `ak_ingest_umsf` | Ingest a full UMSF document (multi-event) |
| `ak_hook_fire` | Capture a real-time event (message / tool call / edit / ...) |
| `ak_stats` | Current vault counts |
| `ak_dream` | Run an offline consolidation cycle (merge / dedup / disambiguate) |

Call `ak_stats` first if you are unsure of the vault state; if you don't
know the vault path, check `args[1]` in the MCP client config.

## Repo layout

```
src/agent_knowledge/
  core/        # Vault, Source, Entity, UMSF definitions
  search/      # Exact + BM25 + Graph + RRF + Reranker
  dreaming/    # Light / REM / Deep three-phase consolidation
  hooks/       # Event capture
  adapters/    # markdown_memory / claude_code / mcp / ...
  cli.py
docs/          # Architecture, adapters, MCP integration
benchmark/     # LongMemEval-S + synthetic smoke
tests/
```

## Conventions when modifying this repo

- **Zero external services** is a core promise. Put optional capabilities
  in `[project.optional-dependencies]`; do not pollute the base install.
- Cross the data contract via `core/umsf.py` only. Adapters must not
  reach into internal data structures.
- Docs favor information density. Avoid narrative process descriptions
  and redundant comparisons (mirror the existing docs style).
- Tests use synthetic data. **Never** commit real business IDs, URLs,
  personal names, or proprietary terminology.
- The public API is whatever `src/agent_knowledge/__init__.py` exposes
  in `__all__`.

## Where to look for what

| To do this | Look here |
|-----------|-----------|
| Write a new adapter | [`docs/adapters.md`](docs/adapters.md) + `src/agent_knowledge/adapters/base.py` |
| Change retrieval scoring | `src/agent_knowledge/search/reranker.py`, `engine.py` |
| Change consolidation rules | `src/agent_knowledge/dreaming/` |
| Run the benchmark | `benchmark/run_benchmark.py` + [`BENCHMARK.md`](BENCHMARK.md) |
| Tweak MCP tool schemas | `src/agent_knowledge/adapters/mcp/tools.py` |

## Citation

For academic use, see [`CITATION.cff`](CITATION.cff).
