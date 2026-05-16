# Changelog

## [0.3.0] - 2026-05-16

### Added

**UMSF (Universal Memory Source Format)**
- `core/umsf.py` — schema, validator, renderer, `ingest_umsf()` helper
- 8 event types: `message`, `tool_use`, `tool_result`, `decision`,
  `file_change`, `error`, `session_start`, `session_end`
- 7 source types: `conversation`, `curated_memory`, `skill_artifact`,
  `tool_trace`, `decision`, `file_change`, `document`
- `Source.umsf` audit-trail field on every UMSF-ingested Source
- `EventIndex` (SQLite at `vault/.ak-events.db`), indexed by
  type/session/agent/timestamp; populated by `ingest_umsf()`

**Pull-adapter framework**
- `adapters/base.py` — `BasePullAdapter` ABC with `discover()` /
  `fetch()`; base `pull()` handles UMSF ingestion, fingerprint dedup,
  dry-run, error collection, state persistence
- `adapters/markdown_memory.py` — scans `~/.<agent>/` for `MEMORY.md` /
  `USER.md` / `CLAUDE.md` / `AGENTS.md`. Built-in slots: `claude-code`,
  `codex`, plus two generic slots (`hermes`, `openclaw`)
- `adapters/claude_code.py` — parses `~/.claude/projects/**/*.jsonl`
  transcripts into UMSF
- `ak pull` CLI flags: `--agent / --path / --force / --dry-run /
  --no-jsonl / --projects / --since-days / --limit`

**MCP server (subpackage)**
- `adapters/mcp/` split into `server.py` / `tools.py` / `resources.py`
  / `prompts.py`
- New `ak_ingest_umsf` MCP tool for multi-event submissions

**Entity extraction (reverse-filter)**
- `core/entity_extractor.py` — wide-net candidates + stopword filter
  (~300 CN, ~120 EN, ~50 CJK particles)
- Typed entities: `person`, `org`, `place`, `document`, `event`,
  `time`, `money`, `quantity`, `version`, `identifier`

**CLI**
- `_read_text_tolerant()` — UTF-8 → UTF-8-SIG → system locale fallback;
  fixes Windows zh-CN (CP936) markdown ingestion

### Changed (BREAKING)

- `HookEvent` removed; hooks now use `UMSFEvent`:
  - `Hook.process(event, vault, compiler)` →
    `Hook.process(event, session_id, vault, compiler)`
  - `HookManager.fire(event)` → `HookManager.fire(doc: UMSFDocument)`
  - `ak_hook_fire` schema: `{event_type, data, session_id}` →
    flat `{type, content, role, name, args, ...}`
- `agent_knowledge.adapters.mcp_server` →
  `agent_knowledge.adapters.mcp`
- `Compiler._HEURISTIC_PATTERNS` removed (moved to `entity_extractor`)
- Claim → entity linking: earliest-mention wins
- `vault/.ak-pull-state.json` schema namespaced by adapter name;
  v0.2.0 state files are ignored and recreated on first `ak pull`

### Fixed

- Windows zh-CN encoding failures in `ak ingest --file` /
  `ak batch-ingest`
- Search engine: fact-extractor child-doc dedup correctly maps
  `#fact_` IDs back to parent source

### Migration from v0.2.0

- Existing Sources work unchanged. New UMSF-ingested Sources
  additionally carry `Source.umsf`.
- `.ak-pull-state.json` is re-created (namespaced) on first
  `ak pull` — no data loss, just a one-time re-pull.
- `HookEvent` imports break — switch to `UMSFEvent` from
  `agent_knowledge.core.umsf`.
- `agent_knowledge.adapters.mcp_server` imports break — switch to
  `agent_knowledge.adapters.mcp`.

## [0.2.0] - 2026-05-14

### Added
- MCP Server (JSON-RPC 2.0 over stdio): `ak_query`, `ak_ingest`,
  `ak_stats`, `ak_dream`, `ak_hook_fire`, `ak_hook_list`,
  `ak_hook_stats`
- MCP Resources: browse entities and sources via `resources/list` and
  `resources/read`
- MCP Prompts: `knowledge_summary`, `contradiction_check`
- Auto-capture Hook system: 7 built-in hooks (`message`, `tool_use`,
  `response`, `error`, `session_end`, `decision`, `file_change`)
- File watcher: `ak hooks --watch DIR`
- LongMemEval-S benchmark runner with `--sample N` and `--workers N`
- PyPI trusted publishing
- CI matrix: Python 3.10–3.13 on ubuntu-latest

## [0.1.0] - 2026-05-13

### Added
- Knowledge compilation: Claims + Evidence + Compiled Truth + Timeline
- 4-path search: Exact + BM25 + Graph + Reranker with weighted RRF
- Dream Cycle: Light → REM → Deep, 6-dim scoring
- Polarity-based contradiction detection
- Query rewriter (keyword + entity variants)
- Memory cleaner / consolidator (Jaccard + durability-aware
  supersession)
- CLI: `init`, `ingest`, `query`, `stats`, `dream`, `batch-ingest`,
  `lint`
- Synthetic smoke benchmark (~25 QA pairs)
