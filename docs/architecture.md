# Architecture

External systems → **UMSF** (universal boundary format) → **Compiler**
distills claims/entities → **Vault** (YAML + SQLite event index) →
**Search** answers queries with provenance.

For the contributor's perspective on writing a new adapter, see
[adapters.md](adapters.md).

## Five layers

```
┌─────────────────────────────────────────────────────────────┐
│  Adapter Layer  (Pull + Push, all speak UMSF)               │
│                                                             │
│  Pull adapters (BasePullAdapter)         Push adapters       │
│  ─────────────────────────────           ─────────────       │
│  markdown_memory                         MCP server         │
│    Claude Code / Codex / custom          ak_ingest_umsf     │
│    MEMORY.md / USER.md / CLAUDE.md       ak_hook_fire       │
│  claude_code                             ak_query / …        │
│    ~/.claude/projects/*.jsonl            (any MCP client)   │
│  (your adapter here — 80 lines)                             │
└──────────────────────────┬──────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  UMSF Boundary  (core/umsf.py)                              │
│                                                             │
│    UMSFDocument                                             │
│      agent: claude-code | codex | custom | …                │
│      source_type: conversation | curated_memory | …         │
│      events[]: message | tool_use | decision | …            │
└──────────────────────────┬──────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  Knowledge Layer                                            │
│                                                             │
│   Compiler (core/compiler.py)                               │
│     ├── Extract claims (section-aware + signal regexes)     │
│     ├── Extract entities (entity_extractor.py)              │
│     │     reverse-filter: wide-net candidates → stopwords   │
│     ├── Link claims ↔ entities (earliest-mention wins)      │
│     ├── Update Compiled Truth                               │
│     └── Append Timeline                                     │
│                                                             │
│   Hook System (hooks/, UMSF-native)                         │
│     └── 7 built-in hooks: message/tool/response/error/      │
│         session/decision/file_change                        │
│                                                             │
│   Dream Cycle (dreaming/cycle.py)                           │
│     └── Light → REM → Deep, 6-dim scoring → promote         │
└────────────┬───────────────────────────────┬────────────────┘
             ▼                               ▼
┌────────────────────────────┐  ┌──────────────────────────────┐
│  Vault (YAML)              │  │  EventIndex (SQLite)         │
│  one file per source /     │  │  one row per UMSF event,     │
│  entity / concept / …      │  │  indexed by type / session / │
│                            │  │  agent / timestamp           │
└────────────┬───────────────┘  └──────────────────────────────┘
             ▼
┌─────────────────────────────────────────────────────────────┐
│  Search Layer (search/)                                     │
│                                                             │
│   BM25Index    Exact substring index    GraphIndex          │
│       │              │                       │              │
│       └─── Query Rewriter (entity hints, paraphrases) ──┐   │
│                                                         ▼   │
│                                            Weighted RRF     │
│                                            fusion (k=60)    │
│                                                  │          │
│                                  TF-IDF 7-signal Reranker   │
│                                                                │
│   Optional: VectorIndex (sentence-transformers or HTTP API)  │
│   Optional: FactExtractor (by-the-way mention indexing)      │
└─────────────────────────────────────────────────────────────┘
```

## Ingestion data flow

```
Adapter (or MCP client)
  │
  │ produces a UMSFDocument
  ▼
ingest_umsf(compiler, doc)                       ── core/umsf.py
  ├─► validate(doc)                              schema + non-empty
  ├─► render(doc) → markdown body                events → ## sections
  ├─► compiler.ingest(text, source_type, …)      ── core/compiler.py
  │     ├─► _extract_claims()                    section + signal regex
  │     ├─► _extract_entities()                  ── entity_extractor.py
  │     │     1. typed extractors (email/URL/date/money/version/ID)
  │     │     2. wide-net noun-shaped candidates
  │     │     3. negative filter (~300 CN + ~120 EN stopwords)
  │     │     4. suffix-based type tagging (公司→org, 报告→document …)
  │     ├─► link claims to entities (earliest mention wins)
  │     ├─► update Entity.compiled_truth (consolidate)
  │     └─► save_source / save_entity to Vault YAML
  ├─► source.umsf = doc.to_dict()                audit trail
  └─► EventIndex.add(events)                     time-ordered store
```

`HookManager.fire(doc)` invokes the same `ingest_umsf` path, then fans
out per-event to specialized hooks (DecisionHook, ToolHook, …) so they
can emit higher-confidence claims for events that match their patterns.

## Dream cycle (offline consolidation)

```
Cron trigger
  │
  ▼
Light                  scan recent sources (since N hours)
  │                    re-extract candidate claims
  ▼
REM                    Jaccard-similarity clustering
  │                    find cross-cluster entities
  ▼
Deep                   6-dim weighted scoring
                       Frequency        (0.24) evidence count + cluster
                       Relevance        (0.30) decision > learning > …
                       Query diversity  (0.15) referenced across contexts
                       Recency          (0.15) durability-aware decay
                       Consolidation    (0.10) already linked to entity
                       Concept richness (0.06) length + cross-cluster
  │
  ├─ score ≥ 0.45 → promote to Entity Compiled Truth
  └─ score <  0.45 → discard
```

## Vault schema

| Path | What | When written |
|------|------|--------------|
| `.ak-schema.yaml` | Vault metadata (version, language) | `ak init` |
| `sources/<id>.yaml` | Raw source + extracted IDs + optional `umsf` audit dict | Every ingestion |
| `entities/<id>.yaml` | Compiled Truth, claims, timeline, foresight, backlinks | Every ingestion that touches the entity |
| `concepts/<id>.yaml` | Cross-entity concept page | Future / manual |
| `syntheses/<id>.yaml` | Cross-source synthesis report | Future / manual |
| `reports/<name>.yaml` | Lint output, contradiction reports | `ak lint` |
| `.ak-pull-state.json` | Per-adapter pull dedup state, namespaced by `adapter.name` | Every `ak pull` |
| `.ak-events.db` | SQLite mirror of UMSF events | Every UMSF ingestion |

## Design decisions

### Single boundary format (UMSF)

Without UMSF, every new adapter chose its own data shape and called
Compiler differently — N adapters × M Compiler entry points. UMSF
collapses that to N+1: each adapter produces `UMSFDocument`, the
Compiler reads only `UMSFDocument`.

Adding Claude Code support was ~280 lines of adapter code and zero core
changes. The same is true for any agent ecosystem that emits a
conversation log.

### YAML as primary store, SQLite as secondary index

YAML wins on inspectability (`cat`, `vim`, git diffs), portability
(only `pyyaml` needed), and tooling (Obsidian / Notion can read it).

It loses on structured queries — "show every `tool_use` event in
session X ordered by timestamp" is expensive against YAML files but
trivial against SQLite. `EventIndex` is that secondary index, written
through `ingest_umsf()` and guarded by try/except so EventIndex
failures never block primary persistence.

YAML scales to ~10K pages; past that, the migration path is a SQLite
read cache for hot paths while keeping YAML as source of truth.

### Reverse-filter entity extraction

Positive rules don't scale across domains — every new field (finance,
medical, legal) brings new vocabulary patterns can't anticipate.
Stopword lists do scale because they're language-level, not
domain-level.

The inversion: extract any noun-shaped span, then filter out the known
non-entities. Empirically this took a finance/Chinese-domain
`AGENTS.md` from 49 entities (all `named`) to 152 entities across 6
typed categories.

### Compiled Truth vs RAG

Traditional RAG re-derives the answer from chunks on every query;
consistency depends on chunk retrieval quality each time.

Compiled Truth pre-computes the "current best understanding" of each
entity at ingest time, merges new claims into the existing structure,
and flags contradictions. Queries read the pre-compiled page directly,
so provenance and confidence are built into the data model rather than
reconstructed from chunks.

### Dream cycle vs real-time consolidation

Real-time consolidation adds latency to every interaction and can't see
cross-source patterns without a global pass anyway. Batch
consolidation lets the REM phase cluster across the full vault and the
Deep phase use global signals (e.g. frequency) for scoring.

Trade-off: new claims aren't immediately promoted. For most workflows
that's fine — most queries are about old knowledge.

### Zero-dep over best-quality at the boundary

We chose not to bundle `jieba` / `spacy` / NER models because:

- `pip install agent-knowledge` is one line and only pulls `pyyaml`.
  Adding tokenizers (4 MB) or NER models (50 MB+ per language) makes
  the install 10–50× larger.
- Reverse-filter stopword segmentation captures ~80% of meaningful
  entities in mixed CJK/English text.
- Users who need more can drop in `.ak-entities.yaml` (custom patterns
  + glossary) or replace `entity_extractor.py`.

LLM-augmented compilation is planned for v0.4.0 as an additional layer
on top of the existing reverse filter.
