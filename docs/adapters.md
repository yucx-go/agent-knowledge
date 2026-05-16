# Writing Adapters

The boundary between an external system and the knowledge core is a
single format: **UMSF** (Universal Memory Source Format). Every adapter
speaks UMSF; the core only reads UMSF.

| Flavor | When to use | Implementation |
|--------|-------------|----------------|
| **Pull adapter** | The source has its own storage (files, SQLite, API). You read it on a schedule. | Subclass [`BasePullAdapter`](../src/agent_knowledge/adapters/base.py) — implement `discover()` and `fetch()`. |
| **Push adapter** | The source calls you over [MCP](https://modelcontextprotocol.io/). | Use the existing [MCP server](../src/agent_knowledge/adapters/mcp/). No new code on your side. |

This guide focuses on pull adapters.

## The UMSF target format

```python
UMSFDocument(
    agent="claude-code",         # who produced this data
    source_type="conversation",  # conversation | curated_memory | tool_trace
                                 # | decision | file_change | skill_artifact
                                 # | document
    session_id="...",            # opaque grouping key
    title="What I did Tuesday",
    occurred_at=1715763200.0,    # unix timestamp
    events=[
        UMSFEvent(type="message", role="user", content="..."),
        UMSFEvent(type="tool_use", name="bash", args={...}),
        UMSFEvent(type="tool_result", result="..."),
        UMSFEvent(type="decision", content="...", rationale="..."),
        # also: file_change, error, session_start, session_end
    ],
    context={"path": "...", "cwd": "...", "git_branch": "main"},
)
```

For prose sources (a `MEMORY.md` file, a wiki page), skip `events` and
put the body in `context["body"]`:

```python
UMSFDocument(
    agent="claude-code",
    source_type="curated_memory",
    title="CLAUDE.md",
    context={"body": "## decisions\n- chose Postgres", "path": "..."},
)
```

`umsf.render()` turns events into a markdown body the Compiler can
extract claims from. You don't format text yourself.

## Minimum adapter

```python
from agent_knowledge.adapters.base import BasePullAdapter, DiscoveredItem
from agent_knowledge.core.umsf import UMSFDocument

class MyAdapter(BasePullAdapter):
    name = "my-source"  # state-file key; must be stable

    def discover(self, **opts) -> list[DiscoveredItem]:
        ...   # enumerate what's available to ingest

    def fetch(self, item: DiscoveredItem) -> UMSFDocument | None:
        ...   # convert one item; return None to skip silently
```

`name` is critical — it keys the per-adapter namespace in
`.ak-pull-state.json`. Pick something stable.

## Worked example: Obsidian adapter

```python
@dataclass
class DiscoveredNote(DiscoveredItem):
    path: Optional[Path] = None
    relative: str = ""
    size_bytes: int = 0
    mtime: float = 0.0

    def __post_init__(self) -> None:
        if self.path is None:
            return
        if not self.id:          self.id = str(self.path)
        if not self.fingerprint: self.fingerprint = f"{self.size_bytes}:{self.mtime}"
        if not self.label:       self.label = f"[obsidian] {self.relative}"
        self.metadata.setdefault("relative", self.relative)


class ObsidianAdapter(BasePullAdapter):
    name = "obsidian"

    def __init__(self, vault, obsidian_root: Path):
        super().__init__(vault)
        self.obsidian_root = Path(obsidian_root).expanduser().resolve()

    def discover(self, **opts) -> list[DiscoveredNote]:
        if not self.obsidian_root.exists():
            return []
        results = []
        for p in self.obsidian_root.rglob("*.md"):
            if not p.is_file():
                continue
            try:
                st = p.stat()
            except OSError:
                continue
            results.append(DiscoveredNote(
                path=p,
                relative=str(p.relative_to(self.obsidian_root)),
                size_bytes=st.st_size,
                mtime=st.st_mtime,
            ))
        return results

    def fetch(self, item: DiscoveredItem) -> Optional[UMSFDocument]:
        if not isinstance(item, DiscoveredNote) or item.path is None:
            return None
        text = item.path.read_text(encoding="utf-8")
        if not text.strip():
            return None
        return UMSFDocument(
            agent="obsidian",
            source_type="curated_memory",
            title=item.label,
            context={"body": text, "path": str(item.path),
                     "relative": item.relative},
        )
```

Wire into the CLI by appending to `runs` in
[`cli.py:cmd_pull`](../src/agent_knowledge/cli.py), or call directly:

```python
adapter = ObsidianAdapter(vault, obsidian_root=Path("~/notes"))
report = adapter.pull()
```

## ID vs fingerprint

- `id` identifies a thing across time (path, primary key, UUID).
  Used as the dedup key. Must be stable.
- `fingerprint` identifies a *version* of that thing (size+mtime, etag,
  content hash). Must change when content changes.

Putting mtime in `id` means every edit creates a new "item" and never
deduplicates.

## What the base class handles

| Feature | How |
|---------|-----|
| UMSF ingestion | `pull()` calls `ingest_umsf(compiler, doc)` for every fetched doc |
| Audit trail | Every Source carries the full UMSF dict on `Source.umsf` |
| Event index | UMSF events mirrored into SQLite `EventIndex` |
| Hooks | Each event dispatched through registered hooks |
| Dedup | `{adapter_name: {item_id: {fingerprint, ...}}}` in `.ak-pull-state.json` |
| `--force` | Re-ingest everything, bypass fingerprint check |
| `--dry-run` | `discover()` only, no fetch / ingest / state writes |
| Errors | Exceptions in `fetch()` caught, recorded in `report.errors`, others continue |

## UMSF event reference

| Type | When | Key fields |
|------|------|------------|
| `message` | User or assistant utterance | `role`, `content` |
| `tool_use` | Agent invoked a tool | `name`, `args` |
| `tool_result` | Tool returned data or error | `result`, `error`, `metadata.tool_use_id` |
| `decision` | Explicit user/agent decision | `content`, `rationale` |
| `file_change` | Filesystem write/edit/delete | `path`, optional `before` / `after` |
| `error` | Unhandled exception | `error`, `name`, `metadata.context` |
| `session_start` | Boundary marker | — |
| `session_end` | Boundary marker | optional `metadata.messages` |

## Reference adapters

| Adapter | Lines | What it covers |
|---------|------:|----------------|
| [`markdown_memory.py`](../src/agent_knowledge/adapters/markdown_memory.py) | ~220 | `MEMORY.md/USER.md/CLAUDE.md/AGENTS.md` under `~/.claude/`, `~/.codex/`, custom `~/.<agent>/` paths |
| [`claude_code.py`](../src/agent_knowledge/adapters/claude_code.py) | ~280 | JSONL transcripts under `~/.claude/projects/` |
| [`mcp/`](../src/agent_knowledge/adapters/mcp/) | ~890 (5 files) | MCP push server |

## Pitfalls

- **Storing huge tool outputs verbatim** — the Compiler truncates source
  body to 10K chars anyway. Cap per-event content yourself (`claude_code.py`
  uses 1500 chars).
- **`agent=""`** — agent flows into Source.umsf and EventIndex; empty
  agent means you can't filter by source later.
- **`id` includes mtime** — every edit becomes a new "item" and dedup
  never fires. Use stable `id`, volatile `fingerprint`.
- **Hand-rendering markdown into events** — UMSF events are
  *structured*. `tool_use.args` is a dict, not a pre-formatted string.
  Let `umsf.render()` do the formatting.
- **Skipping `__post_init__`** — the base requires `id` and
  `fingerprint`. Subclasses must fill them in.

## Testing

Cover at minimum: ingestion happens, dedup works, `--force` re-ingests,
empty/invalid files are handled gracefully. See
[`tests/test_markdown_memory.py`](../tests/test_markdown_memory.py) and
[`tests/test_claude_code.py`](../tests/test_claude_code.py) for full
patterns.
