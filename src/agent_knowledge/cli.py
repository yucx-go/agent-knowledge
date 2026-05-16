"""CLI entry point: ak init / ak ingest / ak query / ak stats / ak lint"""

from __future__ import annotations

import argparse
import locale
import sys
from pathlib import Path


def _read_text_tolerant(path: Path) -> str:
    """Read a text file, falling back across common encodings.

    Tries in order:
      1. UTF-8 (dominant encoding for code/notes)
      2. UTF-8 with BOM (Excel/Notepad on Windows save like this)
      3. System default (CP936 on zh-CN Windows, CP1252 on en-Windows, …)

    Raises a clear UnicodeDecodeError if none work, naming the file and
    the encodings tried so the user can re-save explicitly.
    """
    # 1. UTF-8 — the case we want to win
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        pass

    # 2. UTF-8 with byte-order mark
    try:
        return path.read_text(encoding="utf-8-sig")
    except UnicodeDecodeError:
        pass

    # 3. System default — only worth trying if it differs from UTF-8
    sys_enc = locale.getpreferredencoding(False)
    if sys_enc.lower() not in ("utf-8", "utf8"):
        try:
            return path.read_text(encoding=sys_enc)
        except UnicodeDecodeError:
            pass

    raise UnicodeDecodeError(
        "utf-8", b"", 0, 1,
        f"Could not decode {path} as UTF-8, UTF-8-SIG, or system encoding "
        f"({sys_enc}). Re-save the file as UTF-8 to ingest it.",
    )


def cmd_init(args: argparse.Namespace) -> None:
    from .core.vault import Vault

    vault = Vault(args.vault_path)
    if vault.is_initialized:
        print(f"Vault already exists at {vault.root}")
        return
    vault.init(lang=args.lang)
    print(f"✅ Vault initialized at {vault.root}")
    stats = vault.stats()
    print(f"   Directories: {', '.join(vault.DIRS)}")


def cmd_ingest(args: argparse.Namespace) -> None:
    from .core.compiler import Compiler
    from .core.vault import Vault

    vault = Vault(args.vault_path)
    compiler = Compiler(vault)

    if args.file:
        p = Path(args.file).expanduser()
        if not p.exists():
            print(f"❌ File not found: {p}", file=sys.stderr)
            sys.exit(1)
        text = _read_text_tolerant(p)
        title = p.stem
        source_type = "file"
        path = str(p)
    elif args.text:
        text = args.text
        title = ""
        source_type = "text"
        path = None
    else:
        print("❌ Provide --file or --text", file=sys.stderr)
        sys.exit(1)

    source = compiler.ingest(text, title=title, source_type=source_type, path=path)
    print(f"✅ Ingested: {source.title}")
    print(f"   Source ID: {source.id}")
    print(f"   Claims extracted: {len(source.claims_extracted)}")
    print(f"   Entities extracted: {len(source.entities_extracted)}")


def cmd_query(args: argparse.Namespace) -> None:
    from .core.vault import Vault
    from .search.engine import SearchEngine

    vault = Vault(args.vault_path)

    embedder = None
    if getattr(args, "semantic", False):
        from .search.embedder import APIEmbedder
        url = getattr(args, "embedding_url", None) or "http://localhost:11434/v1"
        model = getattr(args, "embedding_model", None) or "text-embedding-ada-002"
        embedder = APIEmbedder(base_url=url, model=model)

    engine = SearchEngine(vault, embedder=embedder)

    results = engine.search(args.query, top_k=args.top_k)
    if not results:
        print("No results found.")
        return

    for i, r in enumerate(results, 1):
        print(f"\n[{i}] ({r.page_type}) {r.title}  [score: {r.score}]")
        print(f"    {r.snippet}")


def cmd_stats(args: argparse.Namespace) -> None:
    from .core.vault import Vault

    vault = Vault(args.vault_path)
    if not vault.is_initialized:
        print(f"❌ No vault at {vault.root}", file=sys.stderr)
        sys.exit(1)
    stats = vault.stats()
    print(f"📊 Vault: {vault.root}")
    for k, v in stats.items():
        print(f"   {k}: {v}")


def cmd_dream(args: argparse.Namespace) -> None:
    from .core.vault import Vault
    from .dreaming.cycle import DreamCycle

    vault = Vault(args.vault_path)
    cycle = DreamCycle(vault)

    print(f"💤 Running dream cycle (window: {args.hours}h)...")
    report = cycle.run(since_hours=args.hours)

    print(f"\n🌙 Dream Report:")
    print(f"   Light: {report.light_candidates} candidates extracted")
    print(f"   REM:   {report.rem_clusters} theme clusters")
    print(f"   Deep:  {report.deep_promoted} promoted / {report.deep_discarded} discarded")
    print(f"   Time:  {report.duration_seconds}s")

    if report.details:
        print(f"\n   Top candidates:")
        for d in report.details[:10]:
            marker = "✅" if d["promoted"] else "  "
            print(f"   {marker} [{d['score']:.3f}] {d['claim']}")


def cmd_batch_ingest(args: argparse.Namespace) -> None:
    from pathlib import Path
    from .core.compiler import Compiler
    from .core.vault import Vault

    vault = Vault(args.vault_path)
    compiler = Compiler(vault)

    src_dir = Path(args.source_dir).expanduser()
    if not src_dir.is_dir():
        print(f"❌ Not a directory: {src_dir}", file=sys.stderr)
        sys.exit(1)

    files = sorted(src_dir.glob(args.pattern))
    if not files:
        print(f"❌ No files matching '{args.pattern}' in {src_dir}")
        sys.exit(1)

    print(f"📥 Batch ingesting {len(files)} files from {src_dir}")
    success = 0
    errors = 0
    for f in files:
        try:
            text = _read_text_tolerant(f)
            source = compiler.ingest(
                text, title=f.stem, source_type="file", path=str(f)
            )
            print(f"   ✅ {f.name} → {len(source.claims_extracted)} claims, {len(source.entities_extracted)} entities")
            success += 1
        except Exception as e:
            print(f"   ❌ {f.name}: {e}")
            errors += 1

    print(f"\n📊 Done: {success} ingested, {errors} failed")


def _default_vault_path() -> str:
    """Return ``~/.agent-knowledge/vault`` (created if missing).

    Used when ``ak mcp`` / the ``compiled-memory-mcp`` entry point is
    invoked without an explicit path — gives MCP clients a working
    default without forcing every user to think about where to store
    their knowledge before they can try the server.
    """
    default = Path.home() / ".agent-knowledge" / "vault"
    default.mkdir(parents=True, exist_ok=True)
    return str(default)


def cmd_mcp(args: argparse.Namespace) -> None:
    from .adapters.mcp import run_stdio

    run_stdio(args.vault_path or _default_vault_path())


def mcp_entry() -> None:
    """Console-script entry point for ``compiled-memory-mcp``.

    Bypasses argparse so that MCP clients can spawn the server with a
    single command (``compiled-memory-mcp``) — vault path is taken from
    ``$AGENT_KNOWLEDGE_VAULT`` or the first positional argument, falling
    back to ``~/.agent-knowledge/vault``.
    """
    import os

    from .adapters.mcp import run_stdio

    vault_path = (
        (sys.argv[1] if len(sys.argv) > 1 else None)
        or os.environ.get("AGENT_KNOWLEDGE_VAULT")
        or _default_vault_path()
    )
    run_stdio(vault_path)


def cmd_hooks(args: argparse.Namespace) -> None:
    import json
    import time

    from .core.umsf import UMSFDocument, UMSFEvent
    from .core.vault import Vault
    from .hooks.base import HookManager

    vault = Vault(args.vault_path)
    if not vault.is_initialized:
        print(f"❌ No vault at {vault.root}", file=sys.stderr)
        sys.exit(1)

    manager = HookManager(vault)
    manager.register_defaults()

    if args.fire:
        # Manual fire mode — accepts either a single UMSFEvent (flat) or a
        # full UMSFDocument (under "doc" key).
        try:
            payload = json.loads(args.fire)
        except json.JSONDecodeError as e:
            print(f"❌ Invalid JSON: {e}", file=sys.stderr)
            sys.exit(1)

        if isinstance(payload, dict) and "doc" in payload:
            # Full UMSFDocument
            doc = UMSFDocument.from_dict(payload["doc"])
        else:
            # Single-event shorthand
            event_type = payload.get("type") or payload.get("event_type") or "message"
            event = UMSFEvent(
                type=event_type,
                ts=payload.get("ts", time.time()),
                role=payload.get("role", ""),
                content=payload.get("content", ""),
                name=payload.get("name", ""),
                args=payload.get("args", {}) or {},
                result=payload.get("result", ""),
                error=payload.get("error", ""),
                path=payload.get("path", ""),
                metadata=payload.get("metadata", {}) or {},
            )
            source_type_map = {
                "decision": "decision",
                "file_change": "file_change",
                "tool_use": "tool_trace",
                "tool_result": "tool_trace",
                "error": "tool_trace",
            }
            doc = UMSFDocument(
                agent=payload.get("agent", "cli") or "cli",
                source_type=source_type_map.get(event_type, "conversation"),
                session_id=payload.get("session_id", ""),
                occurred_at=event.ts,
                events=[event],
            )

        results = manager.fire(doc)
        total_claims = sum(len(r.claims) for r in results)
        print(f"✅ Fired UMSFDocument with {len(doc.events)} event(s)")
        print(f"   Agent:            {doc.agent}")
        print(f"   Source type:      {doc.source_type}")
        print(f"   Hooks matched:    {len(results)}")
        print(f"   Claims extracted: {total_claims}")

    elif args.watch:
        # File watch mode
        from .hooks.file_hook import FileWatcher

        watcher = FileWatcher(
            watch_dir=args.watch,
            manager=manager,
            interval=args.interval,
        )
        print(f"👁️  Watching {args.watch} (interval: {args.interval}s)")
        print("   Press Ctrl+C to stop")
        try:
            watcher.run()
        except KeyboardInterrupt:
            watcher.stop()
            print("\n🛑 Stopped.")

    else:
        # List hooks
        hooks = manager.list_hooks()
        print(f"🪝 Registered hooks ({len(hooks)}):")
        for h in hooks:
            print(f"   {h['name']}: {', '.join(h['event_types'])}")


def cmd_pull(args: argparse.Namespace) -> None:
    """Run pull adapters against external agent data.

    Two adapters dispatch in sequence:
      - MarkdownMemoryAdapter — MEMORY.md/USER.md/SOUL.md/CLAUDE.md/AGENTS.md
      - ClaudeCodeAdapter      — JSONL session transcripts under ~/.claude/projects/

    Both run by default. Filter with ``--agent claude-code`` to keep only
    Claude-Code-related sources (both adapters still run, just narrowed).
    Use ``--no-jsonl`` to skip the JSONL adapter entirely.
    """
    from .adapters.claude_code import ClaudeCodeAdapter
    from .adapters.markdown_memory import KNOWN_AGENTS, MarkdownMemoryAdapter
    from .core.vault import Vault

    vault = Vault(args.vault_path)
    if not vault.is_initialized:
        print(f"❌ No vault at {vault.root}", file=sys.stderr)
        sys.exit(1)

    agents = args.agent if args.agent else None
    custom_paths = args.path if args.path else None

    # Build the run plan: list of (adapter, label, options dict)
    runs: list[tuple[object, str, dict]] = []
    runs.append((
        MarkdownMemoryAdapter(vault),
        "markdown-memory",
        {"agents": agents, "custom_paths": custom_paths},
    ))

    cc_filtered_out = bool(agents) and "claude-code" not in agents
    if not args.no_jsonl and not cc_filtered_out:
        runs.append((
            ClaudeCodeAdapter(vault),
            "claude-code",
            {
                "projects": args.projects if args.projects else None,
                "since_days": args.since_days,
                "limit": args.limit,
            },
        ))

    # Dispatch all adapters and print per-adapter reports
    total_ingested = 0
    total_errors = 0
    for adapter, label, opts in runs:
        if args.dry_run:
            files = adapter.discover(**opts)
            print(f"🔍 [{label}] Would pull {len(files)} item(s):")
            for f in files[:30]:
                print(f"   {f.label}  [{f.fingerprint}]")
            if len(files) > 30:
                print(f"   … and {len(files) - 30} more")
            print()
            continue

        report = adapter.pull(force=args.force, **opts)
        total_ingested += len(report.ingested)
        total_errors += len(report.errors)

        print(f"📥 [{label}] {report.duration_seconds:.2f}s")
        print(f"   Discovered: {len(report.discovered)}")
        print(f"   Ingested:   {len(report.ingested)}")
        print(f"   Skipped:    {len(report.skipped)} (unchanged or empty)")
        print(f"   Errors:     {len(report.errors)}")

        if report.ingested and len(report.discovered) <= 20:
            print("\n   Ingested:")
            ingested_set = set(report.ingested)
            for item in report.discovered:
                # The base class uses item.id as the state key; we don't
                # have a direct map back to source_id from item, so we
                # show whichever items weren't in the skipped/error sets.
                if item.id not in report.skipped and not any(
                    e[0] == item.id for e in report.errors
                ):
                    print(f"   ✅ {item.label}")

        if report.errors:
            print("\n   ⚠️  Errors:")
            for item_id, msg in report.errors[:10]:
                print(f"   ❌ {item_id}: {msg}")
        print()

    if not args.dry_run:
        print(f"📊 Total: {total_ingested} ingested, {total_errors} errors")
    elif not runs:
        print("🔍 No adapters configured — markdown-memory was the only default")
        print(f"   Known markdown-memory agents: {', '.join(KNOWN_AGENTS)}")


def cmd_lint(args: argparse.Namespace) -> None:
    from .core.compiler import Compiler
    from .core.vault import Vault

    vault = Vault(args.vault_path)
    compiler = Compiler(vault)

    contradictions = compiler.detect_contradictions()
    stats = vault.stats()

    print(f"🔍 Lint: {vault.root}")
    print(f"   Sources: {stats['sources']}, Entities: {stats['entities']}")
    print(f"   Contradictions: {len(contradictions)}")

    if contradictions:
        print("\n⚠️  Contradictions found:")
        for c in contradictions:
            print(f"   Entity: {c['entity']}")
            print(f"     A: {c['claim_a']['text']} (conf: {c['claim_a']['confidence']})")
            print(f"     B: {c['claim_b']['text']} (conf: {c['claim_b']['confidence']})")
    else:
        print("   ✅ No contradictions detected")


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="ak",
        description="agent-knowledge: AI Agent Knowledge Management System",
    )
    sub = parser.add_subparsers(dest="command")

    # init
    p_init = sub.add_parser("init", help="Initialize a new knowledge vault")
    p_init.add_argument("vault_path", help="Path for the vault")
    p_init.add_argument("--lang", default="zh", choices=["zh", "en"])

    # ingest
    p_ingest = sub.add_parser("ingest", help="Ingest text or file into the vault")
    p_ingest.add_argument("vault_path", help="Vault path")
    p_ingest.add_argument("--file", "-f", help="File to ingest")
    p_ingest.add_argument("--text", "-t", help="Text to ingest")

    # query
    p_query = sub.add_parser("query", help="Search the knowledge vault")
    p_query.add_argument("vault_path", help="Vault path")
    p_query.add_argument("query", help="Search query")
    p_query.add_argument("--top-k", type=int, default=5)
    p_query.add_argument(
        "--semantic", action="store_true",
        help="Enable vector semantic search (requires an OpenAI-compatible embedding API)",
    )
    p_query.add_argument(
        "--embedding-url", default=None,
        help="Embedding API base URL (default: http://localhost:11434/v1)",
    )
    p_query.add_argument(
        "--embedding-model", default=None,
        help="Embedding model name (default: text-embedding-ada-002)",
    )

    # stats
    p_stats = sub.add_parser("stats", help="Show vault statistics")
    p_stats.add_argument("vault_path", help="Vault path")

    # dream
    p_dream = sub.add_parser("dream", help="Run dream cycle (memory consolidation)")
    p_dream.add_argument("vault_path", help="Vault path")
    p_dream.add_argument("--hours", type=float, default=24.0, help="Process sources from last N hours (default: 24)")

    # batch-ingest
    p_batch = sub.add_parser("batch-ingest", help="Batch ingest files from a directory")
    p_batch.add_argument("vault_path", help="Vault path")
    p_batch.add_argument("source_dir", help="Directory containing files to ingest")
    p_batch.add_argument("--pattern", default="*.md", help="Glob pattern (default: *.md)")

    # mcp
    p_mcp = sub.add_parser("mcp", help="Start MCP server (JSON-RPC over stdio)")
    p_mcp.add_argument(
        "vault_path",
        nargs="?",
        default=None,
        help="Vault path (default: ~/.agent-knowledge/vault)",
    )

    # hooks
    p_hooks = sub.add_parser("hooks", help="Manage auto-capture hooks")
    p_hooks.add_argument("vault_path", help="Vault path")
    p_hooks.add_argument("--watch", metavar="DIR", help="Watch directory for file changes")
    p_hooks.add_argument("--interval", type=float, default=5.0, help="Polling interval in seconds (default: 5)")
    p_hooks.add_argument("--fire", metavar="JSON", help="Fire event from JSON string")

    # pull — ingest data from external agent installations
    p_pull = sub.add_parser(
        "pull",
        help=(
            "Pull from agent installations: curated MEMORY.md files "
            "(Claude Code / Codex / custom) AND Claude Code JSONL "
            "transcripts."
        ),
    )
    p_pull.add_argument("vault_path", help="Vault path")
    p_pull.add_argument(
        "--agent", action="append",
        choices=["hermes", "openclaw", "claude-code", "codex"],
        help="Limit to specific agent (repeat for multiple)",
    )
    p_pull.add_argument(
        "--path", action="append", metavar="FILE",
        help="Custom file path to ingest (markdown adapter, repeat for multiple)",
    )
    p_pull.add_argument(
        "--force", action="store_true",
        help="Re-ingest even if fingerprint is unchanged",
    )
    p_pull.add_argument(
        "--dry-run", action="store_true",
        help="Show what would be pulled without ingesting",
    )

    # Claude Code JSONL adapter options
    p_pull.add_argument(
        "--no-jsonl", action="store_true",
        help="Skip the Claude Code JSONL transcript adapter",
    )
    p_pull.add_argument(
        "--projects", action="append", metavar="SUBSTRING",
        help="(claude-code) Filter projects whose slug contains SUBSTRING",
    )
    p_pull.add_argument(
        "--since-days", type=float, default=None, metavar="N",
        help="(claude-code) Only include sessions modified in the last N days",
    )
    p_pull.add_argument(
        "--limit", type=int, default=None, metavar="N",
        help="(claude-code) Cap to N most-recently-modified sessions",
    )

    # lint
    p_lint = sub.add_parser("lint", help="Check vault health")
    p_lint.add_argument("vault_path", help="Vault path")

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(1)

    commands = {
        "init": cmd_init,
        "ingest": cmd_ingest,
        "query": cmd_query,
        "stats": cmd_stats,
        "dream": cmd_dream,
        "batch-ingest": cmd_batch_ingest,
        "pull": cmd_pull,
        "lint": cmd_lint,
        "mcp": cmd_mcp,
        "hooks": cmd_hooks,
    }
    commands[args.command](args)


if __name__ == "__main__":
    main()
