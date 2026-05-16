"""MCP tool definitions + handlers.

Each handler is a free function ``tool_x(server, args) -> dict`` that
takes the :class:`MCPServer` instance (for vault/hook access) and the
caller's arguments. The MCPServer dispatches to these via
:func:`handle_call`.

Adding a new tool
-----------------
1. Append a schema dict to :data:`TOOLS` (name, description, inputSchema).
2. Write a handler ``def tool_<name>(server, args) -> dict``.
3. Register it in the ``_HANDLERS`` map at the bottom of this file.

Tool handlers MUST return MCP content responses::

    {"content": [{"type": "text", "text": "..."}], "isError": False?}
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .server import MCPServer


# ── Tool Definitions (MCP schema) ──

TOOLS: list[dict] = [
    {
        "name": "ak_query",
        "description": "Search the knowledge vault for relevant information.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query text.",
                },
                "top_k": {
                    "type": "integer",
                    "description": "Maximum number of results to return (default: 5).",
                    "default": 5,
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "ak_ingest",
        "description": "Ingest text into the knowledge vault. Extracts claims and entities.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "Text content to ingest.",
                },
                "title": {
                    "type": "string",
                    "description": "Title for the ingested source (optional).",
                    "default": "",
                },
                "source_type": {
                    "type": "string",
                    "description": "Type of source: text, file, url, conversation.",
                    "default": "text",
                },
            },
            "required": ["text"],
        },
    },
    {
        "name": "ak_ingest_umsf",
        "description": (
            "Ingest a complete UMSFDocument. Use this for multi-event "
            "submissions (full conversations, tool traces, etc.). Each "
            "event is fanned out to matching hooks AND the whole document "
            "is rendered into a single Source via the standard pipeline."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "doc": {
                    "type": "object",
                    "description": "A UMSFDocument as a JSON-serializable dict (see core/umsf.py for schema).",
                },
            },
            "required": ["doc"],
        },
    },
    {
        "name": "ak_stats",
        "description": "Get vault statistics: source count, entity count, etc.",
        "inputSchema": {
            "type": "object",
            "properties": {},
        },
    },
    {
        "name": "ak_dream",
        "description": "Run a dream cycle (memory consolidation) over recent sources.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "since_hours": {
                    "type": "number",
                    "description": "Process sources from last N hours (default: 24).",
                    "default": 24.0,
                },
            },
        },
    },
    {
        "name": "ak_hook_fire",
        "description": (
            "Fire a single UMSF event for real-time knowledge capture. "
            "The event is wrapped in a one-event UMSFDocument and dispatched "
            "through the hook system. For multi-event submissions, use ak_ingest_umsf."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "type": {
                    "type": "string",
                    "description": "UMSF event type: message, tool_use, tool_result, decision, file_change, error, session_start, session_end.",
                },
                "content": {"type": "string", "description": "Message body / decision text / error description."},
                "role": {"type": "string", "description": "For 'message' events: user / assistant / system / tool."},
                "name": {"type": "string", "description": "For 'tool_use' events: tool name. For 'error' events: failing tool."},
                "args": {"type": "object", "description": "For 'tool_use' events: structured tool arguments."},
                "result": {"type": "string", "description": "For 'tool_result' events: stringified tool output."},
                "error": {"type": "string", "description": "For 'error' events: error message."},
                "path": {"type": "string", "description": "For 'file_change' events: filesystem path."},
                "metadata": {"type": "object", "description": "Free-form metadata (e.g. session_end events may pass {messages: [...]})."},
                "session_id": {"type": "string", "description": "Session identifier (optional).", "default": ""},
                "agent": {"type": "string", "description": "Agent identifier (claude-code / codex / hermes / external).", "default": "external"},
            },
            "required": ["type"],
        },
    },
    {
        "name": "ak_hook_list",
        "description": "List all registered hooks.",
        "inputSchema": {"type": "object", "properties": {}},
    },
    {
        "name": "ak_hook_stats",
        "description": "Get hook fire statistics.",
        "inputSchema": {"type": "object", "properties": {}},
    },
]


# ── Tool Handlers ──


def tool_query(server: "MCPServer", args: dict) -> dict:
    from ...search.engine import SearchEngine

    query = args.get("query", "")
    top_k = args.get("top_k", 5)

    if not query:
        return _err("query is required")

    engine = SearchEngine(server.vault)
    results = engine.search(query, top_k=top_k)

    if not results:
        return _ok("No results found.")

    lines = []
    for i, r in enumerate(results, 1):
        lines.append(f"[{i}] ({r.page_type}) {r.title} [score: {r.score}]")
        lines.append(f"    {r.snippet}")
    return _ok("\n".join(lines))


def tool_ingest(server: "MCPServer", args: dict) -> dict:
    from ...core.compiler import Compiler

    text = args.get("text", "")
    if not text:
        return _err("text is required")

    title = args.get("title", "")
    source_type = args.get("source_type", "text")

    compiler = Compiler(server.vault)
    source = compiler.ingest(text, title=title, source_type=source_type)

    return _ok(
        f"Ingested: {source.title}\n"
        f"Source ID: {source.id}\n"
        f"Claims: {len(source.claims_extracted)}\n"
        f"Entities: {len(source.entities_extracted)}"
    )


def tool_ingest_umsf(server: "MCPServer", args: dict) -> dict:
    from ...core.umsf import UMSFDocument, UMSFValidationError

    raw = args.get("doc")
    if not isinstance(raw, dict):
        return _err("'doc' must be a UMSFDocument dict")

    try:
        doc = UMSFDocument.from_dict(raw)
    except Exception as e:
        return _err(f"failed to parse doc: {e}")

    manager = server._get_hook_manager()
    try:
        results = manager.fire(doc)
    except UMSFValidationError as e:
        return _err(f"invalid UMSFDocument: {e}")

    total_claims = sum(len(r.claims) for r in results)
    total_entities = sum(len(r.entities_found) for r in results)
    source_ids = [r.source_id for r in results if r.source_id]

    return _ok(
        f"Ingested UMSFDocument: {doc.agent} / {doc.source_type}\n"
        f"Events: {len(doc.events)}\n"
        f"Hooks matched: {len(results)}\n"
        f"Claims extracted: {total_claims}\n"
        f"Entities found: {total_entities}\n"
        f"Source IDs: {', '.join(source_ids) or 'none'}"
    )


def tool_stats(server: "MCPServer", args: dict) -> dict:
    if not server.vault.is_initialized:
        return _err("vault not initialized")

    stats = server.vault.stats()
    lines = [f"{k}: {v}" for k, v in stats.items()]
    return _ok("\n".join(lines))


def tool_dream(server: "MCPServer", args: dict) -> dict:
    from ...dreaming.cycle import DreamCycle

    since_hours = args.get("since_hours", 24.0)
    cycle = DreamCycle(server.vault)
    report = cycle.run(since_hours=since_hours)

    return _ok(
        f"Dream Report:\n"
        f"  Light: {report.light_candidates} candidates\n"
        f"  REM: {report.rem_clusters} clusters\n"
        f"  Deep: {report.deep_promoted} promoted / {report.deep_discarded} discarded\n"
        f"  Time: {report.duration_seconds}s"
    )


def tool_hook_fire(server: "MCPServer", args: dict) -> dict:
    """Fire a single UMSF event by wrapping it in a one-event document."""
    from ...core.umsf import UMSFDocument, UMSFEvent

    event_type = args.get("type", "")
    if not event_type:
        return _err("'type' is required")

    event = UMSFEvent(
        type=event_type,
        ts=time.time(),
        role=args.get("role", ""),
        content=args.get("content", ""),
        name=args.get("name", ""),
        args=args.get("args", {}) or {},
        result=args.get("result", ""),
        error=args.get("error", ""),
        path=args.get("path", ""),
        metadata=args.get("metadata", {}) or {},
    )

    source_type_map = {
        "decision": "decision",
        "file_change": "file_change",
        "tool_use": "tool_trace",
        "tool_result": "tool_trace",
        "error": "tool_trace",
    }
    source_type = source_type_map.get(event_type, "conversation")

    doc = UMSFDocument(
        agent=args.get("agent", "external") or "external",
        source_type=source_type,
        session_id=args.get("session_id", ""),
        occurred_at=event.ts,
        events=[event],
    )

    manager = server._get_hook_manager()
    results = manager.fire(doc)

    total_claims = sum(len(r.claims) for r in results)
    total_entities = sum(len(r.entities_found) for r in results)
    source_ids = [r.source_id for r in results if r.source_id]

    return _ok(
        f"Fired event: {event_type}\n"
        f"Hooks matched: {len(results)}\n"
        f"Claims extracted: {total_claims}\n"
        f"Entities found: {total_entities}\n"
        f"Source IDs: {', '.join(source_ids) or 'none'}"
    )


def tool_hook_list(server: "MCPServer", args: dict) -> dict:
    manager = server._get_hook_manager()
    hooks = manager.list_hooks()
    lines = [f"{h['name']}: {', '.join(h['event_types'])}" for h in hooks]
    return _ok("\n".join(lines) or "No hooks registered.")


def tool_hook_stats(server: "MCPServer", args: dict) -> dict:
    manager = server._get_hook_manager()
    stats = manager.stats()
    lines = [f"{name}: {count}" for name, count in stats.items()]
    return _ok("\n".join(lines) or "No hooks fired yet.")


# ── Dispatch ──

# Map tool name → handler function. Server._handle_tools_call() looks here.
_HANDLERS = {
    "ak_query": tool_query,
    "ak_ingest": tool_ingest,
    "ak_ingest_umsf": tool_ingest_umsf,
    "ak_stats": tool_stats,
    "ak_dream": tool_dream,
    "ak_hook_fire": tool_hook_fire,
    "ak_hook_list": tool_hook_list,
    "ak_hook_stats": tool_hook_stats,
}


def handle_list() -> dict:
    """MCP ``tools/list`` response payload."""
    return {"tools": TOOLS}


def handle_call(server: "MCPServer", params: dict) -> dict:
    """MCP ``tools/call`` dispatch — routes to the named handler.

    Raises :class:`server._RPCError` for unknown tool names so the
    JSON-RPC layer can return a proper -32602 error.
    """
    from .server import _RPCError

    name = params.get("name", "")
    arguments = params.get("arguments", {})
    handler = _HANDLERS.get(name)
    if not handler:
        raise _RPCError(-32602, f"Unknown tool: {name}")
    return handler(server, arguments)


# ── Response helpers ──


def _ok(text: str) -> dict:
    """Standard success response — single text content block."""
    return {"content": [{"type": "text", "text": text}]}


def _err(message: str) -> dict:
    """Standard error response — sets ``isError`` flag."""
    return {"content": [{"type": "text", "text": f"Error: {message}"}], "isError": True}
