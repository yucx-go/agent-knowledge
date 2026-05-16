"""Universal Memory Source Format (UMSF).

A normalized envelope for ingestion. Every adapter — Claude Code,
Codex, IM gateway, Obsidian, a personal MEMORY.md collection, a generic
webhook — converts its native format into UMSF, then hands it to the
Compiler. The Compiler/Vault never sees agent-specific details; it just
consumes UMSF documents.

Why a single format
-------------------
- Adding a new agent system = writing one adapter, no core changes.
- Cross-agent queries work because all sources share one schema.
- Hooks, MCP, file watchers, scheduled pulls all emit UMSF, so the
  ingestion pipeline (Compiler.ingest) has only one input type.

Schema (ak/v1)
--------------
A UMSFDocument has document-level metadata (agent, source_type,
session, occurred_at, actors, context) and a list of events.
Events come in a small typed vocabulary covering conversation,
tool use, decisions, and file changes — see UMSFEventType below.

For "curated memory" sources (MEMORY.md, USER.md style files that
are already prose), put the body in context["body"] and leave events
empty. The renderer falls back to passing the body straight through.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


# Schema identifier — bump on incompatible changes
SCHEMA_VERSION = "ak/v1"

# Allowed source types. The Compiler doesn't enforce this list, but
# adapters and downstream tooling key off it for routing.
SOURCE_TYPES = (
    "conversation",          # turn-by-turn dialogue, multiple events
    "curated_memory",        # already-extracted summary (MEMORY.md style)
    "skill_artifact",        # an agentskills.io skill definition
    "tool_trace",            # tool-call timeline only, no chat
    "decision",              # one or more explicit decisions
    "file_change",           # filesystem mutations
    "document",              # plain document (PDF, HTML, markdown note)
)


class UMSFEventType(str, Enum):
    """The closed vocabulary of event types in a UMSFDocument."""

    MESSAGE = "message"          # role-tagged utterance
    TOOL_USE = "tool_use"        # agent invoked a tool
    TOOL_RESULT = "tool_result"  # tool returned data/error
    DECISION = "decision"        # explicit decision with optional rationale
    FILE_CHANGE = "file_change"  # filesystem write / edit / delete
    ERROR = "error"              # unhandled exception or failure
    SESSION_START = "session_start"
    SESSION_END = "session_end"


@dataclass
class Actor:
    """A participant in the source. Roles align with chat conventions."""
    role: str = "user"   # user / assistant / tool / system
    id: str = ""         # opaque identifier; empty if anonymous


@dataclass
class UMSFEvent:
    """A single event in a UMSFDocument.

    Most fields are optional; populate the ones relevant to ``type``.
    Free-form metadata (e.g. tool_use_id linking tool_result back to
    its tool_use, file diff hunks) goes in ``metadata``.
    """
    type: str = UMSFEventType.MESSAGE.value
    ts: float = field(default_factory=time.time)
    role: str = ""           # message events
    content: str = ""        # message body / decision text / error description
    name: str = ""           # tool_use: tool name; file_change: action label
    args: dict[str, Any] = field(default_factory=dict)  # tool_use args
    result: str = ""         # tool_result: stringified result
    error: str = ""           # tool_result/error: error message
    path: str = ""           # file_change: filesystem path
    before: str = ""         # file_change: old contents (small files only)
    after: str = ""          # file_change: new contents (small files only)
    rationale: str = ""      # decision: why
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class UMSFDocument:
    """A single ingestion unit. The atomic input for Compiler.ingest_umsf."""
    schema: str = SCHEMA_VERSION
    agent: str = "generic"               # claude-code / codex / hermes / ...
    agent_version: str = ""              # optional version string
    source_type: str = "conversation"    # one of SOURCE_TYPES
    session_id: str = ""                 # opaque, agent-defined
    occurred_at: float = field(default_factory=time.time)
    actors: list[Actor] = field(default_factory=list)
    events: list[UMSFEvent] = field(default_factory=list)
    context: dict[str, Any] = field(default_factory=dict)
    title: str = ""                       # optional human-readable title
    summary: str = ""                     # optional pre-extracted summary

    # ── Serialization ──

    def to_dict(self) -> dict:
        return {
            "schema": self.schema,
            "agent": self.agent,
            "agent_version": self.agent_version,
            "source_type": self.source_type,
            "session_id": self.session_id,
            "occurred_at": self.occurred_at,
            "actors": [
                {"role": a.role, "id": a.id} for a in self.actors
            ],
            "events": [
                {k: v for k, v in vars(e).items() if v not in ("", {}, None)}
                | {"type": e.type, "ts": e.ts}
                for e in self.events
            ],
            "context": dict(self.context),
            "title": self.title,
            "summary": self.summary,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "UMSFDocument":
        actors = [Actor(**a) for a in d.get("actors", [])]
        events: list[UMSFEvent] = []
        for e in d.get("events", []):
            # Filter to known UMSFEvent fields to be forward-compatible
            allowed = set(UMSFEvent.__dataclass_fields__.keys())
            events.append(UMSFEvent(**{k: v for k, v in e.items() if k in allowed}))
        return cls(
            schema=d.get("schema", SCHEMA_VERSION),
            agent=d.get("agent", "generic"),
            agent_version=d.get("agent_version", ""),
            source_type=d.get("source_type", "conversation"),
            session_id=d.get("session_id", ""),
            occurred_at=d.get("occurred_at", time.time()),
            actors=actors,
            events=events,
            context=dict(d.get("context", {})),
            title=d.get("title", ""),
            summary=d.get("summary", ""),
        )


# ── Validation ──

class UMSFValidationError(ValueError):
    """Raised when a UMSFDocument fails validation."""


def validate(doc: UMSFDocument) -> None:
    """Validate document structure. Raises UMSFValidationError on problems.

    Lenient by design — we accept unknown event types so that future schema
    extensions don't break existing ingestion. Only the truly broken shapes
    are rejected.
    """
    if not isinstance(doc, UMSFDocument):
        raise UMSFValidationError(f"Expected UMSFDocument, got {type(doc).__name__}")
    if not doc.schema.startswith("ak/"):
        raise UMSFValidationError(f"Unknown schema: {doc.schema!r}")
    if doc.source_type not in SOURCE_TYPES:
        raise UMSFValidationError(
            f"source_type must be one of {SOURCE_TYPES}, got {doc.source_type!r}"
        )
    if not isinstance(doc.events, list):
        raise UMSFValidationError("events must be a list")
    if not isinstance(doc.actors, list):
        raise UMSFValidationError("actors must be a list")

    # A document needs at least one of: events, summary, context['body'].
    # Otherwise there's nothing to ingest.
    if not doc.events and not doc.summary and not doc.context.get("body"):
        raise UMSFValidationError(
            "UMSFDocument must have at least one of: events, summary, or context['body']"
        )


# ── Rendering: UMSF → markdown text for Compiler.ingest ──

def render(doc: UMSFDocument) -> str:
    """Render a UMSFDocument into markdown text suitable for Compiler.ingest.

    Strategy: produce section headers the Compiler already understands
    (decisions/learnings/actions/todo/risks) so claim extraction works
    without changes. Conversational messages go under a Conversation
    section as prose. Tool calls become Actions. Errors become Risks.
    """
    parts: list[str] = []

    title = doc.title or _derive_title(doc)
    if title:
        parts.append(f"# {title}")

    # Curated memory: pass the body straight through. Adapters write
    # the original markdown into context['body'] and the Compiler
    # extracts whatever sections it already supports.
    if doc.source_type == "curated_memory":
        body = doc.context.get("body", "") or doc.summary
        if body:
            parts.append(body.strip())
        return "\n\n".join(p for p in parts if p)

    # Optional summary as Summary section (PRIMARY-ish weight)
    if doc.summary:
        parts.append("## summary")
        parts.append(doc.summary.strip())

    # Group events by section
    decisions: list[UMSFEvent] = []
    actions: list[UMSFEvent] = []
    risks: list[UMSFEvent] = []
    file_changes: list[UMSFEvent] = []
    messages: list[UMSFEvent] = []

    for e in doc.events:
        t = e.type
        if t == UMSFEventType.DECISION.value:
            decisions.append(e)
        elif t in (UMSFEventType.TOOL_USE.value, UMSFEventType.TOOL_RESULT.value):
            actions.append(e)
        elif t == UMSFEventType.ERROR.value:
            risks.append(e)
        elif t == UMSFEventType.FILE_CHANGE.value:
            file_changes.append(e)
        elif t == UMSFEventType.MESSAGE.value:
            messages.append(e)
        # Other event types (session_start/end) intentionally ignored —
        # they carry no claim-bearing content.

    if decisions:
        parts.append("## decisions")
        for e in decisions:
            line = f"- {e.content.strip()}"
            if e.rationale:
                line += f" — rationale: {e.rationale.strip()}"
            parts.append(line)

    if actions:
        parts.append("## actions")
        for e in actions:
            if e.type == UMSFEventType.TOOL_USE.value:
                args_summary = _summarize_args(e.args)
                line = f"- {e.name}({args_summary})" if e.name else f"- {e.content.strip()}"
            else:
                # tool_result
                snippet = (e.result or e.error or e.content).strip().replace("\n", " ")
                line = f"- result: {snippet[:200]}"
            parts.append(line)

    if file_changes:
        parts.append("## actions")  # file changes are also actions
        for e in file_changes:
            label = e.name or "modified"
            parts.append(f"- {label} {e.path}".rstrip())

    if risks:
        parts.append("## risks")
        for e in risks:
            text = (e.content or e.error).strip()
            if text:
                parts.append(f"- {text[:300]}")

    if messages:
        parts.append("## conversation")
        for e in messages:
            role = e.role or "speaker"
            content = e.content.strip()
            if content:
                parts.append(f"**{role}**: {content}")

    return "\n\n".join(p for p in parts if p)


def _summarize_args(args: dict) -> str:
    """One-line summary of tool args for inclusion in a markdown bullet."""
    if not args:
        return ""
    pieces: list[str] = []
    for k, v in args.items():
        if isinstance(v, (str, int, float, bool)):
            s = str(v)
            if len(s) > 60:
                s = s[:60] + "…"
            pieces.append(f"{k}={s}")
        else:
            pieces.append(f"{k}=…")
        if len(pieces) >= 3:
            pieces.append("…")
            break
    return ", ".join(pieces)


def _derive_title(doc: UMSFDocument) -> str:
    """Pick a reasonable title when one isn't provided."""
    if doc.summary:
        # First sentence of summary, capped
        head = doc.summary.strip().split(".")[0].strip()
        if 6 < len(head) < 100:
            return head
    # Fall back to "{agent} {source_type}" — descriptive enough for triage
    parts = [doc.agent, doc.source_type]
    if doc.session_id:
        parts.append(f"session {doc.session_id[:8]}")
    return " · ".join(p for p in parts if p)


# ── Convenience: ingest UMSF straight into a Vault via Compiler ──

def ingest_umsf(compiler, doc: UMSFDocument):
    """Ingest a UMSFDocument via the standard Compiler pipeline.

    The compiler is duck-typed (avoids a circular import of core.compiler).
    The original UMSF dict is preserved on the resulting Source as an
    audit trail — future reprocessing (e.g. re-extracting claims with
    a newer compiler) can re-render from the structured form rather
    than the lossy text body. Returns the created Source object.
    """
    validate(doc)
    text = render(doc)
    path: Optional[str] = doc.context.get("path")
    url: Optional[str] = doc.context.get("url")
    title = doc.title or _derive_title(doc)
    source = compiler.ingest(
        text=text,
        title=title,
        source_type=doc.source_type,
        path=path,
        url=url,
    )
    # Stamp the audit trail and persist. Cheap re-save; vault cache hit.
    source.umsf = doc.to_dict()
    compiler.vault.save_source(source)

    # Mirror events into the structured event index for time-ordered
    # queries (e.g. "list all tool_use events in this session"). The
    # index is the secondary store — failures here do not invalidate
    # the primary Source persistence above.
    if doc.events:
        try:
            compiler.vault.event_index.add(
                source_id=source.id,
                session_id=doc.session_id,
                events=doc.events,
                agent=doc.agent,
            )
        except Exception:
            # EventIndex failures should never block ingestion; surface via
            # logging in production setups, swallow here for safety.
            pass

    return source
