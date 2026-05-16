"""Claude Code adapter — pulls conversation JSONL files into UMSF.

Claude Code stores each session as a single JSONL file under
``~/.claude/projects/{project_slug}/{session_uuid}.jsonl``. Each line
is a separate JSON event. This adapter walks those files and
converts them to UMSFDocuments.

JSONL event shapes (subset we map)
-----------------------------------

::

    # User text message
    {"type":"user","message":{"role":"user","content":"<text>"},
     "uuid":"<event-id>","timestamp":"...","sessionId":"...","cwd":"..."}

    # Assistant turn — content is an array of blocks
    {"type":"assistant","message":{"role":"assistant","content":[
        {"type":"text","text":"..."},
        {"type":"thinking","thinking":"..."},
        {"type":"tool_use","name":"Read","input":{...},"id":"toolu_..."}
    ]}, ...}

    # Tool result — appears as a *user-typed* envelope whose content
    # array contains tool_result blocks
    {"type":"user","message":{"role":"user","content":[
        {"type":"tool_result","tool_use_id":"toolu_...","content":"..."}
    ]}, ...}

    # Pure metadata events (skipped) — permission-mode, file-history-snapshot,
    # last-prompt, attachment, summary

UMSF mapping
------------

Each JSONL session becomes one UMSFDocument with ``agent="claude-code"``,
``source_type="conversation"``, and one UMSFEvent per content block:

    text/plain user message    → UMSFEvent(type="message", role="user",      content=text)
    text block (assistant)      → UMSFEvent(type="message", role="assistant", content=text)
    thinking block              → skipped by default (set include_thinking=True to keep)
    tool_use block              → UMSFEvent(type="tool_use", name=..., args=input)
    tool_result block           → UMSFEvent(type="tool_result", result=..., metadata={tool_use_id})
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator, Optional

from ..core.umsf import UMSFDocument, UMSFEvent
from ..core.vault import Vault
from .base import BasePullAdapter, DiscoveredItem


# Top-level JSONL events that carry no claim-bearing content.
_SKIP_TYPES: frozenset[str] = frozenset({
    "permission-mode",
    "file-history-snapshot",
    "last-prompt",
    "attachment",
    "summary",
    "meta",
})

# Per-event content cap — Claude Code tool results can be megabytes
# (file dumps, large outputs); we don't want a single event blowing up
# the rendered Source.content. EventIndex keeps full strings up to its
# own per-row limits separately.
_MAX_CONTENT_CHARS = 1500


def _claude_root(home: Optional[Path] = None) -> Path:
    """Return ``~/.claude/`` (or the override)."""
    home = home or Path.home()
    return home / ".claude"


def _parse_iso_ts(value: Any) -> float:
    """Parse an ISO-8601 timestamp into a unix float. Returns 0 on failure."""
    if not isinstance(value, str):
        return 0.0
    try:
        # JSONL timestamps are like "2026-05-14T08:21:08.323Z"
        if value.endswith("Z"):
            value = value[:-1] + "+00:00"
        return datetime.fromisoformat(value).timestamp()
    except Exception:
        return 0.0


def _truncate(text: str, limit: int = _MAX_CONTENT_CHARS) -> str:
    """Cap content length, preserving a '…' suffix when truncated."""
    if len(text) <= limit:
        return text
    return text[: limit - 1] + "…"


@dataclass
class DiscoveredSession(DiscoveredItem):
    """A Claude Code session JSONL file."""
    path: Optional[Path] = None
    project_slug: str = ""
    session_uuid: str = ""
    mtime: float = 0.0
    size_bytes: int = 0

    def __post_init__(self) -> None:
        if self.path is None:
            return
        if not self.id:
            self.id = str(self.path)
        if not self.fingerprint:
            self.fingerprint = f"{self.size_bytes}:{self.mtime}"
        if not self.label:
            short_id = self.session_uuid[:8] if self.session_uuid else "?"
            self.label = f"[claude-code] {self.project_slug} :: {short_id}"
        # Surface adapter-specific fields in metadata for state inspection
        self.metadata.setdefault("project_slug", self.project_slug)
        self.metadata.setdefault("session_uuid", self.session_uuid)
        self.metadata.setdefault("size_bytes", self.size_bytes)
        self.metadata.setdefault("mtime", self.mtime)


class ClaudeCodeAdapter(BasePullAdapter):
    """Pull adapter for Claude Code JSONL conversation transcripts.

    Usage::

        adapter = ClaudeCodeAdapter(vault)
        report = adapter.pull()                            # all sessions
        report = adapter.pull(projects=["agent-knowledge"]) # filter
        report = adapter.pull(since_days=7)                # recent only
        report = adapter.pull(limit=10)                    # cap count
    """

    name = "claude-code"

    def __init__(
        self,
        vault: Vault,
        home_override: Optional[Path] = None,
        include_thinking: bool = False,
    ):
        super().__init__(vault)
        self._home = home_override
        self.include_thinking = include_thinking

    # ── Discovery ──

    def discover(
        self,
        projects: Optional[list[str]] = None,
        since_days: Optional[float] = None,
        limit: Optional[int] = None,
        **_: Any,
    ) -> list[DiscoveredSession]:
        """Find Claude Code session JSONL files.

        Args:
            projects: substring filter — match if any item appears in the
                project_slug. e.g. ``["agent-knowledge"]``.
            since_days: only include sessions modified in the last N days.
            limit: cap discovery to this many most-recently-modified
                sessions (after filters apply).
        """
        root = _claude_root(self._home)
        projects_dir = root / "projects"
        if not projects_dir.is_dir():
            return []

        cutoff_ts: Optional[float] = None
        if since_days is not None:
            import time
            cutoff_ts = time.time() - (float(since_days) * 86400)

        results: list[DiscoveredSession] = []

        for project_dir in sorted(projects_dir.iterdir()):
            if not project_dir.is_dir():
                continue
            slug = project_dir.name
            if projects and not any(p in slug for p in projects):
                continue

            for jsonl in project_dir.glob("*.jsonl"):
                try:
                    st = jsonl.stat()
                except OSError:
                    continue
                if cutoff_ts is not None and st.st_mtime < cutoff_ts:
                    continue

                results.append(DiscoveredSession(
                    path=jsonl,
                    project_slug=slug,
                    session_uuid=jsonl.stem,
                    size_bytes=st.st_size,
                    mtime=st.st_mtime,
                ))

        # Sort by mtime descending so --limit picks the most recent
        results.sort(key=lambda s: s.mtime, reverse=True)
        if limit is not None and limit > 0:
            results = results[:limit]
        return results

    # ── Fetch / parse ──

    def fetch(self, item: DiscoveredItem) -> Optional[UMSFDocument]:
        """Parse a JSONL session file into a UMSFDocument.

        Returns None for empty or completely malformed files.
        """
        if not isinstance(item, DiscoveredSession) or item.path is None:
            return None

        events: list[UMSFEvent] = []
        first_user_text: str = ""
        cwd: str = ""
        git_branch: str = ""
        version: str = ""
        agent_session_id: str = item.session_uuid

        for raw in self._iter_jsonl_lines(item.path):
            obj_type = raw.get("type")
            if not obj_type or obj_type in _SKIP_TYPES:
                continue

            ts = _parse_iso_ts(raw.get("timestamp"))

            # Capture session-level context from the first user event we see
            if not cwd and isinstance(raw.get("cwd"), str):
                cwd = raw["cwd"]
            if not git_branch and isinstance(raw.get("gitBranch"), str):
                git_branch = raw["gitBranch"]
            if not version and isinstance(raw.get("version"), str):
                version = raw["version"]
            if not agent_session_id and isinstance(raw.get("sessionId"), str):
                agent_session_id = raw["sessionId"]

            message = raw.get("message") or {}
            content = message.get("content")

            if obj_type == "user":
                # Two cases: string content (real user text) or an array of
                # tool_result blocks. We split them so UMSF events keep their
                # native types.
                if isinstance(content, str):
                    text = content.strip()
                    if not text:
                        continue
                    if not first_user_text:
                        first_user_text = text
                    events.append(UMSFEvent(
                        type="message",
                        ts=ts,
                        role="user",
                        content=_truncate(text),
                    ))
                elif isinstance(content, list):
                    for block in content:
                        ev = self._block_to_event(block, ts, role="user")
                        if ev is not None:
                            events.append(ev)
            elif obj_type == "assistant":
                if isinstance(content, list):
                    for block in content:
                        ev = self._block_to_event(block, ts, role="assistant")
                        if ev is not None:
                            events.append(ev)
                elif isinstance(content, str):
                    text = content.strip()
                    if text:
                        events.append(UMSFEvent(
                            type="message",
                            ts=ts,
                            role="assistant",
                            content=_truncate(text),
                        ))

        if not events:
            return None

        # Build title from first user message; fall back to label otherwise
        title = first_user_text[:80] if first_user_text else item.label
        if first_user_text and len(first_user_text) > 80:
            title = title + "…"

        return UMSFDocument(
            agent="claude-code",
            source_type="conversation",
            session_id=agent_session_id,
            occurred_at=item.mtime,
            title=title,
            events=events,
            context={
                "path": str(item.path),
                "project_slug": item.project_slug,
                "cwd": cwd,
                "git_branch": git_branch,
                "claude_code_version": version,
            },
        )

    # ── Parsing helpers ──

    @staticmethod
    def _iter_jsonl_lines(path: Path) -> Iterator[dict]:
        """Yield decoded JSON objects, silently skipping malformed lines.

        Claude Code's JSONL is normally well-formed, but I've seen the
        occasional truncated line in long sessions; logging would noise
        more than help, so we just skip.
        """
        try:
            with path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(obj, dict):
                        yield obj
        except OSError:
            return

    def _block_to_event(
        self,
        block: dict,
        ts: float,
        role: str,
    ) -> Optional[UMSFEvent]:
        """Map a single content block to a UMSFEvent (or None to skip)."""
        if not isinstance(block, dict):
            return None
        block_type = block.get("type")

        if block_type == "text":
            text = (block.get("text") or "").strip()
            if not text:
                return None
            return UMSFEvent(
                type="message",
                ts=ts,
                role=role,
                content=_truncate(text),
            )

        if block_type == "thinking":
            if not self.include_thinking:
                return None
            text = (block.get("thinking") or "").strip()
            if not text:
                return None
            return UMSFEvent(
                type="message",
                ts=ts,
                role=role,
                content=_truncate(text),
                metadata={"is_thinking": True},
            )

        if block_type == "tool_use":
            return UMSFEvent(
                type="tool_use",
                ts=ts,
                name=str(block.get("name") or ""),
                args=block.get("input") or {},
                metadata={"tool_use_id": str(block.get("id") or "")},
            )

        if block_type == "tool_result":
            content = block.get("content")
            # tool_result content can be a string OR a list of blocks; flatten
            if isinstance(content, list):
                parts = []
                for c in content:
                    if isinstance(c, dict) and c.get("type") == "text":
                        parts.append(c.get("text") or "")
                    elif isinstance(c, str):
                        parts.append(c)
                content_str = "\n".join(parts)
            else:
                content_str = str(content or "")

            is_error = bool(block.get("is_error"))
            return UMSFEvent(
                type="tool_result",
                ts=ts,
                result=_truncate(content_str) if not is_error else "",
                error=_truncate(content_str) if is_error else "",
                metadata={"tool_use_id": str(block.get("tool_use_id") or "")},
            )

        # Unknown block type — log into metadata so audit trail preserves it
        return UMSFEvent(
            type="message",
            ts=ts,
            role=role,
            content="",
            metadata={"unknown_block_type": str(block_type)},
        )
