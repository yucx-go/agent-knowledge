"""Tests for the Claude Code JSONL pull adapter."""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from agent_knowledge.adapters.claude_code import (
    ClaudeCodeAdapter,
    DiscoveredSession,
    _parse_iso_ts,
    _truncate,
)
from agent_knowledge.core.vault import Vault


@pytest.fixture
def vault():
    d = tempfile.mkdtemp(prefix="ak-test-cc-")
    v = Vault(d)
    v.init()
    yield v
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def fake_home(tmp_path):
    """A temp directory used as $HOME — adapter scans tmp/.claude/."""
    return tmp_path


def _write_jsonl(path: Path, events: list[dict]) -> None:
    """Helper: write a list of JSONL events to a session file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for ev in events:
            f.write(json.dumps(ev, ensure_ascii=False) + "\n")


def _claude_session(
    home: Path,
    project_slug: str,
    session_uuid: str,
    events: list[dict],
) -> Path:
    p = home / ".claude" / "projects" / project_slug / f"{session_uuid}.jsonl"
    _write_jsonl(p, events)
    return p


# ── Helpers ──


class TestHelpers:
    def test_parse_iso_ts(self):
        assert _parse_iso_ts("2026-05-14T08:21:08.323Z") > 0
        assert _parse_iso_ts("2026-05-14T08:21:08+00:00") > 0
        assert _parse_iso_ts("garbage") == 0.0
        assert _parse_iso_ts(None) == 0.0
        assert _parse_iso_ts(12345) == 0.0

    def test_truncate(self):
        assert _truncate("hello") == "hello"
        assert _truncate("x" * 2000, limit=100) == "x" * 99 + "…"
        assert len(_truncate("x" * 2000, limit=100)) == 100


# ── Discovery ──


class TestDiscovery:
    def test_empty_home(self, vault, fake_home):
        adapter = ClaudeCodeAdapter(vault, home_override=fake_home)
        assert adapter.discover() == []

    def test_finds_session_files(self, vault, fake_home):
        _claude_session(fake_home, "proj-A", "uuid-1", [
            {"type": "user", "message": {"role": "user", "content": "hi"}},
        ])
        _claude_session(fake_home, "proj-B", "uuid-2", [
            {"type": "user", "message": {"role": "user", "content": "hello"}},
        ])

        adapter = ClaudeCodeAdapter(vault, home_override=fake_home)
        items = adapter.discover()
        assert len(items) == 2
        assert {i.project_slug for i in items} == {"proj-A", "proj-B"}
        assert {i.session_uuid for i in items} == {"uuid-1", "uuid-2"}

    def test_filter_by_projects_substring(self, vault, fake_home):
        _claude_session(fake_home, "agent-knowledge", "u1", [{"type": "user", "message": {"role": "user", "content": "x"}}])
        _claude_session(fake_home, "code2", "u2", [{"type": "user", "message": {"role": "user", "content": "y"}}])

        adapter = ClaudeCodeAdapter(vault, home_override=fake_home)
        items = adapter.discover(projects=["agent-knowledge"])
        assert len(items) == 1
        assert items[0].project_slug == "agent-knowledge"

    def test_since_days_filter(self, vault, fake_home):
        old = _claude_session(fake_home, "p1", "old", [{"type": "user", "message": {"role": "user", "content": "x"}}])
        # Make the file old by setting mtime to 30 days ago
        thirty_days_ago = time.time() - (30 * 86400)
        os.utime(old, (thirty_days_ago, thirty_days_ago))

        _claude_session(fake_home, "p2", "new", [{"type": "user", "message": {"role": "user", "content": "y"}}])

        adapter = ClaudeCodeAdapter(vault, home_override=fake_home)
        items = adapter.discover(since_days=7)
        assert len(items) == 1
        assert items[0].session_uuid == "new"

    def test_limit_returns_most_recent(self, vault, fake_home):
        for i in range(5):
            p = _claude_session(fake_home, f"p{i}", f"u{i}", [
                {"type": "user", "message": {"role": "user", "content": "x"}},
            ])
            # Stagger mtimes so we know the ordering
            os.utime(p, (1000 + i, 1000 + i))

        adapter = ClaudeCodeAdapter(vault, home_override=fake_home)
        items = adapter.discover(limit=2)
        assert len(items) == 2
        # Most recent first
        assert items[0].session_uuid == "u4"
        assert items[1].session_uuid == "u3"

    def test_discovered_session_fingerprint_uses_size_mtime(self, vault, fake_home):
        p = _claude_session(fake_home, "proj", "uuid", [
            {"type": "user", "message": {"role": "user", "content": "x"}},
        ])
        adapter = ClaudeCodeAdapter(vault, home_override=fake_home)
        items = adapter.discover()
        st = p.stat()
        assert items[0].fingerprint == f"{st.st_size}:{st.st_mtime}"


# ── Fetch / parsing ──


class TestFetch:
    def test_user_text_message(self, vault, fake_home):
        _claude_session(fake_home, "proj", "uuid", [
            {
                "type": "user",
                "message": {"role": "user", "content": "我决定使用 React"},
                "timestamp": "2026-05-14T08:21:08.000Z",
                "sessionId": "uuid",
                "cwd": "D:\\repo",
                "gitBranch": "main",
                "version": "2.1.131",
            },
        ])
        adapter = ClaudeCodeAdapter(vault, home_override=fake_home)
        items = adapter.discover()
        doc = adapter.fetch(items[0])

        assert doc is not None
        assert doc.agent == "claude-code"
        assert doc.source_type == "conversation"
        assert doc.session_id == "uuid"
        assert "React" in doc.title
        assert len(doc.events) == 1
        assert doc.events[0].type == "message"
        assert doc.events[0].role == "user"
        assert "React" in doc.events[0].content
        # Session-level context captured
        assert doc.context["cwd"] == "D:\\repo"
        assert doc.context["git_branch"] == "main"
        assert doc.context["claude_code_version"] == "2.1.131"

    def test_assistant_text_block(self, vault, fake_home):
        _claude_session(fake_home, "proj", "uuid", [
            {
                "type": "assistant",
                "message": {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": "Sure, I'll help"},
                    ],
                },
                "timestamp": "2026-05-14T08:21:09.000Z",
            },
        ])
        adapter = ClaudeCodeAdapter(vault, home_override=fake_home)
        doc = adapter.fetch(adapter.discover()[0])

        assert doc is not None
        assert len(doc.events) == 1
        assert doc.events[0].type == "message"
        assert doc.events[0].role == "assistant"
        assert "Sure" in doc.events[0].content

    def test_tool_use_block(self, vault, fake_home):
        _claude_session(fake_home, "proj", "uuid", [
            {
                "type": "assistant",
                "message": {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "toolu_abc123",
                            "name": "Read",
                            "input": {"file_path": "/tmp/foo.py"},
                        },
                    ],
                },
                "timestamp": "2026-05-14T08:21:10.000Z",
            },
        ])
        adapter = ClaudeCodeAdapter(vault, home_override=fake_home)
        doc = adapter.fetch(adapter.discover()[0])

        assert doc is not None
        assert len(doc.events) == 1
        ev = doc.events[0]
        assert ev.type == "tool_use"
        assert ev.name == "Read"
        assert ev.args == {"file_path": "/tmp/foo.py"}
        assert ev.metadata.get("tool_use_id") == "toolu_abc123"

    def test_tool_result_block(self, vault, fake_home):
        _claude_session(fake_home, "proj", "uuid", [
            {
                "type": "user",
                "message": {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "toolu_abc123",
                            "content": "file contents here",
                        },
                    ],
                },
                "timestamp": "2026-05-14T08:21:11.000Z",
            },
        ])
        adapter = ClaudeCodeAdapter(vault, home_override=fake_home)
        doc = adapter.fetch(adapter.discover()[0])

        assert doc is not None
        ev = doc.events[0]
        assert ev.type == "tool_result"
        assert "file contents" in ev.result
        assert ev.metadata.get("tool_use_id") == "toolu_abc123"

    def test_tool_result_error(self, vault, fake_home):
        _claude_session(fake_home, "proj", "uuid", [
            {
                "type": "user",
                "message": {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "toolu_xyz",
                            "content": "command failed: exit 1",
                            "is_error": True,
                        },
                    ],
                },
            },
        ])
        adapter = ClaudeCodeAdapter(vault, home_override=fake_home)
        doc = adapter.fetch(adapter.discover()[0])

        ev = doc.events[0]
        assert ev.type == "tool_result"
        assert "command failed" in ev.error
        assert ev.result == ""

    def test_thinking_skipped_by_default(self, vault, fake_home):
        _claude_session(fake_home, "proj", "uuid", [
            {
                "type": "assistant",
                "message": {
                    "role": "assistant",
                    "content": [
                        {"type": "thinking", "thinking": "Let me think..."},
                        {"type": "text", "text": "Here is my answer"},
                    ],
                },
            },
        ])
        adapter = ClaudeCodeAdapter(vault, home_override=fake_home)
        doc = adapter.fetch(adapter.discover()[0])

        assert len(doc.events) == 1
        assert "thinking" not in doc.events[0].content.lower()
        assert "answer" in doc.events[0].content

    def test_thinking_included_when_requested(self, vault, fake_home):
        _claude_session(fake_home, "proj", "uuid", [
            {
                "type": "assistant",
                "message": {
                    "role": "assistant",
                    "content": [
                        {"type": "thinking", "thinking": "internal reasoning"},
                    ],
                },
            },
        ])
        adapter = ClaudeCodeAdapter(
            vault, home_override=fake_home, include_thinking=True,
        )
        doc = adapter.fetch(adapter.discover()[0])

        assert len(doc.events) == 1
        assert doc.events[0].metadata.get("is_thinking") is True

    def test_metadata_events_skipped(self, vault, fake_home):
        _claude_session(fake_home, "proj", "uuid", [
            {"type": "permission-mode", "permissionMode": "default"},
            {"type": "file-history-snapshot", "snapshot": {}},
            {"type": "last-prompt"},
            {"type": "attachment"},
            {
                "type": "user",
                "message": {"role": "user", "content": "hi"},
            },
        ])
        adapter = ClaudeCodeAdapter(vault, home_override=fake_home)
        doc = adapter.fetch(adapter.discover()[0])

        # Only the user message survives
        assert len(doc.events) == 1
        assert doc.events[0].content == "hi"

    def test_malformed_lines_skipped(self, vault, fake_home):
        p = fake_home / ".claude" / "projects" / "proj" / "uuid.jsonl"
        p.parent.mkdir(parents=True)
        p.write_text(
            json.dumps({"type": "user", "message": {"role": "user", "content": "valid"}}) + "\n"
            "not valid json at all\n"
            + json.dumps({"type": "user", "message": {"role": "user", "content": "also valid"}}) + "\n",
            encoding="utf-8",
        )
        adapter = ClaudeCodeAdapter(vault, home_override=fake_home)
        doc = adapter.fetch(adapter.discover()[0])
        assert len(doc.events) == 2

    def test_empty_session_returns_none(self, vault, fake_home):
        _claude_session(fake_home, "proj", "uuid", [
            {"type": "permission-mode"},  # only metadata
        ])
        adapter = ClaudeCodeAdapter(vault, home_override=fake_home)
        doc = adapter.fetch(adapter.discover()[0])
        assert doc is None

    def test_long_content_truncated(self, vault, fake_home):
        long_text = "x" * 5000
        _claude_session(fake_home, "proj", "uuid", [
            {
                "type": "user",
                "message": {"role": "user", "content": long_text},
            },
        ])
        adapter = ClaudeCodeAdapter(vault, home_override=fake_home)
        doc = adapter.fetch(adapter.discover()[0])

        # Truncated to MAX_CONTENT_CHARS (1500)
        assert len(doc.events[0].content) <= 1500
        assert doc.events[0].content.endswith("…")

    def test_title_from_first_user_message(self, vault, fake_home):
        _claude_session(fake_home, "proj", "uuid", [
            {"type": "permission-mode"},  # ignored
            {
                "type": "user",
                "message": {"role": "user", "content": "评价下这个项目"},
            },
            {
                "type": "user",
                "message": {"role": "user", "content": "second message"},
            },
        ])
        adapter = ClaudeCodeAdapter(vault, home_override=fake_home)
        doc = adapter.fetch(adapter.discover()[0])
        assert "评价下这个项目" in doc.title


# ── End-to-end pull ──


class TestPullEndToEnd:
    def test_pull_creates_source(self, vault, fake_home):
        _claude_session(fake_home, "proj", "uuid", [
            {
                "type": "user",
                "message": {"role": "user", "content": "我决定使用 PostgreSQL"},
                "timestamp": "2026-05-14T08:21:08Z",
                "cwd": "D:\\repo",
            },
            {
                "type": "assistant",
                "message": {
                    "role": "assistant",
                    "content": [
                        {"type": "tool_use", "id": "tu1", "name": "Read", "input": {"file": "x"}},
                    ],
                },
            },
        ])

        adapter = ClaudeCodeAdapter(vault, home_override=fake_home)
        report = adapter.pull()
        assert len(report.ingested) == 1
        # Source has umsf audit trail (Surgery 1)
        source = vault.load_source(report.ingested[0])
        assert source.umsf is not None
        assert source.umsf["agent"] == "claude-code"
        # EventIndex got the events (Surgery 3)
        events = vault.event_index.by_source(source.id)
        assert len(events) == 2
        types = [e["type"] for e in events]
        assert "message" in types
        assert "tool_use" in types

    def test_pull_dedups_unchanged(self, vault, fake_home):
        _claude_session(fake_home, "proj", "uuid", [
            {"type": "user", "message": {"role": "user", "content": "hello"}},
        ])
        adapter = ClaudeCodeAdapter(vault, home_override=fake_home)

        r1 = adapter.pull()
        assert len(r1.ingested) == 1

        r2 = adapter.pull()
        assert len(r2.ingested) == 0
        assert len(r2.skipped) == 1

    def test_pull_force_reingests(self, vault, fake_home):
        _claude_session(fake_home, "proj", "uuid", [
            {"type": "user", "message": {"role": "user", "content": "hello"}},
        ])
        adapter = ClaudeCodeAdapter(vault, home_override=fake_home)
        adapter.pull()
        r2 = adapter.pull(force=True)
        assert len(r2.ingested) == 1

    def test_state_namespace(self, vault, fake_home):
        _claude_session(fake_home, "proj", "uuid", [
            {"type": "user", "message": {"role": "user", "content": "hello"}},
        ])
        adapter = ClaudeCodeAdapter(vault, home_override=fake_home)
        adapter.pull()

        # State file should contain a "claude-code" namespace separate
        # from any markdown-memory state
        state_path = vault.root / adapter.STATE_FILE
        assert state_path.exists()
        state = json.loads(state_path.read_text(encoding="utf-8"))
        assert "claude-code" in state
        items = state["claude-code"]
        assert len(items) == 1
        entry = next(iter(items.values()))
        assert entry["metadata"]["session_uuid"] == "uuid"
        assert entry["metadata"]["project_slug"] == "proj"
