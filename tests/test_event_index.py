"""Tests for the structured event index."""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
import time

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from agent_knowledge.core.compiler import Compiler
from agent_knowledge.core.event_index import EventIndex
from agent_knowledge.core.umsf import UMSFDocument, UMSFEvent, ingest_umsf
from agent_knowledge.core.vault import Vault


@pytest.fixture
def vault():
    d = tempfile.mkdtemp(prefix="ak-test-evidx-")
    v = Vault(d)
    v.init()
    yield v
    shutil.rmtree(d, ignore_errors=True)


# ── Lifecycle ──


class TestLifecycle:
    def test_lazy_init_no_db_until_used(self, vault):
        # Just accessing the property shouldn't create the file
        idx = vault.event_index
        assert isinstance(idx, EventIndex)
        # But once we use it, the file appears
        idx.add("s1", "sess1", [UMSFEvent(type="message", ts=1.0, content="hi")])
        assert idx.db_path.exists()

    def test_db_in_vault_root(self, vault):
        assert vault.event_index.db_path.parent == vault.root


# ── Insertions ──


class TestAdd:
    def test_add_basic(self, vault):
        idx = vault.event_index
        events = [
            UMSFEvent(type="message", ts=1.0, role="user", content="hi"),
            UMSFEvent(type="tool_use", ts=2.0, name="bash", args={"command": "ls"}),
        ]
        n = idx.add("source-1", "sess1", events, agent="claude-code")
        assert n == 2

    def test_add_empty_returns_zero(self, vault):
        idx = vault.event_index
        assert idx.add("s1", "sess1", []) == 0

    def test_add_preserves_metadata(self, vault):
        idx = vault.event_index
        ev = UMSFEvent(
            type="tool_use",
            ts=1.0,
            name="bash",
            args={"command": "pytest", "cwd": "/repo"},
            metadata={"exit_code": 0, "elapsed_ms": 1234},
        )
        idx.add("s1", "sess1", [ev])

        results = idx.by_source("s1")
        assert len(results) == 1
        assert results[0]["args"] == {"command": "pytest", "cwd": "/repo"}
        assert results[0]["metadata"] == {"exit_code": 0, "elapsed_ms": 1234}

    def test_add_multiple_sources(self, vault):
        idx = vault.event_index
        idx.add("s1", "sess1", [UMSFEvent(type="message", ts=1.0, content="a")])
        idx.add("s2", "sess1", [UMSFEvent(type="message", ts=2.0, content="b")])
        idx.add("s3", "sess2", [UMSFEvent(type="message", ts=3.0, content="c")])

        assert len(idx.by_source("s1")) == 1
        assert len(idx.by_source("s2")) == 1
        assert len(idx.by_session("sess1")) == 2
        assert len(idx.by_session("sess2")) == 1


# ── Queries ──


class TestQuery:
    def _populate(self, idx):
        events = [
            UMSFEvent(type="message", ts=100.0, role="user", content="hi"),
            UMSFEvent(type="tool_use", ts=200.0, name="bash"),
            UMSFEvent(type="tool_result", ts=300.0, result="ok"),
            UMSFEvent(type="decision", ts=400.0, content="use postgres"),
            UMSFEvent(type="error", ts=500.0, error="timeout"),
        ]
        idx.add("s1", "sess1", events, agent="claude-code")
        idx.add("s2", "sess2", [
            UMSFEvent(type="tool_use", ts=600.0, name="git"),
        ], agent="codex")

    def test_filter_by_type(self, vault):
        idx = vault.event_index
        self._populate(idx)
        results = idx.query(type="tool_use")
        assert len(results) == 2
        assert all(r["type"] == "tool_use" for r in results)

    def test_filter_by_session(self, vault):
        idx = vault.event_index
        self._populate(idx)
        results = idx.query(session_id="sess1")
        assert len(results) == 5

    def test_filter_by_agent(self, vault):
        idx = vault.event_index
        self._populate(idx)
        results = idx.query(agent="codex")
        assert len(results) == 1
        assert results[0]["name"] == "git"

    def test_filter_by_time_range(self, vault):
        idx = vault.event_index
        self._populate(idx)
        results = idx.query(since_ts=200, until_ts=400)
        assert len(results) == 3
        assert {r["type"] for r in results} == {"tool_use", "tool_result", "decision"}

    def test_combine_filters(self, vault):
        idx = vault.event_index
        self._populate(idx)
        results = idx.query(type="tool_use", session_id="sess1")
        assert len(results) == 1
        assert results[0]["session_id"] == "sess1"

    def test_order_asc(self, vault):
        idx = vault.event_index
        self._populate(idx)
        results = idx.query(session_id="sess1", order="asc")
        timestamps = [r["ts"] for r in results]
        assert timestamps == sorted(timestamps)

    def test_order_desc(self, vault):
        idx = vault.event_index
        self._populate(idx)
        results = idx.query(session_id="sess1", order="desc")
        timestamps = [r["ts"] for r in results]
        assert timestamps == sorted(timestamps, reverse=True)

    def test_limit(self, vault):
        idx = vault.event_index
        self._populate(idx)
        results = idx.query(session_id="sess1", limit=2)
        assert len(results) == 2


# ── Mutations ──


class TestDelete:
    def test_delete_source_events(self, vault):
        idx = vault.event_index
        idx.add("s1", "sess1", [
            UMSFEvent(type="message", ts=1.0, content="a"),
            UMSFEvent(type="message", ts=2.0, content="b"),
        ])
        idx.add("s2", "sess1", [UMSFEvent(type="message", ts=3.0, content="c")])

        deleted = idx.delete_source_events("s1")
        assert deleted == 2
        assert len(idx.by_source("s1")) == 0
        assert len(idx.by_source("s2")) == 1


# ── Stats ──


class TestStats:
    def test_empty_stats(self, vault):
        idx = vault.event_index
        # Lazy init triggers via stats()
        s = idx.stats()
        assert s["total_events"] == 0
        assert s["by_type"] == {}
        assert s["unique_sessions"] == 0

    def test_populated_stats(self, vault):
        idx = vault.event_index
        idx.add("s1", "sess1", [
            UMSFEvent(type="message", ts=1.0, content="a"),
            UMSFEvent(type="tool_use", ts=2.0, name="bash"),
        ], agent="claude-code")
        idx.add("s2", "sess2", [
            UMSFEvent(type="tool_use", ts=3.0, name="git"),
        ], agent="codex")

        s = idx.stats()
        assert s["total_events"] == 3
        assert s["by_type"]["message"] == 1
        assert s["by_type"]["tool_use"] == 2
        assert s["unique_sessions"] == 2
        assert s["by_agent"]["claude-code"] == 2
        assert s["by_agent"]["codex"] == 1


# ── Integration with ingest_umsf ──


class TestIngestUMSFAutoPopulates:
    def test_ingest_umsf_writes_to_event_index(self, vault):
        compiler = Compiler(vault)
        doc = UMSFDocument(
            agent="claude-code",
            source_type="conversation",
            session_id="sess-int",
            events=[
                UMSFEvent(type="message", ts=time.time(), role="user", content="hi"),
                UMSFEvent(type="tool_use", ts=time.time(), name="bash"),
                UMSFEvent(type="message", ts=time.time(), role="assistant", content="done"),
            ],
        )
        source = ingest_umsf(compiler, doc)

        events = vault.event_index.by_source(source.id)
        assert len(events) == 3
        types = [e["type"] for e in events]
        assert types == ["message", "tool_use", "message"]

    def test_curated_memory_with_no_events_creates_no_index_rows(self, vault):
        compiler = Compiler(vault)
        doc = UMSFDocument(
            agent="hermes",
            source_type="curated_memory",
            title="Test memory",
            context={"body": "## decisions\n- chose Postgres"},
        )
        source = ingest_umsf(compiler, doc)

        # No events in doc → nothing indexed (body still ingested as Source)
        events = vault.event_index.by_source(source.id)
        assert events == []

    def test_query_by_session_after_ingest(self, vault):
        compiler = Compiler(vault)
        # Ingest two docs in the same session
        for i in range(2):
            doc = UMSFDocument(
                agent="codex",
                source_type="conversation",
                session_id="multi-doc-sess",
                events=[
                    UMSFEvent(type="message", ts=time.time() + i, content=f"turn {i}"),
                ],
            )
            ingest_umsf(compiler, doc)

        events = vault.event_index.by_session("multi-doc-sess")
        assert len(events) == 2
