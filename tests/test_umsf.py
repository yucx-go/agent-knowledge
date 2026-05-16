"""Tests for UMSF (Universal Memory Source Format)."""

import os
import shutil
import sys
import tempfile
import time

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from agent_knowledge.core.umsf import (
    SCHEMA_VERSION,
    SOURCE_TYPES,
    Actor,
    UMSFDocument,
    UMSFEvent,
    UMSFEventType,
    UMSFValidationError,
    ingest_umsf,
    render,
    validate,
)
from agent_knowledge.core.compiler import Compiler
from agent_knowledge.core.vault import Vault


@pytest.fixture
def vault():
    d = tempfile.mkdtemp(prefix="ak-test-umsf-")
    v = Vault(d)
    v.init()
    yield v
    shutil.rmtree(d, ignore_errors=True)


# ── Schema basics ──

class TestSchema:
    def test_schema_version_constant(self):
        assert SCHEMA_VERSION == "ak/v1"

    def test_source_types_include_known(self):
        assert "conversation" in SOURCE_TYPES
        assert "curated_memory" in SOURCE_TYPES
        assert "skill_artifact" in SOURCE_TYPES
        assert "decision" in SOURCE_TYPES

    def test_event_types_enum(self):
        assert UMSFEventType.MESSAGE.value == "message"
        assert UMSFEventType.TOOL_USE.value == "tool_use"
        assert UMSFEventType.DECISION.value == "decision"
        assert UMSFEventType.FILE_CHANGE.value == "file_change"

    def test_default_doc_uses_current_schema(self):
        doc = UMSFDocument()
        assert doc.schema == SCHEMA_VERSION
        assert doc.source_type == "conversation"


# ── Validation ──

class TestValidation:
    def _doc_with_summary(self, **kw):
        return UMSFDocument(summary="non-empty summary", **kw)

    def test_valid_doc_passes(self):
        doc = self._doc_with_summary()
        validate(doc)  # should not raise

    def test_unknown_schema_rejected(self):
        doc = self._doc_with_summary()
        doc.schema = "openai/v1"
        with pytest.raises(UMSFValidationError, match="Unknown schema"):
            validate(doc)

    def test_unknown_source_type_rejected(self):
        doc = self._doc_with_summary(source_type="totally_made_up")
        with pytest.raises(UMSFValidationError, match="source_type"):
            validate(doc)

    def test_empty_doc_rejected(self):
        doc = UMSFDocument()  # no events, no summary, no body
        with pytest.raises(UMSFValidationError, match="at least one of"):
            validate(doc)

    def test_doc_with_events_passes(self):
        doc = UMSFDocument(events=[
            UMSFEvent(type="message", role="user", content="hi"),
        ])
        validate(doc)

    def test_doc_with_body_in_context_passes(self):
        doc = UMSFDocument(
            source_type="curated_memory",
            context={"body": "## decisions\n- use React"},
        )
        validate(doc)

    def test_non_umsf_rejected(self):
        with pytest.raises(UMSFValidationError, match="UMSFDocument"):
            validate({"schema": "ak/v1"})


# ── Serialization round-trip ──

class TestSerialization:
    def test_round_trip(self):
        doc = UMSFDocument(
            agent="claude-code",
            agent_version="1.0.86",
            source_type="conversation",
            session_id="abc123",
            title="Test session",
            summary="Quick test",
            actors=[Actor(role="user", id="u1"), Actor(role="assistant")],
            events=[
                UMSFEvent(type="message", role="user", content="Hello"),
                UMSFEvent(
                    type="tool_use",
                    name="bash",
                    args={"command": "ls -la"},
                ),
            ],
            context={"cwd": "/home/user"},
        )

        as_dict = doc.to_dict()
        restored = UMSFDocument.from_dict(as_dict)

        assert restored.agent == "claude-code"
        assert restored.session_id == "abc123"
        assert len(restored.actors) == 2
        assert restored.actors[0].role == "user"
        assert restored.actors[0].id == "u1"
        assert len(restored.events) == 2
        assert restored.events[0].content == "Hello"
        assert restored.events[1].name == "bash"
        assert restored.events[1].args == {"command": "ls -la"}

    def test_unknown_event_fields_ignored(self):
        # Forward-compatibility: unknown fields don't crash from_dict
        d = {
            "schema": "ak/v1",
            "events": [
                {"type": "message", "content": "hi", "future_field": "ignore me"}
            ],
        }
        doc = UMSFDocument.from_dict(d)
        assert len(doc.events) == 1
        assert doc.events[0].content == "hi"


# ── Rendering ──

class TestRender:
    def test_curated_memory_passes_body_through(self):
        doc = UMSFDocument(
            source_type="curated_memory",
            title="My memory",
            context={"body": "## decisions\n- prefer pytest"},
        )
        text = render(doc)
        assert "# My memory" in text
        assert "## decisions" in text
        assert "prefer pytest" in text

    def test_decision_event_renders_as_section(self):
        doc = UMSFDocument(
            source_type="decision",
            events=[
                UMSFEvent(
                    type="decision",
                    content="Use React for frontend",
                    rationale="team is more familiar",
                ),
            ],
        )
        text = render(doc)
        assert "## decisions" in text
        assert "Use React for frontend" in text
        assert "team is more familiar" in text

    def test_tool_use_renders_as_action(self):
        doc = UMSFDocument(
            events=[
                UMSFEvent(type="tool_use", name="bash", args={"command": "ls"}),
            ],
        )
        text = render(doc)
        assert "## actions" in text
        assert "bash" in text
        assert "command=ls" in text

    def test_error_renders_as_risk(self):
        doc = UMSFDocument(
            events=[
                UMSFEvent(type="error", content="Connection timeout"),
            ],
        )
        text = render(doc)
        assert "## risks" in text
        assert "Connection timeout" in text

    def test_file_change_renders_as_action(self):
        doc = UMSFDocument(
            events=[
                UMSFEvent(type="file_change", name="modified", path="/tmp/foo.txt"),
            ],
        )
        text = render(doc)
        assert "## actions" in text
        assert "/tmp/foo.txt" in text

    def test_message_renders_as_conversation(self):
        doc = UMSFDocument(
            events=[
                UMSFEvent(type="message", role="user", content="What is X?"),
                UMSFEvent(type="message", role="assistant", content="X is foo"),
            ],
        )
        text = render(doc)
        assert "## conversation" in text
        assert "**user**" in text
        assert "**assistant**" in text
        assert "What is X?" in text

    def test_summary_renders_first(self):
        doc = UMSFDocument(
            summary="Today we picked our auth strategy.",
            events=[
                UMSFEvent(type="decision", content="Use OAuth"),
            ],
        )
        text = render(doc)
        assert "## summary" in text
        assert "Today we picked" in text
        # decisions section comes after summary
        assert text.index("## summary") < text.index("## decisions")

    def test_session_start_end_ignored(self):
        doc = UMSFDocument(
            summary="placeholder",
            events=[
                UMSFEvent(type="session_start"),
                UMSFEvent(type="session_end"),
            ],
        )
        text = render(doc)
        # No content section for these
        assert "session_start" not in text
        assert "session_end" not in text

    def test_title_auto_derived_when_missing(self):
        doc = UMSFDocument(
            agent="codex",
            source_type="decision",
            session_id="9f8e7d6c5b4a",
            events=[UMSFEvent(type="decision", content="ship it")],
        )
        text = render(doc)
        assert text.startswith("# ")
        assert "codex" in text.split("\n")[0].lower()


# ── End-to-end: UMSF → Compiler → Vault ──

class TestIngestUMSF:
    def test_ingest_decision_doc(self, vault):
        compiler = Compiler(vault)
        doc = UMSFDocument(
            agent="claude-code",
            source_type="decision",
            title="Tech stack decision",
            events=[
                UMSFEvent(
                    type="decision",
                    content="Use React for the new frontend",
                    rationale="team familiarity outweighs the bundle size cost",
                ),
            ],
        )
        source = ingest_umsf(compiler, doc)

        assert source.title == "Tech stack decision"
        assert source.source_type == "decision"
        # Decision should have produced at least one claim
        assert len(source.claims_extracted) >= 1

    def test_ingest_curated_memory(self, vault):
        compiler = Compiler(vault)
        doc = UMSFDocument(
            agent="hermes",
            source_type="curated_memory",
            title="Hermes MEMORY.md",
            context={
                "body": "## decisions\n- chose PostgreSQL for JSONB support",
                "path": "/home/user/.hermes/MEMORY.md",
            },
        )
        source = ingest_umsf(compiler, doc)

        assert source.path == "/home/user/.hermes/MEMORY.md"
        assert len(source.claims_extracted) >= 1

    def test_ingest_validates(self, vault):
        compiler = Compiler(vault)
        # Empty doc should fail validation before reaching compiler
        bad = UMSFDocument()
        with pytest.raises(UMSFValidationError):
            ingest_umsf(compiler, bad)

    def test_ingest_tool_trace(self, vault):
        compiler = Compiler(vault)
        doc = UMSFDocument(
            agent="claude-code",
            source_type="tool_trace",
            session_id="abc123",
            events=[
                UMSFEvent(type="tool_use", name="bash", args={"command": "pytest"}),
                UMSFEvent(type="tool_result", result="245 passed"),
                UMSFEvent(type="tool_use", name="git", args={"action": "commit"}),
            ],
        )
        source = ingest_umsf(compiler, doc)
        assert source.source_type == "tool_trace"

    def test_ingest_persists_umsf_audit_trail(self, vault):
        """Source.umsf preserves the original UMSF dict for audit/replay."""
        compiler = Compiler(vault)
        doc = UMSFDocument(
            agent="claude-code",
            source_type="decision",
            title="Audit trail test",
            session_id="sess-abc",
            events=[
                UMSFEvent(type="decision", content="decided to log audit trails"),
            ],
        )
        source = ingest_umsf(compiler, doc)

        # In-memory source has the audit trail
        assert source.umsf is not None
        assert source.umsf["agent"] == "claude-code"
        assert source.umsf["session_id"] == "sess-abc"
        assert len(source.umsf["events"]) == 1

        # Persisted to vault — round-trip survives
        reloaded = vault.load_source(source.id)
        # Clear cache to force re-read
        vault.clear_cache()
        reloaded = vault.load_source(source.id)
        assert reloaded.umsf is not None
        assert reloaded.umsf["agent"] == "claude-code"

    def test_direct_compiler_ingest_leaves_umsf_none(self, vault):
        """Sources created without UMSF have umsf=None (no audit trail)."""
        compiler = Compiler(vault)
        source = compiler.ingest("plain text", title="raw")
        assert source.umsf is None

    def test_ingest_conversation_with_decisions_extracted(self, vault):
        compiler = Compiler(vault)
        doc = UMSFDocument(
            agent="hermes",
            source_type="conversation",
            events=[
                UMSFEvent(type="message", role="user",
                          content="Should we migrate to Postgres?"),
                UMSFEvent(type="message", role="assistant",
                          content="I recommend it for the JSONB support"),
                UMSFEvent(type="decision",
                          content="Migrate from MySQL to PostgreSQL by end of Q2",
                          rationale="JSONB support and MVCC characteristics"),
            ],
        )
        source = ingest_umsf(compiler, doc)
        # Should extract claims from both the conversation and the decision
        assert len(source.claims_extracted) >= 1
