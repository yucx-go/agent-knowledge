"""Tests for the auto-capture hooks system (UMSF-native)."""

from __future__ import annotations

import os
import time

import pytest

from agent_knowledge.core.compiler import Compiler
from agent_knowledge.core.models import ClaimDurability
from agent_knowledge.core.umsf import UMSFDocument, UMSFEvent
from agent_knowledge.core.vault import Vault
from agent_knowledge.hooks.base import Hook, HookManager, HookResult
from agent_knowledge.hooks.decision_hook import DecisionHook
from agent_knowledge.hooks.error_hook import ErrorHook
from agent_knowledge.hooks.file_hook import FileHook, FileWatcher
from agent_knowledge.hooks.message_hook import MessageHook
from agent_knowledge.hooks.response_hook import ResponseHook
from agent_knowledge.hooks.session_hook import SessionHook
from agent_knowledge.hooks.tool_hook import ToolHook


@pytest.fixture
def vault(tmp_path):
    v = Vault(tmp_path / "test_vault")
    v.init()
    return v


@pytest.fixture
def compiler(vault):
    return Compiler(vault)


def _msg(content: str, role: str = "user") -> UMSFEvent:
    return UMSFEvent(type="message", ts=time.time(), role=role, content=content)


# ── MessageHook ──


class TestMessageHook:
    def test_should_fire_preference(self):
        hook = MessageHook()
        assert hook.should_fire(_msg("我喜欢用Python写脚本")) is True

    def test_should_fire_directive(self):
        hook = MessageHook()
        assert hook.should_fire(_msg("记住这个配置文件路径")) is True

    def test_should_fire_foresight(self):
        hook = MessageHook()
        assert hook.should_fire(_msg("下周要完成代码审查")) is True

    def test_should_not_fire_plain(self):
        hook = MessageHook()
        assert hook.should_fire(_msg("今天天气不错")) is False

    def test_should_not_fire_short(self):
        hook = MessageHook()
        assert hook.should_fire(_msg("hi")) is False

    def test_should_not_fire_wrong_type(self):
        hook = MessageHook()
        ev = UMSFEvent(type="error", ts=time.time(), content="我喜欢Python")
        assert hook.should_fire(ev) is False

    def test_should_not_fire_assistant_role(self):
        """User-side only — assistant messages route to ResponseHook."""
        hook = MessageHook()
        assert hook.should_fire(_msg("我喜欢极简风格", role="assistant")) is False

    def test_process_preference(self, vault, compiler):
        hook = MessageHook()
        ev = _msg("我喜欢极简的代码风格")
        result = hook.process(ev, "sess1", vault, compiler)
        assert isinstance(result, HookResult)
        assert len(result.claims) == 1
        assert "preference" in result.claims[0].tags
        assert result.claims[0].durability == ClaimDurability.STABLE
        assert result.source_id != ""

    def test_process_directive(self, vault, compiler):
        hook = MessageHook()
        ev = _msg("记住以后用YAML格式保存配置")
        result = hook.process(ev, "sess1", vault, compiler)
        assert "directive" in result.claims[0].tags

    def test_process_foresight(self, vault, compiler):
        hook = MessageHook()
        ev = _msg("下周要完成API文档")
        result = hook.process(ev, "sess1", vault, compiler)
        assert "foresight" in result.claims[0].tags
        assert result.claims[0].durability == ClaimDurability.TEMPORARY


# ── ResponseHook ──


class TestResponseHook:
    def test_should_fire_commitment(self):
        hook = ResponseHook()
        assert hook.should_fire(_msg("我会立即处理这个问题", role="assistant")) is True

    def test_should_fire_correction(self):
        hook = ResponseHook()
        assert hook.should_fire(_msg("更正一下，实际上应该是这样", role="assistant")) is True

    def test_should_not_fire_user_role(self):
        """Assistant-side only — user messages route to MessageHook."""
        hook = ResponseHook()
        assert hook.should_fire(_msg("我会处理", role="user")) is False

    def test_should_not_fire_plain(self):
        hook = ResponseHook()
        assert hook.should_fire(_msg("好的", role="assistant")) is False

    def test_process_correction(self, vault, compiler):
        hook = ResponseHook()
        ev = _msg("之前说错了，更正：应该是配置在env文件中", role="assistant")
        result = hook.process(ev, "sess1", vault, compiler)
        assert "correction" in result.claims[0].tags
        assert result.claims[0].durability == ClaimDurability.STABLE

    def test_process_commitment(self, vault, compiler):
        hook = ResponseHook()
        ev = _msg("我会马上完成这项任务", role="assistant")
        result = hook.process(ev, "sess1", vault, compiler)
        assert "commitment" in result.claims[0].tags


# ── ToolHook ──


class TestToolHook:
    def test_should_fire_with_name(self):
        hook = ToolHook()
        ev = UMSFEvent(type="tool_use", ts=time.time(), name="bash")
        assert hook.should_fire(ev) is True

    def test_should_not_fire_without_name(self):
        hook = ToolHook()
        ev = UMSFEvent(type="tool_use", ts=time.time())
        assert hook.should_fire(ev) is False

    def test_should_not_fire_wrong_type(self):
        hook = ToolHook()
        ev = UMSFEvent(type="message", ts=time.time(), name="bash")
        assert hook.should_fire(ev) is False

    def test_process_with_args(self, vault, compiler):
        hook = ToolHook()
        ev = UMSFEvent(
            type="tool_use",
            ts=time.time(),
            name="bash",
            args={"command": "ls -la"},
        )
        result = hook.process(ev, "sess1", vault, compiler)
        assert len(result.claims) == 1
        assert "tool_use" in result.claims[0].tags
        assert "bash" in result.claims[0].tags

    def test_process_extracts_keys_from_result(self, vault, compiler):
        hook = ToolHook()
        ev = UMSFEvent(
            type="tool_use",
            ts=time.time(),
            name="api_call",
            result="response: token=abc123 url=https://api.example.com path=/v1/users",
        )
        result = hook.process(ev, "sess1", vault, compiler)
        # Keys section should be present in the claim text
        assert "Keys:" in result.claims[0].text


# ── ErrorHook ──


class TestErrorHook:
    def test_should_fire_with_error(self):
        hook = ErrorHook()
        ev = UMSFEvent(type="error", ts=time.time(), error="Connection refused")
        assert hook.should_fire(ev) is True

    def test_should_fire_with_content(self):
        hook = ErrorHook()
        ev = UMSFEvent(type="error", ts=time.time(), content="Something went wrong")
        assert hook.should_fire(ev) is True

    def test_should_not_fire_empty(self):
        hook = ErrorHook()
        ev = UMSFEvent(type="error", ts=time.time())
        assert hook.should_fire(ev) is False

    def test_process(self, vault, compiler):
        hook = ErrorHook()
        ev = UMSFEvent(
            type="error",
            ts=time.time(),
            error="Database connection timeout",
            name="postgres",
            metadata={"context": "user signup flow"},
        )
        result = hook.process(ev, "sess1", vault, compiler)
        assert len(result.claims) == 1
        assert "error" in result.claims[0].tags
        assert "pitfall" in result.claims[0].tags
        assert result.claims[0].durability == ClaimDurability.STABLE
        # All three components should appear in the summary
        text = result.claims[0].text
        assert "Database connection timeout" in text
        assert "postgres" in text
        assert "user signup flow" in text


# ── SessionHook ──


class TestSessionHook:
    def test_should_fire_with_messages(self):
        hook = SessionHook()
        ev = UMSFEvent(
            type="session_end",
            ts=time.time(),
            metadata={
                "messages": [
                    {"role": "user", "text": "hi"},
                    {"role": "assistant", "text": "hello"},
                ]
            },
        )
        assert hook.should_fire(ev) is True

    def test_should_not_fire_too_few(self):
        hook = SessionHook()
        ev = UMSFEvent(
            type="session_end",
            ts=time.time(),
            metadata={"messages": [{"role": "user", "text": "hi"}]},
        )
        assert hook.should_fire(ev) is False

    def test_process_compresses_messages(self, vault, compiler):
        hook = SessionHook()
        ev = UMSFEvent(
            type="session_end",
            ts=time.time(),
            metadata={
                "messages": [
                    {"role": "user", "text": "讨论架构方案"},
                    {"role": "assistant", "text": "推荐三层架构"},
                    {"role": "user", "text": "采纳这个方案"},
                ],
            },
        )
        result = hook.process(ev, "sess1", vault, compiler)
        assert result.source_id != ""


# ── DecisionHook ──


class TestDecisionHook:
    def test_should_fire_decision_event(self):
        hook = DecisionHook()
        ev = UMSFEvent(type="decision", ts=time.time(), content="决定采用方案B")
        assert hook.should_fire(ev) is True

    def test_should_fire_message_with_decision(self):
        hook = DecisionHook()
        ev = _msg("我们决定切换到PostgreSQL")
        assert hook.should_fire(ev) is True

    def test_should_not_fire_plain_message(self):
        hook = DecisionHook()
        ev = _msg("今天有点忙")
        assert hook.should_fire(ev) is False

    def test_process(self, vault, compiler):
        hook = DecisionHook()
        ev = UMSFEvent(
            type="decision",
            ts=time.time(),
            content="decided to migrate from MySQL to PostgreSQL by Q2",
        )
        result = hook.process(ev, "sess1", vault, compiler)
        assert len(result.claims) == 1
        assert "decision" in result.claims[0].tags
        assert result.claims[0].durability == ClaimDurability.STABLE
        assert result.claims[0].confidence >= 0.8


# ── FileHook ──


class TestFileHook:
    def test_should_fire(self, tmp_path):
        hook = FileHook()
        f = tmp_path / "test.md"
        f.write_text("hello", encoding="utf-8")
        ev = UMSFEvent(type="file_change", ts=time.time(), path=str(f))
        assert hook.should_fire(ev) is True

    def test_should_not_fire_missing(self):
        hook = FileHook()
        ev = UMSFEvent(type="file_change", ts=time.time(), path="/nonexistent/file.md")
        assert hook.should_fire(ev) is False

    def test_should_not_fire_no_path(self):
        hook = FileHook()
        ev = UMSFEvent(type="file_change", ts=time.time())
        assert hook.should_fire(ev) is False

    def test_process(self, vault, compiler, tmp_path):
        hook = FileHook()
        f = tmp_path / "notes.md"
        # Explicit UTF-8 — Path.write_text uses the system default encoding
        # which is CP936 on Chinese-locale Windows and would not round-trip
        # through hook.process (which reads as utf-8).
        f.write_text("## 决策\n- 采用新架构方案", encoding="utf-8")
        ev = UMSFEvent(type="file_change", ts=time.time(), path=str(f))
        result = hook.process(ev, "sess1", vault, compiler)
        assert result.source_id != ""

    def test_process_binary_skip(self, vault, compiler, tmp_path):
        hook = FileHook()
        f = tmp_path / "binary.md"
        f.write_bytes(b"\x80\x81\x82\xfe\xff" * 100)
        ev = UMSFEvent(type="file_change", ts=time.time(), path=str(f))
        result = hook.process(ev, "sess1", vault, compiler)
        assert result.source_id == ""


# ── FileWatcher ──


class TestFileWatcher:
    def test_poll_once_detects_changes(self, vault, tmp_path):
        watch_dir = tmp_path / "watch"
        watch_dir.mkdir()

        manager = HookManager(vault)
        manager.register_defaults()

        watcher = FileWatcher(watch_dir, manager, interval=1.0)
        watcher.poll_once()  # initial scan

        (watch_dir / "new.md").write_text("## 学习\n- 新发现", encoding="utf-8")
        changed = watcher.poll_once()
        assert len(changed) == 1
        assert "new.md" in changed[0]

    def test_poll_once_no_change(self, vault, tmp_path):
        watch_dir = tmp_path / "watch"
        watch_dir.mkdir()
        (watch_dir / "existing.md").write_text("old content")

        manager = HookManager(vault)
        manager.register_defaults()

        watcher = FileWatcher(watch_dir, manager, interval=1.0)
        watcher.poll_once()
        changed = watcher.poll_once()
        assert len(changed) == 0

    def test_poll_detects_modification(self, vault, tmp_path):
        watch_dir = tmp_path / "watch"
        watch_dir.mkdir()
        f = watch_dir / "doc.md"
        f.write_text("v1")

        manager = HookManager(vault)
        manager.register_defaults()

        watcher = FileWatcher(watch_dir, manager, interval=1.0)
        watcher.poll_once()

        time.sleep(0.05)
        f.write_text("v2")
        os.utime(str(f), (time.time() + 1, time.time() + 1))

        changed = watcher.poll_once()
        assert len(changed) == 1

    def test_stop(self, vault, tmp_path):
        watch_dir = tmp_path / "watch"
        watch_dir.mkdir()
        manager = HookManager(vault)
        watcher = FileWatcher(watch_dir, manager, interval=0.1)
        watcher.stop()
        assert watcher._running is False


# ── HookManager ──


def _doc(events: list[UMSFEvent], **kw) -> UMSFDocument:
    """Helper to build a single-document with the right source_type."""
    # Pick a source_type that will pass UMSF validation for the events
    type_to_source = {
        "decision": "decision",
        "file_change": "file_change",
        "tool_use": "tool_trace",
        "tool_result": "tool_trace",
        "error": "tool_trace",
    }
    if events:
        st = type_to_source.get(events[0].type, "conversation")
    else:
        st = kw.pop("source_type", "conversation")
    return UMSFDocument(
        agent=kw.pop("agent", "test"),
        source_type=kw.pop("source_type", st),
        events=events,
        **kw,
    )


class TestHookManager:
    def test_register_and_fire(self, vault):
        manager = HookManager(vault)
        manager.register_defaults()

        doc = _doc([_msg("我喜欢用Vim编辑器")])
        results = manager.fire(doc)
        assert len(results) >= 1

    def test_stats_tracking(self, vault):
        manager = HookManager(vault)
        manager.register_defaults()

        doc = _doc([UMSFEvent(type="error", ts=time.time(), error="something broke")])
        manager.fire(doc)
        stats = manager.stats()
        assert stats.get("error", 0) >= 1

    def test_list_hooks(self, vault):
        manager = HookManager(vault)
        manager.register_defaults()
        hooks = manager.list_hooks()
        names = {h["name"] for h in hooks}
        assert names == {"message", "tool_use", "response", "error", "session", "decision", "file_change"}

    def test_no_match(self, vault):
        manager = HookManager(vault)
        manager.register_defaults()
        # session_start has no hook; full doc still ingests via track 1
        doc = _doc([UMSFEvent(type="session_start", ts=time.time())])
        results = manager.fire(doc)
        # Track 1 (full ingest) may add a result; track 2 yields nothing
        # The key invariant is no fan-out hooks fired
        stats = manager.stats()
        assert stats.get("message", 0) == 0
        assert stats.get("decision", 0) == 0

    def test_decision_hook_fires_on_message(self, vault):
        manager = HookManager(vault)
        manager.register_defaults()

        doc = _doc([_msg("我们决定采用方案B")])
        results = manager.fire(doc)
        # DecisionHook should fire alongside MessageHook
        all_tags: list[str] = []
        for r in results:
            for c in r.claims:
                all_tags.extend(c.tags)
        assert "decision" in all_tags

    def test_empty_manager(self, vault):
        manager = HookManager(vault)
        # No hooks registered. Track 1 (full ingest) still produces a source_id.
        doc = _doc([_msg("test message")])
        results = manager.fire(doc)
        # At least the full-ingest result, but no hook fan-out
        stats = manager.stats()
        assert stats == {}  # no hooks registered, so no counts

    def test_invalid_doc_rejected(self, vault):
        from agent_knowledge.core.umsf import UMSFValidationError

        manager = HookManager(vault)
        bad = UMSFDocument()  # no events, no summary, no body
        with pytest.raises(UMSFValidationError):
            manager.fire(bad)


# ── MCP Integration ──


class TestMCPHookIntegration:
    def test_hook_fire_tool(self, vault):
        from agent_knowledge.adapters.mcp import MCPServer

        server = MCPServer(vault)
        result = server._tool_hook_fire({
            "type": "error",
            "error": "test error",
            "name": "test_tool",
            "session_id": "test",
        })
        text = result["content"][0]["text"]
        assert "Fired event: error" in text
        assert "Hooks matched:" in text

    def test_hook_fire_missing_type(self, vault):
        from agent_knowledge.adapters.mcp import MCPServer

        server = MCPServer(vault)
        result = server._tool_hook_fire({})
        assert result.get("isError") is True

    def test_hook_list_tool(self, vault):
        from agent_knowledge.adapters.mcp import MCPServer

        server = MCPServer(vault)
        result = server._tool_hook_list({})
        text = result["content"][0]["text"]
        assert "message" in text
        assert "error" in text

    def test_hook_stats_tool(self, vault):
        from agent_knowledge.adapters.mcp import MCPServer

        server = MCPServer(vault)
        # Fire something first
        server._tool_hook_fire({"type": "error", "error": "test"})
        result = server._tool_hook_stats({})
        text = result["content"][0]["text"]
        assert "error:" in text

    def test_tools_list_includes_hooks_and_umsf(self, vault):
        from agent_knowledge.adapters.mcp import TOOLS

        tool_names = [t["name"] for t in TOOLS]
        assert "ak_hook_fire" in tool_names
        assert "ak_hook_list" in tool_names
        assert "ak_hook_stats" in tool_names
        assert "ak_ingest_umsf" in tool_names

    def test_ingest_umsf_tool(self, vault):
        from agent_knowledge.adapters.mcp import MCPServer

        server = MCPServer(vault)
        doc_dict = {
            "schema": "ak/v1",
            "agent": "claude-code",
            "source_type": "decision",
            "title": "Test decision",
            "events": [
                {
                    "type": "decision",
                    "ts": time.time(),
                    "content": "decided to use UMSF as boundary format",
                },
            ],
        }
        result = server._tool_ingest_umsf({"doc": doc_dict})
        text = result["content"][0]["text"]
        assert "Ingested UMSFDocument" in text
        assert "claude-code" in text

    def test_ingest_umsf_validation_error(self, vault):
        from agent_knowledge.adapters.mcp import MCPServer

        server = MCPServer(vault)
        # Empty events + no summary + no body → validation fails
        result = server._tool_ingest_umsf({"doc": {"schema": "ak/v1"}})
        assert result.get("isError") is True

    def test_ingest_umsf_missing_doc(self, vault):
        from agent_knowledge.adapters.mcp import MCPServer

        server = MCPServer(vault)
        result = server._tool_ingest_umsf({})
        assert result.get("isError") is True

    def test_mcp_dispatch_hook_tools(self, vault):
        """Test that MCP dispatch routes hook tools correctly."""
        from agent_knowledge.adapters.mcp import MCPServer

        server = MCPServer(vault)
        resp = server.handle_message({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {"name": "ak_hook_list", "arguments": {}},
        })
        assert resp["id"] == 1
        assert "result" in resp
        assert "content" in resp["result"]
