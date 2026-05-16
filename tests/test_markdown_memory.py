"""Tests for the markdown memory pull adapter."""

import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from agent_knowledge.adapters.markdown_memory import (
    KNOWN_AGENTS,
    DiscoveredFile,
    MarkdownMemoryAdapter,
    PullReport,
    _agent_root,
)
from agent_knowledge.core.vault import Vault


@pytest.fixture
def vault():
    d = tempfile.mkdtemp(prefix="ak-test-mdm-")
    v = Vault(d)
    v.init()
    yield v
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def fake_home(tmp_path):
    """Provide a path that adapter will treat as $HOME via home_override."""
    return tmp_path


# ── Discovery ──

class TestDiscovery:
    def test_known_agents_constant(self):
        assert "hermes" in KNOWN_AGENTS
        assert "openclaw" in KNOWN_AGENTS
        assert "claude-code" in KNOWN_AGENTS
        assert "codex" in KNOWN_AGENTS

    def test_empty_home(self, vault, fake_home):
        adapter = MarkdownMemoryAdapter(vault, home_override=fake_home)
        files = adapter.discover()
        assert files == []

    def test_finds_hermes_files(self, vault, fake_home):
        hermes = fake_home / ".hermes"
        hermes.mkdir()
        (hermes / "MEMORY.md").write_text("# Memory\n- entry one\n", encoding="utf-8")
        (hermes / "USER.md").write_text("# User\n- profile data\n", encoding="utf-8")

        adapter = MarkdownMemoryAdapter(vault, home_override=fake_home)
        files = adapter.discover()

        assert len(files) == 2
        agents = {f.agent for f in files}
        assert agents == {"hermes"}
        roles = {f.role for f in files}
        assert "memory" in roles
        assert "user_profile" in roles

    def test_finds_all_known_paths(self, vault, fake_home):
        for d in (".hermes", ".openclaw", ".claude", ".codex"):
            (fake_home / d).mkdir()
        (fake_home / ".hermes" / "MEMORY.md").write_text("h", encoding="utf-8")
        (fake_home / ".openclaw" / "MEMORY.md").write_text("o", encoding="utf-8")
        (fake_home / ".claude" / "CLAUDE.md").write_text("c", encoding="utf-8")
        (fake_home / ".codex" / "AGENTS.md").write_text("x", encoding="utf-8")

        adapter = MarkdownMemoryAdapter(vault, home_override=fake_home)
        files = adapter.discover()

        agents = {f.agent for f in files}
        assert agents == {"hermes", "openclaw", "claude-code", "codex"}

    def test_filter_by_agent(self, vault, fake_home):
        (fake_home / ".hermes").mkdir()
        (fake_home / ".hermes" / "MEMORY.md").write_text("h", encoding="utf-8")
        (fake_home / ".openclaw").mkdir()
        (fake_home / ".openclaw" / "MEMORY.md").write_text("o", encoding="utf-8")

        adapter = MarkdownMemoryAdapter(vault, home_override=fake_home)
        files = adapter.discover(agents=["hermes"])

        assert len(files) == 1
        assert files[0].agent == "hermes"

    def test_empty_agent_list_skips_known_paths(self, vault, fake_home):
        (fake_home / ".hermes").mkdir()
        (fake_home / ".hermes" / "MEMORY.md").write_text("h", encoding="utf-8")

        adapter = MarkdownMemoryAdapter(vault, home_override=fake_home)
        files = adapter.discover(agents=[])

        assert files == []

    def test_custom_path(self, vault, fake_home, tmp_path):
        custom = tmp_path / "my_notes.md"
        custom.write_text("custom notes content", encoding="utf-8")

        adapter = MarkdownMemoryAdapter(vault, home_override=fake_home)
        files = adapter.discover(agents=[], custom_paths=[str(custom)])

        assert len(files) == 1
        assert files[0].agent == "custom"
        assert files[0].path == custom.resolve()

    def test_nonexistent_custom_path_silently_skipped(self, vault, fake_home):
        adapter = MarkdownMemoryAdapter(vault, home_override=fake_home)
        files = adapter.discover(
            agents=[], custom_paths=["/nonexistent/file.md"]
        )
        assert files == []

    def test_discovered_file_title(self, vault, fake_home):
        hermes = fake_home / ".hermes"
        hermes.mkdir()
        (hermes / "MEMORY.md").write_text("x", encoding="utf-8")
        adapter = MarkdownMemoryAdapter(vault, home_override=fake_home)
        files = adapter.discover()
        assert "[hermes]" in files[0].title
        assert "MEMORY" in files[0].title
        assert "memory" in files[0].title


# ── Pull ──

class TestPull:
    def _setup_hermes(self, fake_home, content="# Memory\n## decisions\n- Use React"):
        hermes = fake_home / ".hermes"
        hermes.mkdir()
        (hermes / "MEMORY.md").write_text(content, encoding="utf-8")
        return hermes / "MEMORY.md"

    def test_pull_ingests(self, vault, fake_home):
        self._setup_hermes(fake_home)

        adapter = MarkdownMemoryAdapter(vault, home_override=fake_home)
        report = adapter.pull()

        assert len(report.ingested) == 1
        assert vault.list_sources() != []
        assert report.duration_seconds >= 0

    def test_pull_skips_unchanged(self, vault, fake_home):
        self._setup_hermes(fake_home)
        adapter = MarkdownMemoryAdapter(vault, home_override=fake_home)

        r1 = adapter.pull()
        assert len(r1.ingested) == 1

        r2 = adapter.pull()
        assert len(r2.ingested) == 0
        assert len(r2.skipped) == 1

    def test_pull_force_reingests(self, vault, fake_home):
        self._setup_hermes(fake_home)
        adapter = MarkdownMemoryAdapter(vault, home_override=fake_home)
        adapter.pull()

        r2 = adapter.pull(force=True)
        assert len(r2.ingested) == 1

    def test_pull_detects_mtime_change(self, vault, fake_home):
        path = self._setup_hermes(fake_home)
        adapter = MarkdownMemoryAdapter(vault, home_override=fake_home)
        adapter.pull()

        # Modify file (write new content + bump mtime)
        import time as _t
        _t.sleep(0.01)
        path.write_text("# Memory\n## decisions\n- Use Vue", encoding="utf-8")
        # Force mtime forward in case filesystem resolution is coarse
        new_mtime = path.stat().st_mtime + 1
        os.utime(path, (new_mtime, new_mtime))

        r2 = adapter.pull()
        assert len(r2.ingested) == 1

    def test_dry_run(self, vault, fake_home):
        self._setup_hermes(fake_home)
        adapter = MarkdownMemoryAdapter(vault, home_override=fake_home)

        report = adapter.pull(dry_run=True)
        assert len(report.discovered) == 1
        assert len(report.ingested) == 0
        assert vault.list_sources() == []
        # State file should not have been written
        assert not (vault.root / adapter.STATE_FILE).exists()

    def test_empty_file_skipped(self, vault, fake_home):
        hermes = fake_home / ".hermes"
        hermes.mkdir()
        (hermes / "MEMORY.md").write_text("", encoding="utf-8")
        (hermes / "USER.md").write_text("   \n  \n", encoding="utf-8")

        adapter = MarkdownMemoryAdapter(vault, home_override=fake_home)
        report = adapter.pull()
        assert len(report.ingested) == 0
        assert len(report.skipped) == 2

    def test_state_file_persists(self, vault, fake_home):
        self._setup_hermes(fake_home)
        adapter = MarkdownMemoryAdapter(vault, home_override=fake_home)
        adapter.pull()

        state_path = vault.root / adapter.STATE_FILE
        assert state_path.exists()
        state = json.loads(state_path.read_text(encoding="utf-8"))
        # State is now namespaced by adapter.name to allow multiple adapters
        # to coexist in a single vault.
        assert "markdown-memory" in state
        items = state["markdown-memory"]
        assert len(items) == 1
        entry = next(iter(items.values()))
        assert "fingerprint" in entry
        assert "metadata" in entry
        assert entry["metadata"]["agent"] == "hermes"
        assert entry["metadata"]["role"] == "memory"

    def test_filter_by_agent_in_pull(self, vault, fake_home):
        (fake_home / ".hermes").mkdir()
        (fake_home / ".hermes" / "MEMORY.md").write_text("h-content", encoding="utf-8")
        (fake_home / ".openclaw").mkdir()
        (fake_home / ".openclaw" / "MEMORY.md").write_text("o-content", encoding="utf-8")

        adapter = MarkdownMemoryAdapter(vault, home_override=fake_home)
        report = adapter.pull(agents=["hermes"])
        assert len(report.ingested) == 1

    def test_custom_path_pull(self, vault, fake_home, tmp_path):
        custom = tmp_path / "private_notes.md"
        custom.write_text(
            "## decisions\n- Custom note: prefer pytest over unittest",
            encoding="utf-8",
        )

        adapter = MarkdownMemoryAdapter(vault, home_override=fake_home)
        report = adapter.pull(agents=[], custom_paths=[str(custom)])

        assert len(report.ingested) == 1


# ── Path resolution ──

class TestAgentRoot:
    def test_hermes_default(self, fake_home):
        # On non-Windows, falls back to ~/.hermes
        if os.name != "nt":
            assert _agent_root("hermes", home=fake_home) == fake_home / ".hermes"

    def test_openclaw(self, fake_home):
        assert _agent_root("openclaw", home=fake_home) == fake_home / ".openclaw"

    def test_claude_code(self, fake_home):
        assert _agent_root("claude-code", home=fake_home) == fake_home / ".claude"

    def test_codex(self, fake_home):
        assert _agent_root("codex", home=fake_home) == fake_home / ".codex"

    def test_unknown_agent_raises(self, fake_home):
        with pytest.raises(ValueError):
            _agent_root("nonexistent", home=fake_home)
