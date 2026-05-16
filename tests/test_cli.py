"""Tests for CLI commands."""

import os
import shutil
import sys
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

TEST_DIR = "/tmp/ak-test-cli"


@pytest.fixture(autouse=True)
def clean_vault():
    if os.path.exists(TEST_DIR):
        shutil.rmtree(TEST_DIR)
    yield
    if os.path.exists(TEST_DIR):
        shutil.rmtree(TEST_DIR)


class TestCLI:
    def _run(self, args: list[str]) -> None:
        """Run CLI with args, catching SystemExit."""
        from agent_knowledge.cli import main
        sys.argv = args
        try:
            main()
        except SystemExit as e:
            if e.code and e.code != 0:
                raise

    def test_init(self, capsys):
        self._run(["ak", "init", TEST_DIR])
        captured = capsys.readouterr()
        assert "Vault initialized" in captured.out

    def test_ingest_text(self, capsys):
        self._run(["ak", "init", TEST_DIR])
        self._run(["ak", "ingest", TEST_DIR, "-t",
                    "## 决策\n- MCP 协议是 Agent 通信标准方案"])
        captured = capsys.readouterr()
        assert "Ingested" in captured.out

    def test_ingest_file(self, capsys, tmp_path):
        self._run(["ak", "init", TEST_DIR])
        f = tmp_path / "test.md"
        f.write_text("## 学习\n- BM25 对中文检索效果需要优化")
        self._run(["ak", "ingest", TEST_DIR, "-f", str(f)])
        captured = capsys.readouterr()
        assert "Ingested" in captured.out

    def test_query(self, capsys):
        self._run(["ak", "init", TEST_DIR])
        self._run(["ak", "ingest", TEST_DIR, "-t",
                    "## 决策\n- 使用 React 框架构建前端应用"])
        self._run(["ak", "query", TEST_DIR, "React"])
        captured = capsys.readouterr()
        assert "React" in captured.out

    def test_stats(self, capsys):
        self._run(["ak", "init", TEST_DIR])
        self._run(["ak", "stats", TEST_DIR])
        captured = capsys.readouterr()
        assert "sources: 0" in captured.out

    def test_dream(self, capsys):
        self._run(["ak", "init", TEST_DIR])
        self._run(["ak", "ingest", TEST_DIR, "-t",
                    "## 决策\n- 重要架构决策：选择编译式知识图谱方案"])
        self._run(["ak", "dream", TEST_DIR, "--hours", "9999"])
        captured = capsys.readouterr()
        assert "Dream Report" in captured.out

    def test_lint(self, capsys):
        self._run(["ak", "init", TEST_DIR])
        self._run(["ak", "lint", TEST_DIR])
        captured = capsys.readouterr()
        assert "Lint" in captured.out

    def test_batch_ingest(self, capsys, tmp_path):
        self._run(["ak", "init", TEST_DIR])
        # Create test files
        for i in range(3):
            f = tmp_path / f"doc-{i}.md"
            f.write_text(f"## 决策\n- 决策 {i} 关于系统架构的重要选择")
        self._run(["ak", "batch-ingest", TEST_DIR, str(tmp_path)])
        captured = capsys.readouterr()
        assert "3 ingested" in captured.out
