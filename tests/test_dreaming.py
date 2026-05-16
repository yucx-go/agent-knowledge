"""Tests for Dream Cycle."""

import os
import shutil
import sys
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from agent_knowledge.core.vault import Vault
from agent_knowledge.core.compiler import Compiler
from agent_knowledge.dreaming.cycle import DreamCycle, PROMOTION_THRESHOLD

TEST_DIR = "/tmp/ak-test-dream"


@pytest.fixture(autouse=True)
def clean_vault():
    if os.path.exists(TEST_DIR):
        shutil.rmtree(TEST_DIR)
    vault = Vault(TEST_DIR)
    vault.init()
    yield vault
    if os.path.exists(TEST_DIR):
        shutil.rmtree(TEST_DIR)


class TestDreamCycle:
    def test_empty_vault(self, clean_vault):
        cycle = DreamCycle(clean_vault)
        report = cycle.run()
        assert report.light_candidates == 0
        assert report.deep_promoted == 0

    def test_dream_with_data(self, clean_vault):
        compiler = Compiler(clean_vault)
        compiler.ingest(
            "## 决策\n- MCP 协议替代 REST\n- 选择 React 框架\n## 学习\n- BM25 中文效果一般",
            title="Day 1",
        )
        compiler.ingest(
            "## 决策\n- 开启 Dreaming 功能\n## 操作\n- 修改 openclaw.json 配置",
            title="Day 2",
        )

        cycle = DreamCycle(clean_vault)
        report = cycle.run(since_hours=9999)

        assert report.light_candidates > 0
        assert report.deep_promoted + report.deep_discarded == report.light_candidates
        assert report.duration_seconds >= 0

    def test_promotion_threshold(self, clean_vault):
        compiler = Compiler(clean_vault)
        # Decisions should score higher than todos
        compiler.ingest(
            "## 决策\n- 这是一个重要的架构决策关于系统设计方向\n## 待办\n- 买咖啡",
            title="Mixed",
        )

        cycle = DreamCycle(clean_vault)
        report = cycle.run(since_hours=9999)

        # Decision should be promoted, todo might not
        assert report.deep_promoted >= 1

    def test_dream_report_details(self, clean_vault):
        compiler = Compiler(clean_vault)
        compiler.ingest("## 决策\n- 使用 YAML 格式存储知识库数据", title="Format Decision")

        cycle = DreamCycle(clean_vault)
        report = cycle.run(since_hours=9999)

        assert len(report.details) > 0
        for d in report.details:
            assert "claim" in d
            assert "score" in d
            assert "promoted" in d
            assert 0 <= d["score"] <= 1

    def test_scoring_dimensions(self, clean_vault):
        """Verify that different content types get different scores."""
        compiler = Compiler(clean_vault)
        compiler.ingest(
            "## 决策\n- 关键架构决策：使用编译式知识图谱而非纯 RAG 方案\n"
            "## 待办\n- 检查一下邮件",
            title="Varied",
        )

        cycle = DreamCycle(clean_vault)
        report = cycle.run(since_hours=9999)

        # Should have at least 2 candidates with different scores
        if len(report.details) >= 2:
            scores = [d["score"] for d in report.details]
            assert max(scores) > min(scores), "Different content types should get different scores"
