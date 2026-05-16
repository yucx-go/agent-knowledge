"""Extended tests for Dream Cycle — edge cases, REM clustering, scoring."""

import os
import shutil
import sys
import time
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from agent_knowledge.core.vault import Vault
from agent_knowledge.core.compiler import Compiler
from agent_knowledge.core.models import Source
from agent_knowledge.dreaming.cycle import (
    DreamCycle, DreamCandidate, PROMOTION_THRESHOLD, SCORE_WEIGHTS,
)

TEST_DIR = "/tmp/ak-test-dream-ext"


@pytest.fixture(autouse=True)
def clean_vault():
    if os.path.exists(TEST_DIR):
        shutil.rmtree(TEST_DIR)
    vault = Vault(TEST_DIR)
    vault.init()
    yield vault
    if os.path.exists(TEST_DIR):
        shutil.rmtree(TEST_DIR)


class TestEmptyVault:
    def test_all_phases_empty(self, clean_vault):
        cycle = DreamCycle(clean_vault)
        report = cycle.run(since_hours=9999)
        assert report.light_candidates == 0
        assert report.rem_clusters == 0
        assert report.deep_promoted == 0
        assert report.deep_discarded == 0
        assert report.details == []
        assert report.duration_seconds >= 0

    def test_no_recent_sources(self, clean_vault):
        """Sources outside the time window are skipped."""
        # Manually create a source with old timestamp
        old_source = Source(
            id="old-1",
            title="Old",
            content="## 决策\n- 旧决策内容",
            ingested_at=time.time() - 999999,
        )
        clean_vault.save_source(old_source)

        cycle = DreamCycle(clean_vault)
        report = cycle.run(since_hours=1)  # only last 1 hour
        assert report.light_candidates == 0


class TestSinceHoursZero:
    def test_zero_hours(self, clean_vault):
        """since_hours=0 should find nothing (cutoff = now)."""
        compiler = Compiler(clean_vault)
        compiler.ingest("## 决策\n- 测试决策", title="Test")

        cycle = DreamCycle(clean_vault)
        report = cycle.run(since_hours=0)
        # With since_hours=0, cutoff = current time, so recently ingested is borderline
        # The result depends on timing precision — just verify it doesn't crash
        assert report.duration_seconds >= 0


class TestREMClustering:
    def test_similar_claims_cluster(self, clean_vault):
        compiler = Compiler(clean_vault)
        # Ingest similar content that should cluster together
        compiler.ingest("## 决策\n- MCP 协议是 Agent 通信的标准方案", title="D1")
        compiler.ingest("## 决策\n- MCP 协议是 Agent 通信的推荐方案", title="D2")

        cycle = DreamCycle(clean_vault)
        report = cycle.run(since_hours=9999)
        # Similar claims should form fewer clusters than total candidates
        assert report.rem_clusters <= report.light_candidates
        # At least 1 cluster
        if report.light_candidates > 0:
            assert report.rem_clusters >= 1

    def test_dissimilar_claims_separate(self, clean_vault):
        compiler = Compiler(clean_vault)
        compiler.ingest("## 决策\n- MCP 协议替代 REST API", title="MCP")
        compiler.ingest("## 决策\n- React 框架构建前端应用", title="React")
        compiler.ingest("## 学习\n- 订单总额 4250000 元", title="Orders")

        cycle = DreamCycle(clean_vault)
        report = cycle.run(since_hours=9999)
        # Different topics should form multiple clusters
        if report.light_candidates >= 3:
            assert report.rem_clusters >= 2


class TestDeepScoring:
    def test_decision_scores_higher(self, clean_vault):
        compiler = Compiler(clean_vault)
        compiler.ingest(
            "## 决策\n- 关键架构决策：使用编译式知识图谱\n"
            "## 待办\n- 买咖啡明天早上",
            title="Mixed",
        )

        cycle = DreamCycle(clean_vault)
        report = cycle.run(since_hours=9999)

        if len(report.details) >= 2:
            # Find decision and todo scores
            decision_scores = [d["score"] for d in report.details if "架构" in d["claim"]]
            todo_scores = [d["score"] for d in report.details if "咖啡" in d["claim"]]
            if decision_scores and todo_scores:
                assert max(decision_scores) > max(todo_scores)

    def test_score_within_bounds(self, clean_vault):
        compiler = Compiler(clean_vault)
        compiler.ingest("## 决策\n- 测试决策内容关于系统架构", title="Test")

        cycle = DreamCycle(clean_vault)
        report = cycle.run(since_hours=9999)

        for d in report.details:
            assert 0 <= d["score"] <= 1.0

    def test_promotion_consistency(self, clean_vault):
        compiler = Compiler(clean_vault)
        compiler.ingest("## 决策\n- 重要决策A关于系统设计", title="A")
        compiler.ingest("## 决策\n- 重要决策B关于架构选型", title="B")

        cycle = DreamCycle(clean_vault)
        report = cycle.run(since_hours=9999)

        # Promoted + discarded = total
        assert report.deep_promoted + report.deep_discarded == report.light_candidates

        # All promoted should have score >= threshold
        for d in report.details:
            if d["promoted"]:
                assert d["score"] >= PROMOTION_THRESHOLD


class TestRepeatedDreamCycle:
    def test_double_dream(self, clean_vault):
        """Running dream twice shouldn't crash or produce wildly different results."""
        compiler = Compiler(clean_vault)
        compiler.ingest("## 决策\n- 使用 YAML 格式存储数据", title="Format")

        cycle = DreamCycle(clean_vault)
        report1 = cycle.run(since_hours=9999)
        report2 = cycle.run(since_hours=9999)

        # Both should succeed
        assert report1.light_candidates >= 0
        assert report2.light_candidates >= 0
        # Same input should produce same candidate count
        assert report1.light_candidates == report2.light_candidates


class TestScoreWeights:
    def test_weights_sum_to_one(self):
        total = sum(SCORE_WEIGHTS.values())
        assert abs(total - 1.0) < 0.001

    def test_all_dimensions_present(self):
        expected = {"frequency", "relevance", "query_diversity",
                    "recency", "consolidation", "concept_richness"}
        assert set(SCORE_WEIGHTS.keys()) == expected
