"""Tests for EverOS-inspired improvements: durability, supersession, foresight, decay."""

import os
import shutil
import time
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from agent_knowledge.core.models import (
    Claim,
    ClaimDurability,
    ClaimStatus,
    CompiledTruth,
    Evidence,
    ForesightItem,
    ForesightStatus,
)
from agent_knowledge.core.vault import Vault
from agent_knowledge.core.compiler import Compiler
from agent_knowledge.search.consolidator import MemoryConsolidator
from agent_knowledge.dreaming.cycle import DreamCycle, DreamCandidate

TEST_DIR = "/tmp/ak-test-everos"


@pytest.fixture(autouse=True)
def clean_vault():
    if os.path.exists(TEST_DIR):
        shutil.rmtree(TEST_DIR)
    vault = Vault(TEST_DIR)
    vault.init()
    yield vault
    if os.path.exists(TEST_DIR):
        shutil.rmtree(TEST_DIR)


# ============================================================
# P0-1: Claim durability
# ============================================================

class TestClaimDurability:
    def test_durability_enum_values(self):
        assert ClaimDurability.STABLE.value == "stable"
        assert ClaimDurability.TEMPORARY.value == "temporary"
        assert ClaimDurability.UNKNOWN.value == "unknown"

    def test_claim_default_durability(self):
        c = Claim(text="some fact")
        assert c.durability == ClaimDurability.UNKNOWN

    def test_claim_to_dict_includes_durability(self):
        c = Claim(text="test", durability=ClaimDurability.STABLE)
        d = c.to_dict()
        assert d["durability"] == "stable"

    def test_claim_from_dict_with_durability(self):
        d = {"text": "test", "durability": "temporary"}
        c = Claim.from_dict(d)
        assert c.durability == ClaimDurability.TEMPORARY

    def test_claim_from_dict_without_durability(self):
        d = {"text": "test"}
        c = Claim.from_dict(d)
        assert c.durability == ClaimDurability.UNKNOWN

    def test_decision_claims_are_stable(self, clean_vault):
        compiler = Compiler(clean_vault)
        text = """## 决策
- 使用 MCP 协议替代 REST API
"""
        source = compiler.ingest(text, title="Test")
        entities = clean_vault.list_entities()
        # Check claims via re-extraction
        claims = compiler._extract_claims(text, "test")
        decision_claims = [c for c in claims if "decision" in c.tags]
        for c in decision_claims:
            assert c.durability == ClaimDurability.STABLE

    def test_todo_claims_are_temporary(self, clean_vault):
        compiler = Compiler(clean_vault)
        text = """## 待办
- 跟进 prod 环境异常并修复
"""
        claims = compiler._extract_claims(text, "test")
        todo_claims = [c for c in claims if "todo" in c.tags]
        for c in todo_claims:
            assert c.durability == ClaimDurability.TEMPORARY

    def test_temporal_words_make_temporary(self, clean_vault):
        compiler = Compiler(clean_vault)
        text = """## 学习
- 最近发现 BM25 对中文效果一般需要调整分词器
"""
        claims = compiler._extract_claims(text, "test")
        assert len(claims) >= 1
        assert claims[0].durability == ClaimDurability.TEMPORARY

    def test_plain_learning_is_unknown(self, clean_vault):
        compiler = Compiler(clean_vault)
        text = """## 学习
- BM25 对中文分词效果一般需要调优参数配置方案
"""
        claims = compiler._extract_claims(text, "test")
        assert len(claims) >= 1
        assert claims[0].durability == ClaimDurability.UNKNOWN

    def test_durability_roundtrip_via_vault(self, clean_vault):
        """Ensure durability survives save/load cycle."""
        from agent_knowledge.core.models import Entity
        entity = Entity(name="TestEntity", entity_type="tech")
        claim = Claim(text="test fact", durability=ClaimDurability.STABLE)
        entity.compiled_truth.claims.append(claim)
        clean_vault.save_entity(entity)
        clean_vault.clear_cache()  # force reload from disk
        loaded = clean_vault.load_entity(entity.id)
        assert loaded is not None
        assert loaded.compiled_truth.claims[0].durability == ClaimDurability.STABLE


# ============================================================
# P0-2: Consolidator time-priority supersession
# ============================================================

class TestSupersession:
    def test_newer_supersedes_older(self):
        facts = [
            {"text": "MCP is the standard protocol", "source_id": "s1", "timestamp": 100},
            {"text": "MCP is the standard protocol for agents", "source_id": "s2", "timestamp": 200},
        ]
        consolidated = MemoryConsolidator.consolidate(facts)
        auth = [c for c in consolidated if c.weight == 1.0]
        superseded = [c for c in consolidated if c.weight == 0.3]
        assert len(auth) == 1
        assert auth[0].timestamp == 200  # newer is authoritative
        assert len(superseded) == 1
        assert superseded[0].timestamp == 100

    def test_superseded_marked_in_compiler(self, clean_vault):
        """Temporary claims use lower threshold (0.5) for supersession."""
        from agent_knowledge.core.models import Entity
        entity = Entity(name="TestProject", entity_type="project")

        # Create two similar temporary claims
        old_claim = Claim(
            text="这周很忙需要加班处理紧急任务",
            durability=ClaimDurability.TEMPORARY,
            created_at=time.time() - 86400,
        )
        old_claim.evidence.append(Evidence(source_id="s1", text=old_claim.text, weight=0.7))

        new_claim = Claim(
            text="这周很忙需要加班处理多个项目",
            durability=ClaimDurability.TEMPORARY,
            created_at=time.time(),
        )
        new_claim.evidence.append(Evidence(source_id="s2", text=new_claim.text, weight=0.7))

        entity.compiled_truth.claims = [old_claim, new_claim]
        Compiler._consolidate_entity_claims(entity)

        # With Jaccard between the two > 0.5, the old one should be superseded
        sim = MemoryConsolidator._jaccard_similarity(old_claim.text, new_claim.text)
        if sim >= 0.5:
            assert old_claim.status == ClaimStatus.SUPERSEDED

    def test_dissimilar_facts_not_superseded(self):
        facts = [
            {"text": "Python is great for scripting", "source_id": "s1", "timestamp": 100},
            {"text": "Rust is fast for systems programming", "source_id": "s2", "timestamp": 200},
        ]
        consolidated = MemoryConsolidator.consolidate(facts)
        auth = [c for c in consolidated if c.weight == 1.0]
        assert len(auth) == 2  # both remain authoritative


# ============================================================
# P1: Foresight
# ============================================================

class TestForesight:
    def test_foresight_item_creation(self):
        fi = ForesightItem(
            text="下周完成报告",
            valid_until=time.time() + 7 * 86400,
            source_id="s1",
        )
        assert fi.status == ForesightStatus.PENDING

    def test_foresight_serialization(self):
        fi = ForesightItem(
            text="计划迁移数据库",
            valid_until=time.time() + 14 * 86400,
            source_id="s1",
            status=ForesightStatus.PENDING,
        )
        d = fi.to_dict()
        assert d["status"] == "pending"
        assert d["text"] == "计划迁移数据库"

        fi2 = ForesightItem.from_dict(d)
        assert fi2.text == fi.text
        assert fi2.status == ForesightStatus.PENDING

    def test_foresight_extraction_next_week(self, clean_vault):
        compiler = Compiler(clean_vault)
        fi = compiler._extract_foresight("下周要完成代码审查", "s1")
        assert fi is not None
        assert fi.valid_until > time.time()
        # Should be ~7 days from now
        delta = fi.valid_until - time.time()
        assert 6 * 86400 < delta < 8 * 86400

    def test_foresight_extraction_next_month(self, clean_vault):
        compiler = Compiler(clean_vault)
        fi = compiler._extract_foresight("下个月计划重构搜索模块", "s1")
        assert fi is not None
        delta = fi.valid_until - time.time()
        assert 29 * 86400 < delta < 31 * 86400

    def test_foresight_extraction_default(self, clean_vault):
        compiler = Compiler(clean_vault)
        fi = compiler._extract_foresight("需要优化查询性能", "s1")
        assert fi is not None
        delta = fi.valid_until - time.time()
        assert 13 * 86400 < delta < 15 * 86400  # default 14 days

    def test_no_foresight_for_plain_text(self, clean_vault):
        compiler = Compiler(clean_vault)
        fi = compiler._extract_foresight("BM25 对中文效果一般", "s1")
        assert fi is None

    def test_foresight_in_compiled_truth(self):
        ct = CompiledTruth()
        fi = ForesightItem(text="TODO: 完成文档", valid_until=time.time() + 86400, source_id="s1")
        ct.foresight.append(fi)
        assert len(ct.foresight) == 1

    def test_foresight_vault_roundtrip(self, clean_vault):
        """Foresight survives save/load cycle."""
        from agent_knowledge.core.models import Entity
        entity = Entity(name="ForesightTest", entity_type="project")
        fi = ForesightItem(
            text="计划下周发布 v0.2",
            valid_until=time.time() + 7 * 86400,
            source_id="s1",
        )
        entity.compiled_truth.foresight.append(fi)
        clean_vault.save_entity(entity)
        clean_vault.clear_cache()
        loaded = clean_vault.load_entity(entity.id)
        assert loaded is not None
        assert len(loaded.compiled_truth.foresight) == 1
        assert loaded.compiled_truth.foresight[0].text == "计划下周发布 v0.2"
        assert loaded.compiled_truth.foresight[0].status == ForesightStatus.PENDING

    def test_foresight_expiry_in_dream(self, clean_vault):
        """Dream cycle Light phase expires overdue foresight items."""
        from agent_knowledge.core.models import Entity
        entity = Entity(name="ExpiryTest", entity_type="project")
        # Add an already-expired foresight item
        fi = ForesightItem(
            text="应该在上周完成的任务",
            valid_until=time.time() - 86400,  # expired yesterday
            source_id="s1",
        )
        entity.compiled_truth.foresight.append(fi)
        clean_vault.save_entity(entity)

        cycle = DreamCycle(clean_vault)
        cycle._expire_foresight()

        clean_vault.clear_cache()
        loaded = clean_vault.load_entity(entity.id)
        assert loaded.compiled_truth.foresight[0].status == ForesightStatus.EXPIRED

    def test_foresight_pending_not_expired(self, clean_vault):
        """Foresight with future valid_until stays pending."""
        from agent_knowledge.core.models import Entity
        entity = Entity(name="PendingTest", entity_type="project")
        fi = ForesightItem(
            text="下月要做的事",
            valid_until=time.time() + 30 * 86400,
            source_id="s1",
        )
        entity.compiled_truth.foresight.append(fi)
        clean_vault.save_entity(entity)

        cycle = DreamCycle(clean_vault)
        cycle._expire_foresight()

        clean_vault.clear_cache()
        loaded = clean_vault.load_entity(entity.id)
        assert loaded.compiled_truth.foresight[0].status == ForesightStatus.PENDING


# ============================================================
# P2: Durability-based decay in Dream Cycle
# ============================================================

class TestDurabilityDecay:
    def _make_candidate(self, durability: ClaimDurability, age_days: float) -> DreamCandidate:
        claim = Claim(
            text="test claim for decay",
            durability=durability,
            created_at=time.time() - age_days * 86400,
            tags=["decision"],
        )
        claim.evidence.append(Evidence(source_id="s1", text="test", weight=0.7))
        dc = DreamCandidate(claim=claim, source_id="s1")
        dc._cluster_size = 1
        dc._cross_cluster_entities = set()
        return dc

    def test_temporary_decays_faster(self, clean_vault):
        """Temporary claims should have lower recency score at same age."""
        cycle = DreamCycle(clean_vault)
        age = 5.0  # 5 days old

        temp = self._make_candidate(ClaimDurability.TEMPORARY, age)
        stable = self._make_candidate(ClaimDurability.STABLE, age)
        unknown = self._make_candidate(ClaimDurability.UNKNOWN, age)

        score_temp = cycle._score_candidate(temp)
        score_stable = cycle._score_candidate(stable)
        score_unknown = cycle._score_candidate(unknown)

        # At 5 days: temporary (half-life 3) < unknown (half-life 7) < stable (half-life 30)
        assert score_temp < score_unknown < score_stable

    def test_temporary_half_life_3_days(self, clean_vault):
        """At 3 days, temporary claim recency should be ~0.5."""
        cycle = DreamCycle(clean_vault)
        dc = self._make_candidate(ClaimDurability.TEMPORARY, 3.0)
        score = cycle._score_candidate(dc)
        # Extract recency component: 2^(-3/3) = 0.5
        recency = 2 ** (-3.0 / 3.0)
        assert abs(recency - 0.5) < 0.01

    def test_stable_half_life_30_days(self, clean_vault):
        """At 30 days, stable claim recency should be ~0.5."""
        cycle = DreamCycle(clean_vault)
        dc = self._make_candidate(ClaimDurability.STABLE, 30.0)
        # 2^(-30/30) = 0.5
        recency = 2 ** (-30.0 / 30.0)
        assert abs(recency - 0.5) < 0.01

    def test_unknown_half_life_7_days(self, clean_vault):
        """At 7 days, unknown claim recency should be ~0.5."""
        cycle = DreamCycle(clean_vault)
        dc = self._make_candidate(ClaimDurability.UNKNOWN, 7.0)
        recency = 2 ** (-7.0 / 7.0)
        assert abs(recency - 0.5) < 0.01

    def test_fresh_claims_score_similar(self, clean_vault):
        """At age 0, all durabilities should have similar scores."""
        cycle = DreamCycle(clean_vault)
        temp = self._make_candidate(ClaimDurability.TEMPORARY, 0.0)
        stable = self._make_candidate(ClaimDurability.STABLE, 0.0)
        unknown = self._make_candidate(ClaimDurability.UNKNOWN, 0.0)

        s_temp = cycle._score_candidate(temp)
        s_stable = cycle._score_candidate(stable)
        s_unknown = cycle._score_candidate(unknown)

        # All should be very close when age=0 (recency=1.0 for all)
        assert abs(s_temp - s_stable) < 0.01
        assert abs(s_temp - s_unknown) < 0.01
