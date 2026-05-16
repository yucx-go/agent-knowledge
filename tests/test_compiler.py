"""Tests for Knowledge Compiler."""

import os
import shutil
import sys
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from agent_knowledge.core.vault import Vault
from agent_knowledge.core.compiler import Compiler

TEST_DIR = "/tmp/ak-test-compiler"


@pytest.fixture(autouse=True)
def clean_vault():
    if os.path.exists(TEST_DIR):
        shutil.rmtree(TEST_DIR)
    vault = Vault(TEST_DIR)
    vault.init()
    yield vault
    if os.path.exists(TEST_DIR):
        shutil.rmtree(TEST_DIR)


class TestClaimExtraction:
    def test_extract_decisions(self, clean_vault):
        compiler = Compiler(clean_vault)
        text = """# 2026-05-13

## 决策
- 使用 MCP 协议替代 REST API
- 选择 React 而非 Vue

## 学习
- BM25 对中文分词效果一般
"""
        source = compiler.ingest(text, title="Test Day")
        assert len(source.claims_extracted) == 3  # 2 decisions + 1 learning

    def test_extract_with_sections(self, clean_vault):
        compiler = Compiler(clean_vault)
        text = """## 操作
- 完成记忆系统优化
- 执行 batch-catch-up

## 待办
- 跟进 prod 环境异常
"""
        source = compiler.ingest(text, title="Actions")
        assert len(source.claims_extracted) >= 2  # 操作 + 待办 sections

    def test_empty_text(self, clean_vault):
        compiler = Compiler(clean_vault)
        source = compiler.ingest("", title="Empty")
        assert len(source.claims_extracted) == 0

    def test_short_lines_ignored(self, clean_vault):
        compiler = Compiler(clean_vault)
        text = "## Notes\n- ok\n- yes\n- This is a meaningful claim about system design"
        source = compiler.ingest(text, title="Short")
        # "ok" and "yes" are < 10 chars, should be filtered
        assert len(source.claims_extracted) == 1


class TestEntityExtraction:
    def test_extract_tech_entities(self, clean_vault):
        compiler = Compiler(clean_vault)
        text = "We migrated from REST to MCP protocol. OpenClaw already supports it."
        source = compiler.ingest(text, title="Migration")
        assert len(source.entities_extracted) >= 1  # at least MCP

    def test_entity_backlinks(self, clean_vault):
        compiler = Compiler(clean_vault)

        # Ingest two sources mentioning same entity
        s1 = compiler.ingest("Using MCP for agent communication", title="Doc 1")
        s2 = compiler.ingest("MCP timeout issue found and fixed", title="Doc 2")

        # Entity should have backlinks to both sources
        for eid in clean_vault.list_entities():
            entity = clean_vault.load_entity(eid)
            if entity and entity.name == "MCP":
                assert len(entity.backlinks) >= 2
                break


class TestCompiledTruth:
    def test_compiled_truth_updates(self, clean_vault):
        compiler = Compiler(clean_vault)

        text = """## 决策
- MCP 协议是 Agent 通信的标准方案
- MCP 延迟增加了 15ms 但兼容性提升
"""
        compiler.ingest(text, title="MCP Decision")

        # Check MCP entity has compiled truth
        for eid in clean_vault.list_entities():
            entity = clean_vault.load_entity(eid)
            if entity and entity.name == "MCP":
                assert entity.compiled_truth.summary != ""
                assert len(entity.compiled_truth.timeline) > 0
                break


class TestContradictionDetection:
    def test_no_contradictions_by_default(self, clean_vault):
        compiler = Compiler(clean_vault)
        compiler.ingest("## 决策\n- MCP is fast and reliable", title="D1")
        contradictions = compiler.detect_contradictions()
        assert len(contradictions) == 0
