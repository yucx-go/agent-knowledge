"""Tests for Search Engine."""

import os
import shutil
import sys
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from agent_knowledge.core.vault import Vault
from agent_knowledge.core.compiler import Compiler
from agent_knowledge.search.engine import SearchEngine, BM25Index

TEST_DIR = "/tmp/ak-test-search"


@pytest.fixture(autouse=True)
def clean_vault():
    if os.path.exists(TEST_DIR):
        shutil.rmtree(TEST_DIR)
    vault = Vault(TEST_DIR)
    vault.init()
    yield vault
    if os.path.exists(TEST_DIR):
        shutil.rmtree(TEST_DIR)


class TestBM25Index:
    def test_empty_index(self):
        idx = BM25Index()
        results = idx.search("test")
        assert len(results) == 0

    def test_basic_search(self):
        idx = BM25Index()
        idx.add("d1", "Doc 1", "MCP protocol for agent communication", "source")
        idx.add("d2", "Doc 2", "React component library ecosystem", "source")
        idx.add("d3", "Doc 3", "MCP timeout bug and fix", "source")

        results = idx.search("MCP")
        assert len(results) >= 2
        # MCP docs should rank higher
        ids = [r.id for r in results]
        assert "d1" in ids
        assert "d3" in ids

    def test_chinese_tokenization(self):
        idx = BM25Index()
        idx.add("d1", "记忆系统", "记忆系统优化方案使用混合检索", "source")
        idx.add("d2", "订单分析", "订单数据分析看板和深度报告", "source")

        results = idx.search("记忆")
        assert len(results) >= 1
        assert results[0].id == "d1"

    def test_top_k_limit(self):
        idx = BM25Index()
        for i in range(20):
            idx.add(f"d{i}", f"Doc {i}", f"content about topic {i}", "source")

        results = idx.search("topic", top_k=5)
        assert len(results) == 5


class TestSearchEngine:
    def test_search_over_vault(self, clean_vault):
        compiler = Compiler(clean_vault)
        compiler.ingest("MCP 协议是 Agent 通信的标准方案", title="MCP Doc")
        compiler.ingest("订单总额 4250000 元", title="Orders")

        engine = SearchEngine(clean_vault)
        results = engine.search("MCP")
        assert len(results) >= 1
        assert any("MCP" in r.title for r in results)

    def test_auto_build_index(self, clean_vault):
        compiler = Compiler(clean_vault)
        compiler.ingest("Test document about search", title="Search Test")

        engine = SearchEngine(clean_vault)
        # Should auto-build on first search
        results = engine.search("search")
        assert len(results) >= 1

    def test_entity_search(self, clean_vault):
        compiler = Compiler(clean_vault)
        compiler.ingest("ExampleAgent 支持 MCP 协议", title="ExampleAgent MCP")

        engine = SearchEngine(clean_vault)
        results = engine.search("ExampleAgent")
        # Should find both entity and source
        types = {r.page_type for r in results}
        assert len(results) >= 1
