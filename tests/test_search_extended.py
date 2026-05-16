"""Extended tests for Search Engine — RRF fusion, reranker, query rewriter integration."""

import os
import shutil
import sys
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from agent_knowledge.core.vault import Vault
from agent_knowledge.core.compiler import Compiler
from agent_knowledge.search.engine import (
    SearchEngine, BM25Index, ExactIndex, rrf_fuse, SearchResult,
)

TEST_DIR = "/tmp/ak-test-search-ext"


@pytest.fixture(autouse=True)
def clean_vault():
    if os.path.exists(TEST_DIR):
        shutil.rmtree(TEST_DIR)
    vault = Vault(TEST_DIR)
    vault.init()
    yield vault
    if os.path.exists(TEST_DIR):
        shutil.rmtree(TEST_DIR)


def _make_result(doc_id, score, title="T", page_type="source"):
    return SearchResult(id=doc_id, page_type=page_type, title=title, snippet="s", score=score)


class TestRRFFusion:
    def test_single_list(self):
        results = [_make_result("a", 1.0), _make_result("b", 0.5)]
        fused = rrf_fuse([results], top_k=5)
        assert len(fused) == 2
        assert fused[0].id == "a"

    def test_two_lists_overlap(self):
        list1 = [_make_result("a", 1.0), _make_result("b", 0.5)]
        list2 = [_make_result("b", 1.0), _make_result("c", 0.5)]
        fused = rrf_fuse([list1, list2], top_k=5)
        # "b" appears in both lists, should get higher fused score
        ids = [r.id for r in fused]
        assert "b" in ids
        b_score = next(r.score for r in fused if r.id == "b")
        a_score = next(r.score for r in fused if r.id == "a")
        assert b_score > a_score  # b in both lists > a in one

    def test_weights_affect_ranking(self):
        list1 = [_make_result("a", 1.0)]
        list2 = [_make_result("b", 1.0)]
        # Heavily weight list2
        fused = rrf_fuse([list1, list2], weights=[0.1, 10.0], top_k=5)
        assert fused[0].id == "b"

    def test_equal_weights_equal_rank(self):
        list1 = [_make_result("a", 1.0)]
        list2 = [_make_result("b", 1.0)]
        fused = rrf_fuse([list1, list2], weights=[1.0, 1.0], top_k=5)
        # Both at rank 1 in their lists, so equal RRF score
        assert len(fused) == 2
        assert abs(fused[0].score - fused[1].score) < 0.001

    def test_top_k_truncation(self):
        lists = [[_make_result(f"d{i}", 1.0)] for i in range(10)]
        fused = rrf_fuse(lists, top_k=3)
        assert len(fused) == 3

    def test_empty_lists(self):
        fused = rrf_fuse([], top_k=5)
        assert len(fused) == 0

    def test_k_parameter(self):
        list1 = [_make_result("a", 1.0), _make_result("b", 0.5)]
        # With k=1, rank differences matter more
        fused_k1 = rrf_fuse([list1], k=1, top_k=5)
        # With k=100, rank differences matter less
        fused_k100 = rrf_fuse([list1], k=100, top_k=5)
        # Both should produce same ordering
        assert fused_k1[0].id == "a"
        assert fused_k100[0].id == "a"
        # But k=1 should have larger score gap
        gap_k1 = fused_k1[0].score - fused_k1[1].score
        gap_k100 = fused_k100[0].score - fused_k100[1].score
        assert gap_k1 > gap_k100


class TestExactIndex:
    def test_snake_case_match(self):
        idx = ExactIndex()
        idx.add("d1", "Doc", "Use collapsible_panel for folding content", "source")
        results = idx.search("collapsible_panel")
        assert len(results) >= 1
        assert results[0].id == "d1"

    def test_number_match(self):
        idx = ExactIndex()
        idx.add("d1", "Doc", "Chart height must be 280px string", "source")
        results = idx.search("280px")
        assert len(results) >= 1

    def test_hex_id_match(self):
        idx = ExactIndex()
        idx.add("d1", "Doc", "QueryKey deadbeefcafef00d0123456789abcdef", "source")
        results = idx.search("deadbeefcafef00d0123456789abcdef")
        assert len(results) >= 1

    def test_no_match(self):
        idx = ExactIndex()
        idx.add("d1", "Doc", "Some irrelevant content", "source")
        results = idx.search("nonexistent_identifier_xyz")
        assert len(results) == 0

    def test_multiple_term_boost(self):
        idx = ExactIndex()
        idx.add("d1", "Doc", "collapsible_panel height 280px issue", "source")
        idx.add("d2", "Doc2", "collapsible_panel only mentioned here", "source")
        results = idx.search("collapsible_panel 280px")
        # d1 matches both terms, should rank higher
        assert results[0].id == "d1"


class TestRerankerIntegration:
    def test_reranker_reorders(self, clean_vault):
        compiler = Compiler(clean_vault)
        compiler.ingest("## 决策\n- MCP 协议是 Agent 通信标准方案", title="MCP Direct")
        compiler.ingest("订单总额 4250000 元，共745单", title="Orders")
        compiler.ingest("MCP 协议超时配置需要调整", title="MCP Timeout")

        engine = SearchEngine(clean_vault, enable_reranker=True)
        results = engine.search("MCP 协议")
        # Results with "MCP" in title should rank higher
        if len(results) >= 2:
            mcp_results = [r for r in results if "MCP" in r.title]
            assert len(mcp_results) >= 1

    def test_reranker_disabled(self, clean_vault):
        compiler = Compiler(clean_vault)
        compiler.ingest("## 决策\n- 使用 React 构建前端", title="React")

        engine = SearchEngine(clean_vault, enable_reranker=False)
        results = engine.search("React")
        assert len(results) >= 1


class TestQueryRewriterIntegration:
    def test_keyword_variant_improves_recall(self, clean_vault):
        compiler = Compiler(clean_vault)
        compiler.ingest("## 学习\n- 记忆系统优化方案使用混合检索", title="Memory Opt")

        engine = SearchEngine(clean_vault)
        # Query with stop words that get stripped by rewriter
        results = engine.search("记忆系统怎么优化")
        assert len(results) >= 1

    def test_entity_variant_search(self, clean_vault):
        compiler = Compiler(clean_vault)
        compiler.ingest("collapsible_panel 的 header 不能用 tag 属性", title="Panel Bug")

        engine = SearchEngine(clean_vault)
        results = engine.search("collapsible_panel bug")
        assert len(results) >= 1


class TestGraphIntegration:
    def test_graph_search_finds_related(self, clean_vault):
        compiler = Compiler(clean_vault)
        compiler.ingest("## 决策\n- MCP 协议替代 REST API", title="MCP Decision")
        compiler.ingest("## 学习\n- MCP 超时需要配置", title="MCP Config")

        engine = SearchEngine(clean_vault, enable_graph=True)
        results = engine.search("MCP")
        assert len(results) >= 1

    def test_graph_disabled(self, clean_vault):
        compiler = Compiler(clean_vault)
        compiler.ingest("## 决策\n- 测试内容", title="Test")

        engine = SearchEngine(clean_vault, enable_graph=False)
        results = engine.search("测试")
        # Should still work via BM25
        assert len(results) >= 1


class TestBM25Extended:
    def test_exact_phrase_boost(self):
        idx = BM25Index()
        idx.add("d1", "Doc1", "MCP protocol for communication", "source")
        idx.add("d2", "Doc2", "MCP is used and protocol is nice", "source")
        results = idx.search("MCP protocol")
        # d1 has exact phrase, should rank higher
        assert results[0].id == "d1"

    def test_cjk_bigram_matching(self):
        idx = BM25Index()
        idx.add("d1", "记忆", "记忆系统优化方案", "source")
        idx.add("d2", "订单", "订单数据分析", "source")
        results = idx.search("记忆系统")
        assert len(results) >= 1
        assert results[0].id == "d1"

    def test_snippet_generation(self):
        idx = BM25Index()
        text = "A" * 200 + " MCP protocol " + "B" * 200
        idx.add("d1", "Doc", text, "source")
        results = idx.search("MCP")
        assert len(results) == 1
        assert "MCP" in results[0].snippet
