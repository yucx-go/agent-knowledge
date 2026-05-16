"""Tests for QueryRewriter."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from agent_knowledge.search.query_rewriter import QueryRewriter


class TestQueryRewriter:
    """Test query rewriting for multi-route search."""

    def test_basic_rewrite(self):
        rq = QueryRewriter.rewrite("dashboard widget chart height 格式要求")
        assert rq.original == "dashboard widget chart height 格式要求"
        assert "chart" in rq.keyword
        assert "height" in rq.keyword

    def test_strip_chinese_stopwords(self):
        rq = QueryRewriter.rewrite("订单服务的 QueryKey 是什么")
        assert "QueryKey" in rq.keyword
        assert "订单服务" in rq.keyword
        # Stop words should be stripped
        assert "什么" not in rq.keyword.split()

    def test_extract_entities(self):
        rq = QueryRewriter.rewrite("collapsible_panel header 的限制")
        assert "collapsible_panel" in rq.entity

    def test_extract_hex_ids(self):
        rq = QueryRewriter.rewrite("query key deadbeefcafef00d0123456789abcdef")
        assert "deadbeefcafef00d0123456789abcdef" in rq.entity

    def test_extract_prefixed_ids(self):
        rq = QueryRewriter.rewrite("用户 ou_abc123 的信息")
        assert "ou_abc123" in rq.entity

    def test_empty_entity_for_plain_query(self):
        rq = QueryRewriter.rewrite("月度预算达成率")
        # No technical identifiers
        assert rq.entity == ""

    def test_semantic_passthrough(self):
        rq = QueryRewriter.rewrite("为什么选了 React？")
        assert rq.semantic == "为什么选了 React？"

    def test_english_stopwords(self):
        rq = QueryRewriter.rewrite("what is the budget for Q1")
        assert "budget" in rq.keyword
        assert "Q1" in rq.keyword
