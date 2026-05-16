"""Tests for MemoryConsolidator."""

import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from agent_knowledge.search.consolidator import MemoryConsolidator, ConsolidatedFact


class TestMemoryConsolidator:
    """Test fact deduplication and consolidation."""

    def test_empty_input(self):
        result = MemoryConsolidator.consolidate([])
        assert result == []

    def test_single_fact(self):
        facts = [{"text": "React 被选为前端框架", "source_id": "s1", "timestamp": 100}]
        result = MemoryConsolidator.consolidate(facts)
        assert len(result) == 1
        assert result[0].fact == "React 被选为前端框架"
        assert result[0].weight == 1.0

    def test_duplicate_facts_keep_newest(self):
        facts = [
            {"text": "决定用 React 替代 Vue，原因是团队更熟悉", "source_id": "s1", "timestamp": 100},
            {"text": "决定用 React 替代 Vue，原因是团队更熟悉 React", "source_id": "s2", "timestamp": 200},
        ]
        result = MemoryConsolidator.consolidate(facts)
        # Newest version (s2) should be authoritative
        authoritative = [f for f in result if f.weight == 1.0]
        assert len(authoritative) == 1
        assert authoritative[0].source == "s2"

    def test_superseded_facts_downweighted(self):
        facts = [
            {"text": "决定用 React 替代 Vue，原因是团队更熟悉", "source_id": "s1", "timestamp": 100},
            {"text": "决定用 React 替代 Vue，原因是团队更熟悉 React 生态系统", "source_id": "s2", "timestamp": 200},
        ]
        result = MemoryConsolidator.consolidate(facts)
        superseded = [f for f in result if f.supersedes == "superseded"]
        assert len(superseded) == 1
        assert superseded[0].weight < 1.0

    def test_different_facts_not_merged(self):
        facts = [
            {"text": "React 是前端框架", "source_id": "s1", "timestamp": 100},
            {"text": "PostgreSQL 是数据库", "source_id": "s2", "timestamp": 200},
        ]
        result = MemoryConsolidator.consolidate(facts)
        authoritative = [f for f in result if f.weight == 1.0]
        assert len(authoritative) == 2

    def test_jaccard_similarity(self):
        # High similarity
        sim = MemoryConsolidator._jaccard_similarity(
            "决定用 React 替代 Vue",
            "决定用 React 替代 Vue 框架"
        )
        assert sim > 0.5

        # Low similarity
        sim = MemoryConsolidator._jaccard_similarity(
            "React 前端框架",
            "PostgreSQL 数据库"
        )
        assert sim < 0.3
