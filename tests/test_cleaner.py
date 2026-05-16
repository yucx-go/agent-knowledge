"""Tests for MemoryCleaner."""

import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from agent_knowledge.search.cleaner import MemoryCleaner


class TestMemoryCleaner:
    """Test markdown cleaning for indexing."""

    def test_strip_backticks(self):
        text = "使用 `collapsible_panel` 组件"
        cleaned = MemoryCleaner.clean(text)
        assert "collapsible_panel" in cleaned
        assert "`" not in cleaned

    def test_strip_bold(self):
        text = "这是 **重要** 的内容"
        cleaned = MemoryCleaner.clean(text)
        assert "重要" in cleaned
        assert "**" not in cleaned

    def test_strip_headers(self):
        text = "## 决策\n- 选择了 React"
        cleaned = MemoryCleaner.clean(text)
        assert cleaned.startswith("决策")
        assert "#" not in cleaned

    def test_strip_links(self):
        text = "[查看文档](https://example.com)"
        cleaned = MemoryCleaner.clean(text)
        assert "查看文档" in cleaned
        assert "https://" not in cleaned

    def test_preserve_content(self):
        text = "仪表盘组件 chart height 必须是带 px 单位的字符串"
        cleaned = MemoryCleaner.clean(text)
        assert "仪表盘组件" in cleaned
        assert "px" in cleaned
        assert "字符串" in cleaned

    def test_normalize_whitespace(self):
        text = "这里   有   多余   空格"
        cleaned = MemoryCleaner.clean(text)
        assert "  " not in cleaned

    def test_strip_code_blocks(self):
        text = "```python\nprint('hello')\n```\n正文内容"
        cleaned = MemoryCleaner.clean(text)
        assert "```" not in cleaned
        assert "print" in cleaned
        assert "正文内容" in cleaned

    def test_strip_table_separators(self):
        text = "| 头1 | 头2 |\n|---|---|\n| 值1 | 值2 |"
        cleaned = MemoryCleaner.clean(text)
        assert "|---|" not in cleaned


class TestEntityExtraction:
    """Test entity extraction from markdown."""

    def test_extract_backtick_entities(self):
        text = "使用 `collapsible_panel` 和 `lark_md` 组件"
        entities = MemoryCleaner.extract_entities(text)
        assert "collapsible_panel" in entities
        assert "lark_md" in entities

    def test_extract_snake_case(self):
        text = "调用 create_blocks 方法"
        entities = MemoryCleaner.extract_entities(text)
        assert "create_blocks" in entities

    def test_extract_prefixed_ids(self):
        text = "表格 tblXXX123 和用户 ou_abc"
        entities = MemoryCleaner.extract_entities(text)
        assert any("tbl" in e for e in entities)

    def test_no_duplicates(self):
        text = "`foo_bar` 和 foo_bar 重复"
        entities = MemoryCleaner.extract_entities(text)
        lower_entities = [e.lower() for e in entities]
        assert lower_entities.count("foo_bar") == 1


class TestCleanForIndex:
    """Test the combined clean + entity extraction."""

    def test_entities_appended(self):
        text = "使用 `collapsible_panel` 组件"
        cleaned, entities = MemoryCleaner.clean_for_index(text)
        assert "collapsible_panel" in entities
        # Cleaned text should have entities appended
        assert "collapsible_panel" in cleaned

    def test_empty_text(self):
        cleaned, entities = MemoryCleaner.clean_for_index("")
        assert cleaned == ""
        assert entities == []
