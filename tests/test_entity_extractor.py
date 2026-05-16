"""Tests for the reverse-filter entity extractor."""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from agent_knowledge.core.entity_extractor import (
    STOPWORDS_CN,
    STOPWORDS_EN,
    extract_entities,
    is_non_entity,
    tag_type,
)


def names_only(result: list[tuple[str, str]]) -> set[str]:
    return {n for n, _ in result}


def types_for(result: list[tuple[str, str]], name: str) -> set[str]:
    return {t for n, t in result if n == name}


# ── Negative filter ──


class TestIsNonEntity:
    def test_short(self):
        assert is_non_entity("a") is True
        assert is_non_entity("") is True

    def test_pure_number(self):
        assert is_non_entity("123") is True
        assert is_non_entity("1,000") is True
        assert is_non_entity("12.34") is True

    def test_cn_stopword(self):
        assert is_non_entity("我们") is True
        assert is_non_entity("但是") is True
        assert is_non_entity("情况") is True

    def test_en_stopword(self):
        assert is_non_entity("The") is True
        assert is_non_entity("And") is True

    def test_cn_compound_of_stopwords(self):
        # "我们这个" = "我们" + "这个", both stopwords
        assert is_non_entity("我们这个") is True

    def test_cn_fragment_starter(self):
        assert is_non_entity("是的") is True  # starts with 是
        assert is_non_entity("在线") is True  # starts with 在

    def test_real_entities_pass(self):
        assert is_non_entity("MCP") is False
        assert is_non_entity("财务部") is False
        assert is_non_entity("PostgreSQL") is False
        assert is_non_entity("记忆系统") is False


# ── Type tagging ──


class TestTagType:
    def test_cjk_org_suffix(self):
        assert tag_type("招商银行") == "org"
        assert tag_type("阿里巴巴集团") == "org"
        assert tag_type("北京大学") == "org"
        assert tag_type("财务部") == "org"

    def test_cjk_place_suffix(self):
        assert tag_type("上海市") == "place"
        assert tag_type("浙江省") == "place"
        assert tag_type("朝阳区") == "place"

    def test_cjk_document_suffix(self):
        assert tag_type("年度报告") == "document"
        assert tag_type("用户协议") == "document"
        assert tag_type("操作手册") == "document"

    def test_cjk_event_suffix(self):
        assert tag_type("年终大会") == "event"
        assert tag_type("产品发布会") == "event"

    def test_cjk_person_suffix(self):
        assert tag_type("李明先生") == "person"
        assert tag_type("王老师") == "person"
        assert tag_type("张经理") == "person"

    def test_english_org_suffix(self):
        assert tag_type("Apple Inc.") == "org"
        assert tag_type("Google LLC") == "org"

    def test_default_named(self):
        assert tag_type("MCP") == "named"
        assert tag_type("PostgreSQL") == "named"
        assert tag_type("记忆系统") == "named"


# ── Typed extractors (high precision) ──


class TestTypedExtractors:
    def test_email(self):
        result = extract_entities("Contact me at alice@example.com for details")
        assert ("alice@example.com", "identifier") in result

    def test_url(self):
        result = extract_entities("See https://example.com/docs for more")
        assert any(name.startswith("https://") and t == "identifier" for name, t in result)

    def test_iso_date(self):
        result = extract_entities("Released on 2026-05-16")
        assert ("2026-05-16", "time") in result

    def test_chinese_full_date(self):
        result = extract_entities("发布日期是 2026年5月16日")
        assert ("2026年5月16日", "time") in result

    def test_chinese_partial_date(self):
        result = extract_entities("2026年的财务总结")
        assert ("2026年", "time") in result

    def test_quarter(self):
        result = extract_entities("Released by Q3 2026 deadline")
        assert ("Q3 2026", "time") in result

    def test_money_prefix(self):
        result = extract_entities("Revenue was $95.5 billion")
        assert any(name.startswith("$95") and t == "money" for name, t in result)

    def test_money_chinese(self):
        result = extract_entities("成本约 ¥10万 / month")
        assert any("¥" in name and t == "money" for name, t in result)

    def test_percentage(self):
        result = extract_entities("增长率达到 74.5% 创历史新高")
        assert ("74.5%", "quantity") in result

    def test_version(self):
        result = extract_entities("升级到 v2.3.0 版本")
        assert ("v2.3.0", "version") in result

    def test_version_before_cjk(self):
        # CJK chars are \w in Python — \b doesn't fire between digit and CJK
        result = extract_entities("发布v2.3版本")
        assert ("v2.3", "version") in result

    def test_strategy_designation(self):
        result = extract_entities("Strategy 5 was approved")
        assert ("Strategy 5", "version") in result

    def test_ticket_id(self):
        result = extract_entities("See JIRA-1234 for the original report")
        assert ("JIRA-1234", "identifier") in result

    def test_typed_substring_dedup(self):
        # "2026年5月" should be subsumed by "2026年5月16日"
        result = extract_entities("会议在 2026年5月16日 召开")
        names = names_only(result)
        assert "2026年5月16日" in names
        assert "2026年5月" not in names


# ── CJK candidate extraction (stopword-segmented) ──


class TestCJKCandidates:
    def test_compound_with_segmentation(self):
        # The classic case: stopwords split adjacent compounds
        result = extract_entities("我们正在开发记忆系统和知识平台来提升效率")
        names = names_only(result)
        assert "记忆系统" in names
        assert "知识平台" in names
        # Stopwords should be filtered out
        assert "我们" not in names
        assert "正在" not in names

    def test_business_domain(self):
        result = extract_entities("财务部使用SAP系统记录凭证和底稿")
        names = names_only(result)
        assert "财务部" in names
        assert "SAP" in names
        assert "凭证" in names

    def test_domain_term_via_org_suffix(self):
        result = extract_entities("公司决定与招商银行建立合作")
        names = names_only(result)
        assert "招商银行" in names
        assert types_for(result, "招商银行") == {"org"}

    def test_person_with_title(self):
        result = extract_entities("会议由李明先生主持")
        names = names_only(result)
        assert "李明先生" in names
        assert types_for(result, "李明先生") == {"person"}


# ── English candidate extraction ──


class TestEnglishCandidates:
    def test_pascal_case(self):
        result = extract_entities("Use ReactNative for cross-platform")
        assert ("ReactNative", "named") in result

    def test_mixed_case_tech_names(self):
        result = extract_entities("Migrate from MySQL to PostgreSQL")
        names = names_only(result)
        assert "MySQL" in names
        assert "PostgreSQL" in names

    def test_acronym(self):
        result = extract_entities("The MCP protocol enables tool calls")
        assert ("MCP", "named") in result

    def test_org_with_suffix(self):
        result = extract_entities("Apple Inc. reported strong earnings")
        assert any(t == "org" for n, t in result if "Apple" in n)

    def test_strip_leading_article(self):
        # "The Northern Hemisphere" — "The" should be stripped
        result = extract_entities("Visit the Northern Hemisphere region")
        names = names_only(result)
        # Should not have "The Northern Hemisphere" as one entity
        assert "The Northern Hemisphere" not in names

    def test_snake_case(self):
        result = extract_entities("The compiled_truth module handles claims")
        assert ("compiled_truth", "named") in result


# ── Filter behaviour ──


class TestFilterBehaviour:
    def test_empty_text(self):
        assert extract_entities("") == []

    def test_pure_stopwords_text(self):
        # Even a long sentence of stopwords yields no entities
        result = extract_entities("这些都是我们今天可能需要处理的情况和问题")
        names = names_only(result)
        # All these are stopwords or stopword compounds
        assert "我们" not in names
        assert "今天" not in names
        assert "情况" not in names
        assert "问题" not in names
        # But "处理" was added as a stopword too — confirm consistency
        assert "处理" not in names

    def test_extra_names_promoted(self):
        # User-defined glossary names should be promoted even if they
        # would otherwise look generic
        result = extract_entities(
            "the project alpha-omega is ongoing",
            extra_names=["alpha-omega"],
        )
        assert any(n == "alpha-omega" for n, _ in result)


# ── Compound stopword decomposition ──


class TestCompoundDecomposition:
    def test_two_stopword_concat(self):
        # "我们这些" = "我们" + "这些" — should be filtered
        assert is_non_entity("我们这些") is True

    def test_three_stopword_concat(self):
        # "情况问题原因" = three stopwords concatenated
        assert is_non_entity("情况问题原因") is True

    def test_partial_stopword_keeps(self):
        # "我们公司" = "我们" + "公司" — "公司" is NOT in stopwords (it's
        # an org suffix). So this should NOT be filtered.
        assert is_non_entity("我们公司") is False
