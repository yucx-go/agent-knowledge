"""QueryRewriter: expand user query into multi-route variants for better recall.

Generates:
1. Keyword query — stripped of filler words, focused on key terms
2. Entity query — extracted technical identifiers for exact match
3. Semantic query — natural language reformulation

Usage example::

    from agent_knowledge.search.query_rewriter import QueryRewriter

    rq = QueryRewriter.rewrite("记忆系统怎么优化")
    print(rq.keyword)   # "记忆系统优化" (stop words removed)
    print(rq.entity)    # "" (no technical identifiers)
    print(rq.semantic)   # "记忆系统怎么优化" (pass-through)

    rq2 = QueryRewriter.rewrite("collapsible_panel height bug")
    print(rq2.entity)   # "collapsible_panel" (snake_case identifier)
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class RewrittenQuery:
    """A set of query variants for multi-route search."""
    original: str
    keyword: str      # stripped keywords
    entity: str       # technical identifiers only
    semantic: str      # natural language (currently same as original)


class QueryRewriter:
    """Rewrite queries for better multi-route recall."""

    # Chinese stop words / filler
    _CN_STOPWORDS = {
        "的", "是", "了", "在", "有", "和", "与", "及", "或",
        "什", "么", "怎", "如", "何", "哪", "个", "些", "为",
        "什么", "怎么", "如何", "哪个", "哪些", "为什么",
        "用", "要", "能", "会", "可", "以", "应", "该",
        "可以", "应该",
        "多", "少", "几", "时", "候", "地", "方",
        "多少", "几个", "时候", "地方",
    }

    # English stop words
    _EN_STOPWORDS = {
        "the", "a", "an", "is", "are", "was", "were", "be", "been",
        "what", "how", "which", "when", "where", "why", "who",
        "do", "does", "did", "can", "could", "should", "would",
        "for", "to", "of", "in", "on", "at", "by", "with", "from",
    }

    @classmethod
    def rewrite(cls, query: str) -> RewrittenQuery:
        """Generate query variants."""
        original = query.strip()

        # Keyword version: strip stop words, keep content words
        keyword = cls._extract_keywords(original)

        # Entity version: extract technical identifiers
        entity = cls._extract_entity_query(original)

        # Semantic version: currently pass-through (can add LLM rewrite later)
        semantic = original

        return RewrittenQuery(
            original=original,
            keyword=keyword,
            entity=entity,
            semantic=semantic,
        )

    @classmethod
    def _extract_keywords(cls, query: str) -> str:
        """Strip filler words, keep content-bearing terms."""
        # Tokenize: split CJK into individual characters, keep latin words
        tokens = re.findall(r'[\u4e00-\u9fff]|[a-zA-Z0-9_.-]+', query)

        # Rebuild CJK compound terms by joining non-stopword chars
        keywords = []
        cjk_buffer = []

        def flush_cjk():
            if cjk_buffer:
                # Join consecutive CJK chars, but filter out stopword subsequences
                term = ''.join(cjk_buffer)
                # Remove trailing stopword chars (的, 了, 是, 吗, 呢, etc.)
                term = re.sub(r'[的了是吗呢啊吧]+$', '', term)
                if term:
                    keywords.append(term)
                cjk_buffer.clear()

        for token in tokens:
            if len(token) == 1 and '\u4e00' <= token <= '\u9fff':
                # Check if this single char is a stopword on its own
                if token in cls._CN_STOPWORDS:
                    flush_cjk()
                else:
                    cjk_buffer.append(token)
            else:
                flush_cjk()
                lower = token.lower()
                if lower not in cls._EN_STOPWORDS and len(token) > 0:
                    keywords.append(token)

        flush_cjk()
        return " ".join(keywords)

    @classmethod
    def _extract_entity_query(cls, query: str) -> str:
        """Extract technical identifiers for exact match."""
        entities = []
        seen = set()

        # Snake_case identifiers
        for m in re.finditer(r'[a-zA-Z][a-zA-Z0-9]*(?:_[a-zA-Z0-9]+)+', query):
            e = m.group()
            if e.lower() not in seen:
                seen.add(e.lower())
                entities.append(e)

        # Prefixed IDs
        for m in re.finditer(r'(?:tbl|ou_|oc_|om_)[a-zA-Z0-9]+', query):
            e = m.group()
            if e.lower() not in seen:
                seen.add(e.lower())
                entities.append(e)

        # Hex strings (8+ chars)
        for m in re.finditer(r'\b[a-f0-9]{8,}\b', query, re.IGNORECASE):
            e = m.group()
            if e.lower() not in seen:
                seen.add(e.lower())
                entities.append(e)

        # Numbers with context
        for m in re.finditer(r'\d+(?:\.\d+)?(?:%|px|GB|MB)', query):
            e = m.group()
            if e not in seen:
                seen.add(e)
                entities.append(e)

        return " ".join(entities) if entities else ""
