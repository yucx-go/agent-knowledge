"""Lightweight reranker: re-score top results by multi-signal fusion.

No external model needed. Uses TF-IDF weighting, entity coverage,
position-aware scoring, and original retrieval scores.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Optional

from ..core.vault import Vault
from .engine import SearchResult


class LightReranker:
    """Re-rank search results using multiple lightweight signals."""

    def __init__(self, vault: Vault):
        self.vault = vault
        self._idf_cache: dict[str, float] | None = None
        self._doc_count: int = 0

    def _build_idf(self) -> None:
        """Build IDF (Inverse Document Frequency) from vault corpus."""
        df: Counter = Counter()
        doc_count = 0
        for sid in self.vault.list_sources():
            source = self.vault.load_source(sid)
            if source:
                doc_count += 1
                terms = set(self._tokenize(source.content))
                for t in terms:
                    df[t] += 1
        for eid in self.vault.list_entities():
            entity = self.vault.load_entity(eid)
            if entity:
                doc_count += 1
                text = entity.name + " " + entity.compiled_truth.summary
                terms = set(self._tokenize(text))
                for t in terms:
                    df[t] += 1

        self._doc_count = max(doc_count, 1)
        self._idf_cache = {}
        for term, freq in df.items():
            self._idf_cache[term] = math.log((self._doc_count + 1) / (freq + 1)) + 1.0

    def _get_idf(self, term: str) -> float:
        """Get IDF weight for a term."""
        if self._idf_cache is None:
            self._build_idf()
        return self._idf_cache.get(term, math.log(self._doc_count + 1) + 1.0)

    def rerank(self, query: str, results: list[SearchResult],
               top_k: int = 10) -> list[SearchResult]:
        """Re-rank results by multi-signal scoring."""
        if not results:
            return []

        query_terms = self._normalize_terms(query)
        query_tokens = self._tokenize(query)
        query_entities = self._extract_entities(query)

        # Normalize original scores using rank-based scoring (1.0, 0.9, 0.8, ...)
        # This preserves ordering from RRF while mapping to a useful 0-1 range
        n = len(results)
        scored = []
        for rank_idx, r in enumerate(results):
            source = self.vault.load_source(r.id)
            content = source.content if source else r.snippet

            # Rank-based normalized original score
            norm_score = max(0.0, 1.0 - rank_idx / max(n, 1))

            signals = self._compute_signals(
                query, query_terms, query_tokens, query_entities,
                content, r, norm_score
            )
            final_score = self._fuse_signals(signals)
            scored.append((r, final_score))

        scored.sort(key=lambda x: x[1], reverse=True)

        return [
            SearchResult(
                id=r.id,
                page_type=r.page_type,
                title=r.title,
                snippet=r.snippet,
                score=round(score, 4),
            )
            for r, score in scored[:top_k]
        ]

    def _compute_signals(self, query: str, query_terms: set[str],
                          query_tokens: list[str], query_entities: list[str],
                          content: str, result: SearchResult,
                          norm_original_score: float = 0.0) -> dict[str, float]:
        """Compute multiple ranking signals."""
        content_lower = content.lower()
        content_normalized = self._strip_markdown(content_lower)
        content_collapsed = re.sub(r'\s+', '', content_normalized)

        # --- Signal 1: TF-IDF weighted keyword match ---
        tfidf_score = 0.0
        tfidf_max = 0.0
        content_tokens = set(self._tokenize(content_normalized))
        for t in query_terms:
            idf = self._get_idf(t)
            tfidf_max += idf
            if t in content_tokens or t in content_collapsed:
                tfidf_score += idf
        tfidf_weighted = tfidf_score / tfidf_max if tfidf_max > 0 else 0.0

        # --- Signal 2: Entity/key-term coverage ---
        # Higher-value terms: technical identifiers, IDs, numbers
        if query_entities:
            entity_hits = sum(
                1 for e in query_entities
                if e.lower() in content_normalized or e.lower() in content_collapsed
            )
            entity_coverage = entity_hits / len(query_entities)
        else:
            entity_coverage = tfidf_weighted  # fallback to keyword overlap

        # --- Signal 3: Exact phrase match ---
        query_stripped = self._strip_markdown(query.lower())
        query_collapsed = re.sub(r'\s+', '', query_stripped)
        exact_phrase = 1.0 if (query_stripped in content_normalized
                              or query_collapsed in content_collapsed) else 0.0

        # --- Signal 4: Title match (TF-IDF weighted) ---
        title_lower = result.title.lower()
        title_tokens = set(self._tokenize(title_lower))
        title_score = 0.0
        for t in query_terms:
            if t in title_tokens or t in title_lower:
                title_score += self._get_idf(t)
        title_match = title_score / tfidf_max if tfidf_max > 0 else 0.0

        # --- Signal 5: Position-aware scoring ---
        # Keywords appearing in the first 20% of document score higher
        position_score = 0.0
        doc_len = max(len(content_normalized), 1)
        head_cutoff = max(200, doc_len // 5)  # first 20% or 200 chars
        head_text = content_normalized[:head_cutoff]
        head_collapsed = re.sub(r'\s+', '', head_text)
        for t in query_terms:
            idf = self._get_idf(t)
            if t in head_text or t in head_collapsed:
                position_score += idf
        position_weighted = position_score / tfidf_max if tfidf_max > 0 else 0.0

        # --- Signal 6: Co-occurrence / proximity ---
        # Find the smallest window in the doc containing the most query terms
        cooccurrence = self._compute_cooccurrence(query_terms, content_normalized)

        # --- Signal 7: Original retrieval score (min-max normalized) ---
        original_score = norm_original_score

        # --- Signal 8: Focused paragraph density ---
        # How concentrated are query keywords within a single paragraph?
        # Uses only high-IDF terms (compound CJK + latin words) to avoid
        # saturation from common unigrams/bigrams.
        focused_density = self._compute_focused_density(
            query, content_normalized
        )

        return {
            "tfidf_weighted": tfidf_weighted,
            "entity_coverage": entity_coverage,
            "exact_phrase": exact_phrase,
            "title_match": title_match,
            "position_weighted": position_weighted,
            "cooccurrence": cooccurrence,
            "original_score": original_score,
            "is_entity": result.page_type == "entity",
            "focused_density": focused_density,
        }

    @staticmethod
    def _fuse_signals(signals: dict[str, float]) -> float:
        """Weighted fusion of all signals.

        Weights tuned via grid search on 48 QA benchmark.
        The 7 base signal weights are FROZEN. New signals are added
        as small additive bonuses that don't redistribute weight.
        """
        # --- Base score: 7 frozen signals (sum of weights = 0.90) ---
        base = (
            signals["tfidf_weighted"] * 0.25
            + signals["entity_coverage"] * 0.10
            + signals["exact_phrase"] * 0.10
            + signals["title_match"] * 0.15
            + signals["position_weighted"] * 0.15
            + signals["cooccurrence"] * 0.10
            + signals["original_score"] * 0.05
        )
        # Apply source-type penalty for entity pages
        # Entity pages are aggregations that match many queries broadly;
        # penalize unless they have strong specific matches
        if signals.get("is_entity", False):
            has_strong_match = (
                signals["exact_phrase"] > 0.5
                or signals["entity_coverage"] > 0.7
            )
            if not has_strong_match:
                base *= 0.5

        # --- Additive bonus: focused paragraph density ---
        # Small bonus (max ~0.03) for docs with query keywords
        # concentrated in a single paragraph. Helps break ties
        # without disturbing base ranking.
        focused = signals.get("focused_density", 0.0)
        base += focused * 0.08

        return base

    def _compute_focused_density(
        self, query: str, content: str,
    ) -> float:
        """Paragraph-level keyword density using high-IDF terms only.

        Splits content into paragraphs and scores each by how many
        high-value query terms (compound CJK words + latin tokens)
        appear within it. Returns the best paragraph's coverage.

        This avoids the saturation problem of document-level co-occurrence
        where common CJK unigrams match everywhere.
        """
        # Extract only high-value terms: original CJK compounds (2+ chars)
        # and latin words, NOT the exploded unigrams/bigrams
        raw = re.findall(r'[\u4e00-\u9fff]+|[a-zA-Z0-9_.-]+', query.lower())
        # Filter to terms with meaningful IDF (skip very common single chars)
        high_value_terms = set()
        for t in raw:
            if len(t) >= 2:  # CJK compounds or latin words 2+ chars
                high_value_terms.add(t)
            elif len(t) == 1 and self._get_idf(t) > 3.0:  # rare single chars
                high_value_terms.add(t)

        if not high_value_terms:
            return 0.0

        # Split into paragraphs (blank lines, headers, bullet groups)
        paragraphs = re.split(r'\n\s*\n|\n(?=#{1,3}\s)|\n(?=[-*]\s)', content)

        best_coverage = 0.0
        for para in paragraphs:
            if len(para.strip()) < 20:
                continue
            para_lower = para.lower()
            para_collapsed = re.sub(r'\s+', '', para_lower)
            hits = sum(
                1 for t in high_value_terms
                if t in para_lower or t in para_collapsed
            )
            coverage = hits / len(high_value_terms)
            if coverage > best_coverage:
                best_coverage = coverage

        return best_coverage

    @staticmethod
    def _compute_cooccurrence(query_terms: set[str], content: str) -> float:
        """Score based on how close query terms appear to each other in content.

        Higher score = more query terms found within a small text window.
        """
        if not query_terms or not content:
            return 0.0

        # Find positions of all query term occurrences
        positions: list[tuple[int, str]] = []
        content_lower = content.lower()
        collapsed = re.sub(r'\s+', '', content_lower)

        for term in query_terms:
            idx = 0
            while True:
                pos = content_lower.find(term, idx)
                if pos < 0:
                    break
                positions.append((pos, term))
                idx = pos + 1

        if not positions:
            return 0.0

        unique_terms_found = set(t for _, t in positions)
        coverage = len(unique_terms_found) / len(query_terms)

        if len(positions) < 2:
            return coverage * 0.5

        # Sort by position and find smallest window containing most distinct terms
        positions.sort()
        best_density = 0.0

        for i in range(len(positions)):
            seen = {positions[i][1]}
            for j in range(i + 1, len(positions)):
                seen.add(positions[j][1])
                window_size = positions[j][0] - positions[i][0] + 1
                # density = distinct terms / window size (normalized)
                density = len(seen) / (1 + window_size / 200.0)
                if density > best_density:
                    best_density = density
                if window_size > 500:  # don't look beyond 500 chars
                    break

        return min(1.0, coverage * 0.5 + best_density * 0.5)

    @staticmethod
    def _normalize_terms(text: str) -> set[str]:
        """Extract normalized query terms."""
        raw = re.findall(r'[\u4e00-\u9fff]+|[a-zA-Z0-9_.-]+', text.lower())
        terms = set(raw)
        for t in raw:
            if all('\u4e00' <= c <= '\u9fff' for c in t) and len(t) >= 2:
                for c in t:
                    terms.add(c)
                # Also add bigrams for CJK compound terms
                for i in range(len(t) - 1):
                    terms.add(t[i:i+2])
        return terms

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """Tokenize for IDF computation: CJK unigrams + bigrams + latin words."""
        text_lower = text.lower()
        raw = re.findall(r"[\u4e00-\u9fff]|[a-zA-Z0-9_.-]+", text_lower)
        tokens = list(raw)
        cjk_chars = [c for c in raw if len(c) == 1 and '\u4e00' <= c <= '\u9fff']
        for i in range(len(cjk_chars) - 1):
            tokens.append(cjk_chars[i] + cjk_chars[i + 1])
        return tokens

    @staticmethod
    def _extract_entities(query: str) -> list[str]:
        """Extract key entities/identifiers from query."""
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

        # Hex strings
        for m in re.finditer(r'\b[a-f0-9]{8,}\b', query, re.IGNORECASE):
            e = m.group()
            if e.lower() not in seen:
                seen.add(e.lower())
                entities.append(e)

        # Numbers with units
        for m in re.finditer(r'\d+(?:\.\d+)?(?:%|px|GB|MB)', query):
            e = m.group()
            if e not in seen:
                seen.add(e)
                entities.append(e)

        # Key CJK compound terms (3+ chars)
        for m in re.finditer(r'[\u4e00-\u9fff]{3,}', query):
            e = m.group()
            if e not in seen:
                seen.add(e)
                entities.append(e)

        return entities

    @staticmethod
    def _strip_markdown(text: str) -> str:
        """Remove markdown formatting for cleaner matching."""
        from .cleaner import MemoryCleaner
        return MemoryCleaner.clean(text)
