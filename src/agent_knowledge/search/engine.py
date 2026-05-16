"""Search engine: BM25 + optional semantic + RRF fusion."""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from ..core.vault import Vault


@dataclass
class SearchResult:
    """A single search result with score and provenance."""

    id: str
    page_type: str  # source / entity / concept
    title: str
    snippet: str
    score: float
    path: Optional[str] = None


class ExactIndex:
    """Exact substring match index for IDs, tokens, API params."""

    def __init__(self):
        self.docs: dict[str, dict] = {}  # doc_id -> {title, text, type}

    def add(self, doc_id: str, title: str, text: str, page_type: str) -> None:
        self.docs[doc_id] = {"title": title, "text": text, "type": page_type}

    def search(self, query: str, top_k: int = 10) -> list[SearchResult]:
        """Find docs containing exact substrings from the query.

        Extracts notable terms (technical names, IDs, numbers, API params)
        and searches for each as a substring. This catches:
        - collapsible_panel, lark_md, tblXXX (API params)
        - 3000, 74.5% (specific numbers)
        - \\| (special characters)
        """
        # Extract searchable terms: words with _, digits, or 3+ chars
        terms = self._extract_search_terms(query)
        if not terms:
            return []

        scored: dict[str, float] = {}
        match_counts: dict[str, int] = {}  # doc_id -> number of distinct terms matched
        best_match: dict[str, str] = {}  # doc_id -> matched term

        for term in terms:
            term_lower = term.lower()
            for doc_id, doc in self.docs.items():
                text_lower = doc["text"].lower()
                count = text_lower.count(term_lower)
                if count > 0:
                    pos = text_lower.find(term_lower)
                    pos_score = 1.0 / (1.0 + pos / 1000.0)
                    # Longer exact matches get higher scores
                    term_score = (count * 2.0 + pos_score) * (1.0 + len(term) / 20.0)
                    # Accumulate scores from multiple matching terms
                    scored[doc_id] = scored.get(doc_id, 0.0) + term_score
                    match_counts[doc_id] = match_counts.get(doc_id, 0) + 1
                    if doc_id not in best_match:
                        best_match[doc_id] = term

        # Bonus for matching multiple distinct terms (better coverage)
        for doc_id in scored:
            mc = match_counts.get(doc_id, 1)
            if mc > 1:
                scored[doc_id] *= (1.0 + 0.3 * (mc - 1))

        scored_list = sorted(scored.items(), key=lambda x: x[1], reverse=True)

        results = []
        for doc_id, score in scored_list[:top_k]:
            doc = self.docs[doc_id]
            matched = best_match.get(doc_id, query)
            text = doc["text"]
            idx = text.lower().find(matched.lower())
            if idx >= 0:
                start = max(0, idx - 80)
                end = min(len(text), idx + len(matched) + 120)
                snippet = text[start:end].replace("\n", " ").strip()
                if start > 0:
                    snippet = "\u2026" + snippet
                if end < len(text):
                    snippet += "\u2026"
            else:
                snippet = text[:200].replace("\n", " ").strip()

            results.append(SearchResult(
                id=doc_id,
                page_type=doc["type"],
                title=doc["title"],
                snippet=snippet,
                score=round(score, 4),
            ))
        return results

    @staticmethod
    def _extract_search_terms(query: str) -> list[str]:
        """Extract notable terms for exact substring search.

        Prioritizes:
        - Technical identifiers: snake_case, camelCase, with dots/dashes
        - IDs and tokens: hex strings, base64-like, tblXXX, ou_XXX
        - Numbers with units: 3000, 74.5%, 8GB
        - Special characters when quoted or escaped: \\|
        """
        import re
        terms = []

        # Snake_case / technical identifiers (e.g., collapsible_panel, lark_md)
        for m in re.finditer(r'[a-zA-Z][a-zA-Z0-9]*(?:_[a-zA-Z0-9]+)+', query):
            terms.append(m.group())

        # Dot-separated identifiers (e.g., end_index)
        for m in re.finditer(r'[a-zA-Z][a-zA-Z0-9]*\.[a-zA-Z][a-zA-Z0-9]*', query):
            terms.append(m.group())

        # IDs / tokens (hex, base64-ish, tbl/ou_ prefixed)
        for m in re.finditer(r'(?:tbl|ou_|oc_|om_|img_|file_)[a-zA-Z0-9]+', query):
            terms.append(m.group())
        for m in re.finditer(r'[a-f0-9]{8,}', query, re.IGNORECASE):
            terms.append(m.group())

        # Numbers with units or percentages
        for m in re.finditer(r'\d+(?:\.\d+)?(?:%|px|GB|MB|KB|\u5143|\u4e07)', query):
            terms.append(m.group())
        # Standalone significant numbers
        for m in re.finditer(r'\b\d{3,}\b', query):
            terms.append(m.group())

        # Escaped special chars
        for m in re.finditer(r'\\[|{}\[\]]', query):
            terms.append(m.group())

        # CJK compound terms (2-4 chars that look like technical terms)
        for m in re.finditer(r'[\u4e00-\u9fff]{2,4}', query):
            t = m.group()
            # Only keep terms that look domain-specific
            if len(t) >= 3:
                terms.append(t)

        # Dedupe preserving order
        seen = set()
        unique = []
        for t in terms:
            if t.lower() not in seen and len(t) >= 2:
                seen.add(t.lower())
                unique.append(t)

        return unique


class BM25Index:
    """Simple in-memory BM25 index over vault pages."""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.docs: dict[str, dict] = {}  # doc_id -> {title, text, type, tokens}
        self.df: Counter = Counter()  # document frequency
        self.avg_dl: float = 0.0

    def add(self, doc_id: str, title: str, text: str, page_type: str) -> None:
        tokens = self._tokenize(text)
        self.docs[doc_id] = {
            "title": title,
            "text": text,
            "type": page_type,
            "tokens": tokens,
            "tf": Counter(tokens),
            "dl": len(tokens),
        }
        for t in set(tokens):
            self.df[t] += 1
        # Recalc avg doc length
        total = sum(d["dl"] for d in self.docs.values())
        self.avg_dl = total / len(self.docs) if self.docs else 0.0

    def search(self, query: str, top_k: int = 10) -> list[SearchResult]:
        query_tokens = self._tokenize(query)
        query_lower = query.lower()
        n = len(self.docs)
        if n == 0:
            return []

        scores: dict[str, float] = {}
        for doc_id, doc in self.docs.items():
            score = 0.0
            for qt in query_tokens:
                if qt not in doc["tf"]:
                    continue
                tf = doc["tf"][qt]
                df = self.df.get(qt, 0)
                idf = math.log((n - df + 0.5) / (df + 0.5) + 1.0)
                tf_norm = (tf * (self.k1 + 1)) / (
                    tf + self.k1 * (1 - self.b + self.b * doc["dl"] / self.avg_dl)
                )
                score += idf * tf_norm

            # Exact-match boost: if query appears as substring in text
            if score > 0:
                text_lower = doc["text"].lower()
                # Boost for exact phrase match
                if query_lower in text_lower:
                    score *= 1.5
                # Boost for exact token/ID match (useful for tokens, hashes, IDs)
                if len(query_lower) >= 6 and query_lower in text_lower:
                    score *= 1.3

            if score > 0:
                scores[doc_id] = score

        sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        results = []
        for doc_id, score in sorted_docs:
            doc = self.docs[doc_id]
            snippet = self._make_snippet(doc["text"], query_tokens)
            results.append(SearchResult(
                id=doc_id,
                page_type=doc["type"],
                title=doc["title"],
                snippet=snippet,
                score=round(score, 4),
            ))
        return results

    def _tokenize(self, text: str) -> list[str]:
        """Tokenization: CJK unigrams + bigrams + Latin words + CJK-latin compounds.
        Bigrams help match compound Chinese terms.
        CJK-latin compounds (e.g., '不能用tag') improve mixed-language matching.
        """
        text_lower = text.lower()
        raw = re.findall(r"[\u4e00-\u9fff]|[a-zA-Z0-9_.-]+", text_lower)
        tokens = list(raw)
        # Add CJK bigrams for better compound term matching
        cjk_chars = [c for c in raw if len(c) == 1 and '\u4e00' <= c <= '\u9fff']
        for i in range(len(cjk_chars) - 1):
            tokens.append(cjk_chars[i] + cjk_chars[i + 1])
        # Add CJK-latin boundary compounds (important for mixed-language content)
        # e.g., adjacent CJK + latin or latin + CJK tokens
        for i in range(len(raw) - 1):
            a, b = raw[i], raw[i + 1]
            a_cjk = len(a) == 1 and '\u4e00' <= a <= '\u9fff'
            b_cjk = len(b) == 1 and '\u4e00' <= b <= '\u9fff'
            a_lat = not a_cjk and len(a) > 0
            b_lat = not b_cjk and len(b) > 0
            if (a_cjk and b_lat) or (a_lat and b_cjk):
                tokens.append(a + b)
        return tokens

    def _make_snippet(self, text: str, query_tokens: list[str], max_len: int = 200) -> str:
        """Extract a relevant snippet around the first query match."""
        text_lower = text.lower()
        best_pos = len(text)
        for qt in query_tokens:
            pos = text_lower.find(qt)
            if 0 <= pos < best_pos:
                best_pos = pos

        start = max(0, best_pos - 50)
        end = min(len(text), start + max_len)
        snippet = text[start:end].replace("\n", " ").strip()
        if start > 0:
            snippet = "…" + snippet
        if end < len(text):
            snippet = snippet + "…"
        return snippet


def rrf_fuse(
    result_lists: list[list[SearchResult]],
    weights: Optional[list[float]] = None,
    k: int = 60,
    top_k: int = 10,
) -> list[SearchResult]:
    """Weighted Reciprocal Rank Fusion across multiple result lists.

    RRF score = sum(weight_i / (k + rank_i)) for each list where doc appears.
    k=60 is the standard constant from the original RRF paper.
    weights default to [1.0, 1.0, ...] (equal weight).
    """
    if weights is None:
        weights = [1.0] * len(result_lists)

    scores: dict[str, float] = {}
    doc_map: dict[str, SearchResult] = {}

    for w, results in zip(weights, result_lists):
        for rank, r in enumerate(results, 1):
            rrf_score = w / (k + rank)
            scores[r.id] = scores.get(r.id, 0.0) + rrf_score
            if r.id not in doc_map:
                doc_map[r.id] = r

    sorted_ids = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

    results = []
    for doc_id, score in sorted_ids:
        r = doc_map[doc_id]
        results.append(SearchResult(
            id=r.id,
            page_type=r.page_type,
            title=r.title,
            snippet=r.snippet,
            score=round(score, 6),
        ))
    return results


class SearchEngine:
    """Unified search: Exact + BM25 + Vector + Graph, RRF fusion + reranker.

    Fusion strategy (weighted RRF):
    1. Collect results from all engines: Exact, BM25, Vector, Graph
    2. Fuse via weighted Reciprocal Rank Fusion
    3. Rerank top candidates with multi-signal scorer
    """

    # Separator used to mark "fact-level child docs" in the index.
    # Doc IDs of the form `{parent_id}#fact_{n}` are mapped back to their
    # parent on result return so callers see only source-level IDs.
    _FACT_ID_SEP = "#fact_"

    def __init__(self, vault: Vault, embedder=None, enable_graph: bool = True,
                 enable_reranker: bool = True, enable_fact_extraction: bool = False):
        self.vault = vault
        self.bm25 = BM25Index()
        self.exact = ExactIndex()
        self.embedder = embedder
        self.vector_index = None
        self.graph_index = None
        self.reranker = None
        self.enable_graph = enable_graph
        self.enable_reranker = enable_reranker
        # Fact extraction surfaces "by the way" style incidental facts as
        # focused child docs. Default off because on retrieval benchmarks
        # dominated by semantic-gap failures (e.g. LongMemEval-S) it adds
        # noise without signal — short fact docs spuriously match abstract
        # queries. Enable for production scenarios where users embed
        # personal facts inside unrelated discussions and BM25-only retrieval
        # struggles with keyword burial. See benchmark/FAILURE_ANALYSIS.md.
        self.enable_fact_extraction = enable_fact_extraction
        self._indexed = False

    def build_index(self) -> int:
        """Build all indexes from vault contents.

        Applies MemoryCleaner to source content before indexing for better
        signal-to-noise ratio. Original content is preserved in vault.
        """
        from .cleaner import MemoryCleaner

        self.bm25 = BM25Index()
        self.exact = ExactIndex()
        count = 0

        # Optionally init vector index
        if self.embedder:
            from .vector import VectorIndex
            self.vector_index = VectorIndex(self.embedder)

        # Optionally init graph index
        if self.enable_graph:
            from .graph import GraphIndex
            self.graph_index = GraphIndex(self.vault)

        # Optionally init fact extractor — pulls "by the way" style
        # incidental personal mentions into focused child docs that BM25
        # can score precisely. See fact_extractor.py for rationale.
        extract_facts = None
        if self.enable_fact_extraction:
            from .fact_extractor import extract_incidental_facts as extract_facts

        # Index sources — clean content before indexing
        for sid in self.vault.list_sources():
            source = self.vault.load_source(sid)
            if source:
                cleaned, _entities = MemoryCleaner.clean_for_index(source.content)
                # Index cleaned text for BM25/exact, but also keep raw for exact
                # since raw may contain IDs/tokens that cleaning preserves anyway
                self.bm25.add(sid, source.title, cleaned, "source")
                self.exact.add(sid, source.title, source.content, "source")
                if self.vector_index:
                    self.vector_index.add(sid, source.title, cleaned, "source")
                count += 1

                # Index extracted incidental facts as child docs.
                # The doc IDs use {parent}#fact_{n} so search() can dedupe
                # back to the parent source on return.
                if extract_facts:
                    facts = extract_facts(source.content)
                    for i, fact in enumerate(facts):
                        fact_id = f"{sid}{self._FACT_ID_SEP}{i}"
                        self.bm25.add(fact_id, source.title, fact, "fact")
                        self.exact.add(fact_id, source.title, fact, "fact")

        # Index entities
        for eid in self.vault.list_entities():
            entity = self.vault.load_entity(eid)
            if entity:
                text_parts = [entity.name, entity.compiled_truth.summary]
                for claim in entity.compiled_truth.active_claims:
                    text_parts.append(claim.text)
                text = "\n".join(text_parts)
                self.bm25.add(eid, entity.name, text, "entity")
                self.exact.add(eid, entity.name, text, "entity")
                if self.vector_index:
                    self.vector_index.add(eid, entity.name, text, "entity")
                count += 1

        # Build vector embeddings
        if self.vector_index:
            self.vector_index.build()

        # Build graph
        if self.graph_index:
            self.graph_index.build()

        # Init reranker
        if self.enable_reranker:
            from .reranker import LightReranker
            self.reranker = LightReranker(self.vault)

        self._indexed = True
        return count

    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        """RRF-fused search with query rewriting and reranking.

        Uses QueryRewriter to generate keyword + entity variants,
        collects results from all engines, fuses via weighted RRF,
        then reranks top candidates.
        """
        if not self._indexed:
            self.build_index()

        from .query_rewriter import QueryRewriter
        rq = QueryRewriter.rewrite(query)

        # Collect candidate lists from each retrieval engine
        candidate_lists: list[list[SearchResult]] = []
        weights: list[float] = []

        # RRF weights tuned via grid search on 48 QA benchmark.
        # Exact substring match
        exact_results = self.exact.search(query, top_k=top_k * 3)
        if exact_results:
            candidate_lists.append(exact_results)
            weights.append(2.0)

        # Exact match with entity query variant
        if rq.entity and rq.entity != query:
            exact_entity = self.exact.search(rq.entity, top_k=top_k * 2)
            if exact_entity:
                candidate_lists.append(exact_entity)
                weights.append(1.0)

        # BM25 keyword search (original query)
        bm25_results = self.bm25.search(query, top_k=top_k * 3)
        if bm25_results:
            candidate_lists.append(bm25_results)
            weights.append(2.0)

        # BM25 with keyword-stripped query
        if rq.keyword and rq.keyword != query:
            bm25_keyword = self.bm25.search(rq.keyword, top_k=top_k * 2)
            if bm25_keyword:
                candidate_lists.append(bm25_keyword)
                weights.append(1.5)

        # Vector semantic search
        if self.vector_index:
            vec_results = self.vector_index.search(query, top_k=top_k * 3)
            if vec_results:
                candidate_lists.append(vec_results)
                weights.append(2.0)

        # Graph traversal
        if self.graph_index:
            graph_results = self.graph_index.search(query, top_k=top_k * 2)
            if graph_results:
                candidate_lists.append(graph_results)
                weights.append(0.5)

        # Fuse all results via weighted RRF — request extra candidates so
        # that fact-level dedup below still leaves enough source-level results.
        rerank_pool_size = max(top_k * 6, 30)
        fused = rrf_fuse(candidate_lists, weights=weights, k=60, top_k=rerank_pool_size)

        # Dedupe fact-level child docs back to their parent source so
        # callers see only source/entity-level IDs. Keep the highest-ranked
        # representative for each parent.
        fused = self._dedupe_by_parent(fused)

        # Rerank if available
        if self.reranker and len(fused) > 1:
            fused = self.reranker.rerank(query, fused, top_k=top_k)
        else:
            fused = fused[:top_k]

        return fused

    def _dedupe_by_parent(self, results: list[SearchResult]) -> list[SearchResult]:
        """Map fact-level doc IDs back to their parent source and dedupe.

        A doc ID of the form `{parent_id}#fact_{n}` indicates an incidental
        fact extracted from the parent source. After RRF, multiple results
        may share the same parent. Keep the highest-ranked (first-seen)
        and drop the rest. Non-fact IDs are unchanged.
        """
        seen: set[str] = set()
        out: list[SearchResult] = []
        for r in results:
            if self._FACT_ID_SEP in r.id:
                parent_id = r.id.split(self._FACT_ID_SEP, 1)[0]
                page_type = "source"
            else:
                parent_id = r.id
                page_type = r.page_type
            if parent_id in seen:
                continue
            seen.add(parent_id)
            out.append(SearchResult(
                id=parent_id,
                page_type=page_type,
                title=r.title,
                snippet=r.snippet,
                score=r.score,
            ))
        return out
