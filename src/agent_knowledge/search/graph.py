"""Knowledge Graph traversal: find related docs via entity links.

Supports 2-hop traversal with distance-decayed scoring and
co-occurrence weighted edges.

Usage example::

    from agent_knowledge.core.vault import Vault
    from agent_knowledge.search.graph import GraphIndex

    vault = Vault("/path/to/vault")
    graph = GraphIndex(vault)
    graph.build()  # builds graph from entity backlinks

    results = graph.search("MCP protocol", top_k=5)
    for r in results:
        print(f"{r.title} [score: {r.score}]")
        # Results found via 1-hop (direct entity mention) or
        # 2-hop (co-occurring entity traversal) with decayed scores.
"""

from __future__ import annotations

from collections import Counter

from ..core.vault import Vault
from .engine import SearchResult


class GraphIndex:
    """In-memory graph index built from entity backlinks.

    Supports 2-hop traversal:
      query → entity A (1-hop) → source S (1-hop score)
      query → entity A → source S → entity B → source T (2-hop score)
    """

    def __init__(self, vault: Vault):
        self.vault = vault
        # entity_name_lower -> entity_id
        self.name_to_id: dict[str, str] = {}
        # entity_id -> list of source_ids (backlinks)
        self.backlinks: dict[str, list[str]] = {}
        # source_id -> set of entity_ids that mention it
        self.source_entities: dict[str, set[str]] = {}
        # (entity_id, source_id) -> co-occurrence count (edge strength)
        self.edge_weight: Counter = Counter()
        # source_id -> {title, type}
        self.source_info: dict[str, dict] = {}
        self._built = False

    def build(self) -> int:
        """Build graph from vault entities. Returns edge count."""
        self.name_to_id = {}
        self.backlinks = {}
        self.source_entities = {}
        self.edge_weight = Counter()
        self.source_info = {}
        edge_count = 0

        for eid in self.vault.list_entities():
            entity = self.vault.load_entity(eid)
            if not entity:
                continue
            self.name_to_id[entity.name.lower()] = entity.id
            for alias in entity.aliases:
                self.name_to_id[alias.lower()] = entity.id
            self.backlinks[entity.id] = entity.backlinks
            for sid in entity.backlinks:
                self.source_entities.setdefault(sid, set()).add(entity.id)
                self.edge_weight[(entity.id, sid)] += 1
            edge_count += len(entity.backlinks)

        for sid in self.vault.list_sources():
            source = self.vault.load_source(sid)
            if source:
                self.source_info[sid] = {"title": source.title, "type": "source"}

        self._built = True
        return edge_count

    def search(self, query: str, top_k: int = 10) -> list[SearchResult]:
        """Find sources via 1-hop and 2-hop entity traversal.

        Scoring:
          1-hop: base_weight * edge_strength  (distance decay = 1.0)
          2-hop: base_weight * edge_strength * 0.5  (distance decay)
        Edge strength = co-occurrence count normalized.
        """
        if not self._built:
            self.build()

        query_lower = query.lower()
        scored: dict[str, float] = {}  # source_id -> accumulated score
        hop_info: dict[str, str] = {}  # source_id -> snippet info

        # Find entities mentioned in query (1-hop roots)
        matched_eids: list[tuple[str, str]] = []  # (entity_name, entity_id)
        for name, eid in self.name_to_id.items():
            if name in query_lower:
                matched_eids.append((name, eid))

        if not matched_eids:
            return []

        # Max edge weight for normalization
        max_ew = max(self.edge_weight.values()) if self.edge_weight else 1

        # 1-hop: entity -> sources
        hop1_sources: set[str] = set()
        for name, eid in matched_eids:
            for sid in self.backlinks.get(eid, []):
                if sid not in self.source_info:
                    continue
                ew = self.edge_weight.get((eid, sid), 1) / max_ew
                hop_score = 1.0 * ew  # decay=1.0 for 1-hop
                scored[sid] = scored.get(sid, 0.0) + hop_score
                hop_info[sid] = f"[1-hop via: {name}]"
                hop1_sources.add(sid)

        # 2-hop: source -> co-occurring entities -> their other sources
        for sid in hop1_sources:
            co_entities = self.source_entities.get(sid, set())
            for co_eid in co_entities:
                # Skip if this entity was already a direct query match
                if co_eid in {eid for _, eid in matched_eids}:
                    continue
                for sid2 in self.backlinks.get(co_eid, []):
                    if sid2 in hop1_sources or sid2 not in self.source_info:
                        continue
                    ew = self.edge_weight.get((co_eid, sid2), 1) / max_ew
                    hop_score = 0.5 * ew  # decay=0.5 for 2-hop
                    scored[sid2] = scored.get(sid2, 0.0) + hop_score
                    if sid2 not in hop_info:
                        hop_info[sid2] = f"[2-hop via: {sid}]"

        # Sort and return top_k
        sorted_sources = sorted(scored.items(), key=lambda x: x[1], reverse=True)[:top_k]

        results = []
        for sid, score in sorted_sources:
            info = self.source_info[sid]
            results.append(SearchResult(
                id=sid,
                page_type=info["type"],
                title=info["title"],
                snippet=hop_info.get(sid, "[via graph]"),
                score=round(score, 4),
            ))

        return results
