"""Dream Cycle: 3-phase memory consolidation.

Light  → scan new sources, extract candidate claims
REM    → theme clustering, cross-source linking
Deep   → 6-dim scoring, promote high-value knowledge
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

from ..core.models import Claim, ClaimDurability, ClaimStatus, Entity, ForesightStatus, Source
from ..core.vault import Vault


@dataclass
class DreamCandidate:
    """A candidate memory surfaced during Light phase."""

    claim: Claim
    source_id: str
    score: float = 0.0
    promoted: bool = False
    # Populated by REM phase
    _cluster_id: int = -1
    _cluster_size: int = 1
    _cross_cluster_entities: set = field(default_factory=set)


@dataclass
class DreamReport:
    """Output of a complete dream cycle."""

    timestamp: float = field(default_factory=time.time)
    light_candidates: int = 0
    rem_clusters: int = 0
    deep_promoted: int = 0
    deep_discarded: int = 0
    duration_seconds: float = 0.0
    details: list[dict] = field(default_factory=list)


# ── Scoring Weights (6 dimensions) ──

SCORE_WEIGHTS = {
    "frequency": 0.24,       # how often this topic appears
    "relevance": 0.30,       # how relevant to core domains
    "query_diversity": 0.15, # referenced from diverse contexts
    "recency": 0.15,         # how recent
    "consolidation": 0.10,   # already partially consolidated
    "concept_richness": 0.06,  # connects to many entities/concepts
}

# Promotion threshold: only candidates scoring above this get promoted
PROMOTION_THRESHOLD = 0.45


class DreamCycle:
    """Execute Light → REM → Deep consolidation."""

    def __init__(self, vault: Vault):
        self.vault = vault

    def run(self, since_hours: float = 24.0) -> DreamReport:
        """Run a complete dream cycle.

        Args:
            since_hours: only process sources ingested within this window
        """
        start = time.time()
        report = DreamReport()

        # Phase 1: Light
        candidates = self._light_phase(since_hours)
        report.light_candidates = len(candidates)

        # Phase 2: REM
        clusters = self._rem_phase(candidates)
        report.rem_clusters = len(clusters)

        # Phase 3: Deep
        promoted, discarded = self._deep_phase(candidates)
        report.deep_promoted = len(promoted)
        report.deep_discarded = len(discarded)

        report.duration_seconds = round(time.time() - start, 2)
        report.details = [
            {
                "claim": c.claim.text[:100],
                "score": round(c.score, 3),
                "promoted": c.promoted,
            }
            for c in candidates
        ]

        return report

    # ── Phase 1: Light ──

    def _light_phase(self, since_hours: float) -> list[DreamCandidate]:
        """Scan recent sources, extract candidate claims.

        Also checks all entities' foresight items for expiry.
        """
        cutoff = time.time() - (since_hours * 3600)
        candidates = []

        for sid in self.vault.list_sources():
            source = self.vault.load_source(sid)
            if not source:
                continue
            if source.ingested_at < cutoff:
                continue

            # Re-extract claims from recent sources
            from ..core.compiler import Compiler
            compiler = Compiler(self.vault)
            claims = compiler._extract_claims(source.content, source.id)

            for claim in claims:
                candidates.append(DreamCandidate(
                    claim=claim,
                    source_id=source.id,
                ))

        # Check foresight expiry across all entities
        self._expire_foresight()

        return candidates

    def _expire_foresight(self) -> None:
        """Mark expired foresight items across all entities."""
        now = time.time()
        for eid in self.vault.list_entities():
            entity = self.vault.load_entity(eid)
            if not entity:
                continue
            changed = False
            for fi in entity.compiled_truth.foresight:
                if fi.status == ForesightStatus.PENDING and fi.valid_until > 0 and now > fi.valid_until:
                    fi.status = ForesightStatus.EXPIRED
                    changed = True
            if changed:
                self.vault.save_entity(entity)

    # ── Phase 2: REM ──

    def _rem_phase(self, candidates: list[DreamCandidate]) -> list[list[DreamCandidate]]:
        """Cluster candidates by Jaccard similarity on token sets.

        Returns clusters and annotates each candidate with:
        - _cluster_id: which cluster it belongs to
        - _cluster_size: how many items in that cluster (frequency signal)
        - _cross_cluster_entities: entities appearing across multiple clusters
        """
        if not candidates:
            return []

        from ..search.consolidator import MemoryConsolidator

        # Build clusters via greedy Jaccard similarity
        clusters: list[list[DreamCandidate]] = []
        assigned: list[int] = [-1] * len(candidates)  # cluster index per candidate

        for i, c in enumerate(candidates):
            if assigned[i] >= 0:
                continue
            # Start new cluster
            cluster_idx = len(clusters)
            cluster = [c]
            assigned[i] = cluster_idx

            for j in range(i + 1, len(candidates)):
                if assigned[j] >= 0:
                    continue
                sim = MemoryConsolidator._jaccard_similarity(c.claim.text, candidates[j].claim.text)
                if sim >= 0.3:  # lower threshold for clustering
                    cluster.append(candidates[j])
                    assigned[j] = cluster_idx

            clusters.append(cluster)

        # Annotate candidates with cluster metadata
        for cluster_idx, cluster in enumerate(clusters):
            for dc in cluster:
                dc._cluster_id = cluster_idx
                dc._cluster_size = len(cluster)

        # Find entities that appear across multiple clusters
        entity_clusters: dict[str, set[int]] = {}  # entity_id -> set of cluster ids
        for i, c in enumerate(candidates):
            if c.claim.entity_id:
                entity_clusters.setdefault(c.claim.entity_id, set()).add(assigned[i])

        cross_cluster_entities = {eid for eid, cids in entity_clusters.items() if len(cids) > 1}

        # Annotate cross-cluster entities
        for c in candidates:
            c._cross_cluster_entities = cross_cluster_entities

        return clusters

    # ── Phase 3: Deep ──

    def _deep_phase(self, candidates: list[DreamCandidate]) -> tuple[list[DreamCandidate], list[DreamCandidate]]:
        """Score each candidate on 6 dimensions, promote above threshold."""
        if not candidates:
            return [], []

        # Score each candidate
        for c in candidates:
            c.score = self._score_candidate(c)

        # Sort by score descending
        candidates.sort(key=lambda x: x.score, reverse=True)

        promoted = []
        discarded = []

        for c in candidates:
            if c.score >= PROMOTION_THRESHOLD:
                c.promoted = True
                self._promote(c)
                promoted.append(c)
            else:
                discarded.append(c)

        return promoted, discarded

    def _score_candidate(self, c: DreamCandidate) -> float:
        """Score a candidate on 6 dimensions with REM cluster boosts."""
        scores = {}

        # 1. Frequency: evidence count + cluster size boost from REM
        evidence_count = len(c.claim.evidence)
        base_freq = min(1.0, evidence_count / 3.0)
        # Cluster size boost: items in larger clusters are more important
        cluster_boost = min(0.3, (c._cluster_size - 1) * 0.1)
        scores["frequency"] = min(1.0, base_freq + cluster_boost)

        # 2. Relevance: decisions and learnings score higher
        if "decision" in c.claim.tags:
            scores["relevance"] = 0.9
        elif "learning" in c.claim.tags:
            scores["relevance"] = 0.8
        elif "action" in c.claim.tags:
            scores["relevance"] = 0.6
        elif "todo" in c.claim.tags:
            scores["relevance"] = 0.3
        else:
            scores["relevance"] = 0.4

        # 3. Query diversity: how many different entities/sources reference this
        source_count = len(set(e.source_id for e in c.claim.evidence))
        scores["query_diversity"] = min(1.0, source_count / 2.0)

        # 4. Recency: exponential decay, half-life varies by durability
        age_days = (time.time() - c.claim.created_at) / 86400
        if c.claim.durability == ClaimDurability.TEMPORARY:
            half_life = 3.0
        elif c.claim.durability == ClaimDurability.STABLE:
            half_life = 30.0
        else:
            half_life = 7.0
        scores["recency"] = 2 ** (-age_days / half_life)

        # 5. Consolidation: already linked to an entity = partially consolidated
        scores["consolidation"] = 0.7 if c.claim.entity_id else 0.2

        # 6. Concept richness: text length + cross-cluster entity boost from REM
        text_len = len(c.claim.text)
        base_richness = min(1.0, text_len / 100.0)
        # Cross-cluster boost: entities spanning multiple clusters are richer concepts
        if c.claim.entity_id and c.claim.entity_id in c._cross_cluster_entities:
            base_richness = min(1.0, base_richness + 0.3)
        scores["concept_richness"] = base_richness

        # Weighted sum
        total = sum(scores[k] * SCORE_WEIGHTS[k] for k in SCORE_WEIGHTS)
        return total

    def _promote(self, c: DreamCandidate) -> None:
        """Promote a candidate: update its entity's compiled truth."""
        if not c.claim.entity_id:
            return

        entity = self.vault.load_entity(c.claim.entity_id)
        if not entity:
            return

        # Add claim to entity's compiled truth if not already present
        existing_ids = {cl.id for cl in entity.compiled_truth.claims}
        if c.claim.id not in existing_ids:
            entity.compiled_truth.claims.append(c.claim)

            # Recompile summary using structured compiler
            from ..core.compiler import Compiler
            entity.compiled_truth.summary = Compiler._compile_summary(
                entity.compiled_truth.active_claims
            )
            entity.compiled_truth.last_compiled_at = time.time()
            entity.updated_at = time.time()
            self.vault.save_entity(entity)
