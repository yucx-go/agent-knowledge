"""MemoryConsolidator: deduplicate and merge similar facts at index time.

When the same fact appears multiple times across sources (e.g., decision
repeated in multiple daily logs), keep the newest version as authoritative
and downweight older duplicates.

Usage example::

    from agent_knowledge.search.consolidator import MemoryConsolidator

    facts = [
        {"text": "MCP is the standard protocol", "source_id": "s1", "timestamp": 100},
        {"text": "MCP is the standard protocol for agents", "source_id": "s2", "timestamp": 200},
    ]

    consolidated = MemoryConsolidator.consolidate(facts)
    # consolidated[0].weight == 1.0  (newest, authoritative)
    # consolidated[1].weight == 0.3  (older duplicate, downweighted)

    # Jaccard similarity between two texts:
    sim = MemoryConsolidator._jaccard_similarity("hello world", "hello there")
    # → 0.333...
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ConsolidatedFact:
    """A deduplicated, authoritative fact with provenance."""
    fact: str
    entities: list[str] = field(default_factory=list)
    source: str = ""  # source_id of the authoritative version
    timestamp: float = field(default_factory=time.time)
    supersedes: Optional[str] = None  # source_id of the older version
    weight: float = 1.0  # indexing weight (superseded facts get < 1.0)


class MemoryConsolidator:
    """Detect and consolidate duplicate/overlapping facts across sources."""

    # Minimum Jaccard similarity to consider two facts as duplicates
    SIMILARITY_THRESHOLD = 0.6

    @classmethod
    def consolidate(cls, facts: list[dict]) -> list[ConsolidatedFact]:
        """Consolidate a list of facts, deduplicating similar ones.

        Input: list of {text, source_id, timestamp, entities}
        Output: list of ConsolidatedFact with supersession links
        """
        if not facts:
            return []

        # Sort by timestamp descending (newest first)
        sorted_facts = sorted(facts, key=lambda f: f.get("timestamp", 0), reverse=True)

        consolidated: list[ConsolidatedFact] = []
        superseded_indices: set[int] = set()

        for i, fact in enumerate(sorted_facts):
            if i in superseded_indices:
                continue

            # This is the newest version — check if any later (older) facts are duplicates
            for j in range(i + 1, len(sorted_facts)):
                if j in superseded_indices:
                    continue

                sim = cls._jaccard_similarity(
                    fact.get("text", ""),
                    sorted_facts[j].get("text", "")
                )
                if sim >= cls.SIMILARITY_THRESHOLD:
                    superseded_indices.add(j)

            # Create consolidated fact
            cf = ConsolidatedFact(
                fact=fact.get("text", ""),
                entities=fact.get("entities", []),
                source=fact.get("source_id", ""),
                timestamp=fact.get("timestamp", 0),
                supersedes=None,
                weight=1.0,
            )
            consolidated.append(cf)

        # Also add superseded facts with reduced weight (for recall)
        for idx in superseded_indices:
            fact = sorted_facts[idx]
            cf = ConsolidatedFact(
                fact=fact.get("text", ""),
                entities=fact.get("entities", []),
                source=fact.get("source_id", ""),
                timestamp=fact.get("timestamp", 0),
                supersedes="superseded",
                weight=0.3,  # heavily downweighted but still searchable
            )
            consolidated.append(cf)

        return consolidated

    @classmethod
    def _jaccard_similarity(cls, text_a: str, text_b: str) -> float:
        """Compute Jaccard similarity between two texts using token sets."""
        tokens_a = cls._tokenize(text_a)
        tokens_b = cls._tokenize(text_b)

        if not tokens_a or not tokens_b:
            return 0.0

        intersection = tokens_a & tokens_b
        union = tokens_a | tokens_b

        return len(intersection) / len(union) if union else 0.0

    @staticmethod
    def _tokenize(text: str) -> set[str]:
        """Tokenize text into a set of normalized terms."""
        text_lower = text.lower()
        # Extract CJK chars + latin words
        tokens = set(re.findall(r'[\u4e00-\u9fff]|[a-zA-Z0-9_]+', text_lower))
        return tokens
