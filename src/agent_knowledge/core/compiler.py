"""Knowledge Compiler: extract claims, entities, build compiled truth.

This is the core differentiator — not just storing memories, but compiling
them into structured, verifiable knowledge.
"""

from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Optional

import yaml

from .models import (
    Claim,
    ClaimDurability,
    ClaimPolarity,
    ClaimStatus,
    CompiledTruth,
    Entity,
    Evidence,
    EvidenceWeight,
    ForesightItem,
    ForesightStatus,
    Source,
    TimelineEvent,
    _detect_polarity,
)
from .vault import Vault

# File name for user-defined entity patterns in vault root
_ENTITIES_CONFIG_FILE = ".ak-entities.yaml"


def _load_entity_config(vault_root: Path) -> Optional[dict]:
    """Load custom entity config from vault root. Returns None if absent."""
    p = vault_root / _ENTITIES_CONFIG_FILE
    if not p.exists():
        return None
    try:
        data = yaml.safe_load(p.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else None
    except Exception:
        return None


# Entity extraction lives in :mod:`.entity_extractor` — a reverse-filter
# pipeline (wide-net candidates → stopword filtering) that handles arbitrary
# domains without per-domain regex maintenance. The Compiler delegates to
# that module for extraction and only handles persistence and linking.


class Compiler:
    """Compiles raw sources into structured knowledge."""

    def __init__(self, vault: Vault):
        self.vault = vault
        self._entity_config: Optional[dict] = None
        self._entity_config_loaded = False

    def ingest(self, text: str, title: str = "", source_type: str = "text",
               path: Optional[str] = None, url: Optional[str] = None) -> Source:
        """Ingest raw text, extract claims and entities, save to vault.

        Returns the created Source object.
        """
        # 1. Create source
        source = Source(
            title=title or self._derive_title(text),
            content=text,
            source_type=source_type,
            path=path,
            url=url,
        )

        # 2. Extract claims
        claims = self._extract_claims(text, source.id)

        # 3. Extract entities
        entities = self._extract_entities(text, claims, source.id)

        # 4. Link
        source.claims_extracted = [c.id for c in claims]
        source.entities_extracted = [e.id for e in entities]

        # 5. Update entity compiled truths
        for entity in entities:
            self._update_compiled_truth(entity, claims, source)

        # 6. Consolidate claims per entity (deduplicate similar claims)
        for entity in entities:
            self._consolidate_entity_claims(entity)

        # 7. Save everything
        self.vault.save_source(source)
        for entity in entities:
            self.vault.save_entity(entity)

        return source

    def _derive_title(self, text: str) -> str:
        """Derive a title from the first meaningful line."""
        for line in text.strip().split("\n"):
            line = line.strip().lstrip("#").strip()
            if line and len(line) > 3:
                return line[:80]
        return "Untitled"

    # Section name mapping for weight/tag assignment
    _SECTION_MAP = {
        # Chinese
        "决策": (EvidenceWeight.PRIMARY, ["decision"]),
        "decisions": (EvidenceWeight.PRIMARY, ["decision"]),
        "decision": (EvidenceWeight.PRIMARY, ["decision"]),
        "学习": (EvidenceWeight.SECONDARY, ["learning"]),
        "learnings": (EvidenceWeight.SECONDARY, ["learning"]),
        "learning": (EvidenceWeight.SECONDARY, ["learning"]),
        "操作": (EvidenceWeight.SECONDARY, ["action"]),
        "operations": (EvidenceWeight.SECONDARY, ["action"]),
        "action": (EvidenceWeight.SECONDARY, ["action"]),
        "actions": (EvidenceWeight.SECONDARY, ["action"]),
        "待办": (EvidenceWeight.TERTIARY, ["todo"]),
        "todo": (EvidenceWeight.TERTIARY, ["todo"]),
        # English prose sections
        "summary": (EvidenceWeight.SECONDARY, ["summary"]),
        "conclusion": (EvidenceWeight.PRIMARY, ["decision"]),
        "conclusions": (EvidenceWeight.PRIMARY, ["decision"]),
        "findings": (EvidenceWeight.SECONDARY, ["learning"]),
        "risks": (EvidenceWeight.SECONDARY, ["risk"]),
        "risk": (EvidenceWeight.SECONDARY, ["risk"]),
        "recommendations": (EvidenceWeight.PRIMARY, ["decision"]),
    }

    # Sentence-level signal patterns for prose extraction
    _DECISION_SIGNALS = re.compile(
        r"(?:decided|chose|selected|adopted|switched|migrated|replaced|will use|going with"
        r"|决定|选择|采用|切换|迁移|替代)",
        re.IGNORECASE,
    )
    _FACT_SIGNALS = re.compile(
        r"(?:revenue|profit|margin|cost|budget|growth|decline|increased|decreased"
        r"|\$[\d,.]+|\d+(?:\.\d+)?%"
        r"|收入|利润|毛利|成本|预算|增长|下降|达成率)",
        re.IGNORECASE,
    )
    _RISK_SIGNALS = re.compile(
        r"(?:risk|warning|issue|bug|failure|problem|concern|critical"
        r"|风险|问题|异常|故障|失败|警告)",
        re.IGNORECASE,
    )

    # ── Durability detection ──
    _TEMPORARY_SIGNALS = re.compile(
        r"(?:最近|这周|目前|临时|本周|今天|当前|暂时|近期"
        r"|recently|currently|this week|right now|for now|temporary|interim)",
        re.IGNORECASE,
    )

    # ── Foresight signals ──
    _FORESIGHT_SIGNALS = re.compile(
        r"(?:要做|计划|下周|待|TODO|需要|打算|准备|即将|下个月"
        r"|plan to|going to|will|need to|todo|intend|schedule|next week|next month)",
        re.IGNORECASE,
    )
    _FORESIGHT_TIME_MAP: list[tuple[re.Pattern, int]] = [
        (re.compile(r"下周|next week", re.IGNORECASE), 7),
        (re.compile(r"下个月|next month", re.IGNORECASE), 30),
        (re.compile(r"明天|tomorrow", re.IGNORECASE), 1),
        (re.compile(r"今天|today", re.IGNORECASE), 1),
    ]
    _FORESIGHT_DEFAULT_DAYS = 14

    def _extract_claims(self, text: str, source_id: str) -> list[Claim]:
        """Extract structured claims from text.

        Strategy:
        1. Markdown list items (- ...) under known sections → structured claims
        2. Prose sentences with decision/fact/risk signals → heuristic claims
        """
        claims = []
        lines = text.strip().split("\n")

        current_section = ""
        prose_buffer: list[str] = []  # accumulate prose lines

        for line in lines:
            stripped = line.strip()

            # Track sections
            if stripped.startswith("##"):
                # Flush prose buffer from previous section
                if prose_buffer:
                    claims.extend(self._extract_prose_claims(
                        " ".join(prose_buffer), source_id, current_section
                    ))
                    prose_buffer = []
                current_section = stripped.lstrip("#").strip().lower()
                continue

            # Skip empty lines and top-level headers
            if not stripped or stripped.startswith("#"):
                continue

            # Markdown list items → structured claims
            if stripped.startswith("- ") and len(stripped) > 10:
                claim_text = stripped[2:].strip()
                weight_enum, tags = self._section_weight_tags(current_section)

                evidence = Evidence(
                    source_id=source_id,
                    text=claim_text,
                    weight=weight_enum.value,
                )
                durability = self._detect_durability(claim_text, tags)
                claim = Claim(
                    text=claim_text,
                    evidence=[evidence],
                    tags=tags,
                    polarity=_detect_polarity(claim_text),
                    durability=durability,
                )
                claim._recalc_confidence()
                claims.append(claim)
            else:
                # Non-list text → accumulate for prose extraction
                if len(stripped) > 20:
                    prose_buffer.append(stripped)

        # Flush remaining prose
        if prose_buffer:
            claims.extend(self._extract_prose_claims(
                " ".join(prose_buffer), source_id, current_section
            ))

        return claims

    @classmethod
    def _detect_durability(cls, text: str, tags: list[str]) -> ClaimDurability:
        """Detect claim durability from text content and tags."""
        if "decision" in tags:
            return ClaimDurability.STABLE
        if "todo" in tags:
            return ClaimDurability.TEMPORARY
        if cls._TEMPORARY_SIGNALS.search(text):
            return ClaimDurability.TEMPORARY
        return ClaimDurability.UNKNOWN

    @classmethod
    def _extract_foresight(cls, text: str, source_id: str) -> Optional[ForesightItem]:
        """Extract a foresight item if the text contains forward-looking signals."""
        if not cls._FORESIGHT_SIGNALS.search(text):
            return None
        # Determine valid_until from time keywords
        days = cls._FORESIGHT_DEFAULT_DAYS
        for pattern, d in cls._FORESIGHT_TIME_MAP:
            if pattern.search(text):
                days = d
                break
        return ForesightItem(
            text=text[:500],
            valid_until=time.time() + days * 86400,
            source_id=source_id,
            status=ForesightStatus.PENDING,
        )

    def _section_weight_tags(self, section: str) -> tuple:
        """Get weight and tags for a section name."""
        if section in self._SECTION_MAP:
            return self._SECTION_MAP[section]
        return (EvidenceWeight.TERTIARY, [])

    def _extract_prose_claims(self, text: str, source_id: str,
                               section: str) -> list[Claim]:
        """Extract claims from prose text using sentence splitting + signal detection."""
        claims = []
        # Split into sentences (handle both . and 。)
        sentences = re.split(r'(?<=[.!?。！？])\s+', text)

        for sent in sentences:
            sent = sent.strip()
            if len(sent) < 15:
                continue

            # Check for signals
            tags = []
            weight = EvidenceWeight.TERTIARY

            if self._DECISION_SIGNALS.search(sent):
                weight = EvidenceWeight.PRIMARY
                tags.append("decision")
            elif self._FACT_SIGNALS.search(sent):
                weight = EvidenceWeight.SECONDARY
                tags.append("fact")
            elif self._RISK_SIGNALS.search(sent):
                weight = EvidenceWeight.SECONDARY
                tags.append("risk")
            else:
                # No strong signal — skip unless in a high-value section
                section_info = self._SECTION_MAP.get(section)
                if section_info and section_info[0].value >= EvidenceWeight.SECONDARY.value:
                    weight = section_info[0]
                    tags = list(section_info[1])
                else:
                    continue  # skip low-signal prose

            evidence = Evidence(
                source_id=source_id,
                text=sent[:500],
                weight=weight.value,
            )
            durability = self._detect_durability(sent[:500], tags)
            claim = Claim(
                text=sent[:500],
                evidence=[evidence],
                tags=tags,
                polarity=_detect_polarity(sent[:500]),
                durability=durability,
            )
            claim._recalc_confidence()
            claims.append(claim)

        return claims

    def _load_extra_config(self) -> tuple[Optional[dict[str, list[str]]], Optional[list[str]]]:
        """Load user-defined extraction config from vault root.

        Returns ``(extra_patterns, extra_names)`` — both optional, both
        passed straight through to :func:`entity_extractor.extract_entities`.
        Patterns are user-defined typed regexes; names are literal strings
        that should always be promoted (project glossary).
        """
        if not self._entity_config_loaded:
            self._entity_config = _load_entity_config(self.vault.root)
            self._entity_config_loaded = True

        if not self._entity_config:
            return None, None

        patterns: dict[str, list[str]] = {}
        for etype, pats in self._entity_config.get("patterns", {}).items():
            if isinstance(pats, list):
                patterns[etype] = [str(p) for p in pats]

        names = self._entity_config.get("entities", [])
        names_list: Optional[list[str]] = None
        if isinstance(names, list) and names:
            names_list = [str(n) for n in names if n]

        return (patterns or None), names_list

    def _extract_entities(self, text: str, claims: list[Claim],
                          source_id: str) -> list[Entity]:
        """Extract entities from text via the reverse-filter pipeline.

        Delegates the regex / stopword work to :mod:`.entity_extractor`;
        this method only handles vault lookup (re-using existing entities
        for repeat names) and claim → entity linking.
        """
        from .entity_extractor import extract_entities

        extra_patterns, extra_names = self._load_extra_config()
        named_pairs = extract_entities(
            text,
            extra_patterns=extra_patterns,
            extra_names=extra_names,
        )

        entities_found: dict[str, Entity] = {}
        for name, entity_type in named_pairs:
            if name in entities_found:
                continue
            existing = self._find_existing_entity(name)
            if existing:
                if source_id not in existing.backlinks:
                    existing.backlinks.append(source_id)
                entities_found[name] = existing
            else:
                entities_found[name] = Entity(
                    name=name,
                    entity_type=entity_type,
                    backlinks=[source_id],
                )

        # Link claims to entities. The entity that appears EARLIEST in
        # the claim text wins — matches typical "subject-first" sentence
        # structure ("MCP 协议是…" → MCP is the subject, not 协议).
        # Resolves a non-determinism that the old pattern-iteration order
        # masked accidentally.
        for claim in claims:
            text_lower = claim.text.lower()
            earliest_pos = -1
            earliest_entity: Optional[Entity] = None
            for name, entity in entities_found.items():
                pos = text_lower.find(name.lower())
                if pos < 0:
                    continue
                if earliest_pos < 0 or pos < earliest_pos:
                    earliest_pos = pos
                    earliest_entity = entity
            if earliest_entity is not None:
                claim.entity_id = earliest_entity.id

        return list(entities_found.values())

    def _find_existing_entity(self, name: str) -> Optional[Entity]:
        """Find an existing entity by name or alias."""
        for eid in self.vault.list_entities():
            entity = self.vault.load_entity(eid)
            if entity and (entity.name.lower() == name.lower()
                          or name.lower() in [a.lower() for a in entity.aliases]):
                return entity
        return None

    def _update_compiled_truth(self, entity: Entity, claims: list[Claim],
                                source: Source) -> None:
        """Update an entity's compiled truth with new claims.

        Merges evidence for same-text claims and generates a structured summary
        grouped by claim type (decision/learning/action/todo).
        """
        ct = entity.compiled_truth

        # Add relevant claims, merging evidence for duplicates
        existing_by_text: dict[str, Claim] = {}
        for c in ct.claims:
            existing_by_text[c.text.strip().lower()] = c

        for claim in claims:
            if claim.entity_id != entity.id:
                continue
            key = claim.text.strip().lower()
            if key in existing_by_text:
                # Merge evidence without duplicating
                existing = existing_by_text[key]
                existing_source_ids = {e.source_id for e in existing.evidence}
                for ev in claim.evidence:
                    if ev.source_id not in existing_source_ids:
                        existing.add_evidence(ev)
            else:
                ct.claims.append(claim)
                existing_by_text[key] = claim

        # Extract foresight items from relevant claims
        for claim in claims:
            if claim.entity_id != entity.id:
                continue
            fi = self._extract_foresight(claim.text, source.id)
            if fi:
                # Avoid duplicate foresight items
                existing_texts = {f.text.strip().lower() for f in ct.foresight}
                if fi.text.strip().lower() not in existing_texts:
                    ct.foresight.append(fi)

        # Add timeline event
        import datetime
        today = datetime.date.today().isoformat()
        relevant_count = len([c for c in claims if c.entity_id == entity.id])
        ct.timeline.append(TimelineEvent(
            date=today,
            title=f"New evidence from: {source.title}",
            body=f"Extracted {relevant_count} claims",
            source_id=source.id,
        ))

        # Recompile structured summary from active claims
        ct.summary = self._compile_summary(ct.active_claims)
        ct.last_compiled_at = time.time()
        entity.updated_at = time.time()

    @staticmethod
    def _compile_summary(active_claims: list[Claim]) -> str:
        """Generate structured summary grouped by claim type."""
        if not active_claims:
            return ""

        # Group by primary tag
        groups: dict[str, list[Claim]] = {}
        for c in active_claims:
            primary_tag = c.tags[0] if c.tags else "other"
            groups.setdefault(primary_tag, []).append(c)

        # Sort each group by confidence descending, take top claims
        label_map = {
            "decision": "决策",
            "learning": "学习",
            "action": "操作",
            "todo": "待办",
            "fact": "事实",
            "risk": "风险",
            "other": "其他",
        }
        # Order: decision > learning > action > fact > risk > todo > other
        type_order = ["decision", "learning", "action", "fact", "risk", "todo", "other"]

        parts = []
        for tag in type_order:
            if tag not in groups:
                continue
            sorted_claims = sorted(groups[tag], key=lambda c: c.confidence, reverse=True)
            label = label_map.get(tag, tag)
            top = sorted_claims[:3]
            items = "; ".join(c.text[:120] for c in top)
            parts.append(f"[{label}] {items}")

        return " | ".join(parts) if parts else active_claims[0].text[:200]

    @staticmethod
    def _consolidate_entity_claims(entity: Entity) -> None:
        """Deduplicate similar claims in an entity using MemoryConsolidator.

        Superseded claims are marked SUPERSEDED, keeping the newest version.
        Temporary claims use a lower similarity threshold (0.5 vs 0.6).
        """
        from ..search.consolidator import MemoryConsolidator

        ct = entity.compiled_truth
        active = ct.active_claims
        if len(active) < 2:
            return

        # Convert claims to consolidator input format
        facts = []
        claim_map: dict[str, Claim] = {}  # claim_id -> Claim
        for c in active:
            facts.append({
                "text": c.text,
                "source_id": c.evidence[0].source_id if c.evidence else "",
                "timestamp": c.created_at,
                "entities": [entity.name],
                "_claim_id": c.id,
            })
            claim_map[c.id] = c

        consolidated = MemoryConsolidator.consolidate(facts)

        # Phase 1: Mark superseded from standard consolidation (Jaccard >= 0.6)
        authoritative_texts = {cf.fact.strip().lower() for cf in consolidated if cf.weight == 1.0}
        for c in active:
            if c.text.strip().lower() not in authoritative_texts:
                from ..search.consolidator import MemoryConsolidator as MC
                for auth_text in authoritative_texts:
                    if MC._jaccard_similarity(c.text, auth_text) >= MC.SIMILARITY_THRESHOLD:
                        c.status = ClaimStatus.SUPERSEDED
                        break

        # Phase 2: Durability-aware supersession for temporary claims
        # Use lower threshold (0.5) for temporary claims that survived Phase 1
        still_active = [c for c in active if c.status == ClaimStatus.ACTIVE]
        # Sort by created_at descending (newest first)
        sorted_active = sorted(still_active, key=lambda c: c.created_at, reverse=True)
        superseded_ids: set[str] = set()
        for i, newer in enumerate(sorted_active):
            if newer.id in superseded_ids:
                continue
            for older in sorted_active[i + 1:]:
                if older.id in superseded_ids:
                    continue
                # Only apply lower threshold when at least one is temporary
                if newer.durability != ClaimDurability.TEMPORARY and older.durability != ClaimDurability.TEMPORARY:
                    continue
                sim = MemoryConsolidator._jaccard_similarity(newer.text, older.text)
                if sim >= 0.5:
                    older.status = ClaimStatus.SUPERSEDED
                    superseded_ids.add(older.id)

    def detect_contradictions(self) -> list[dict]:
        """Scan all entities for contradictory claims."""
        contradictions = []
        for eid in self.vault.list_entities():
            entity = self.vault.load_entity(eid)
            if not entity:
                continue
            pairs = entity.compiled_truth.contradictions
            for a, b in pairs:
                contradictions.append({
                    "entity": entity.name,
                    "claim_a": {"id": a.id, "text": a.text, "confidence": a.confidence},
                    "claim_b": {"id": b.id, "text": b.text, "confidence": b.confidence},
                })
        return contradictions
