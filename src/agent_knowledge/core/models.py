"""Core data models: Claim, Evidence, Entity, Source, CompiledTruth"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class ClaimDurability(Enum):
    STABLE = "stable"  # long-lived facts (e.g., "is a programmer")
    TEMPORARY = "temporary"  # transient state (e.g., "busy this week")
    UNKNOWN = "unknown"  # default


class ClaimPolarity(Enum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"


class ClaimStatus(Enum):
    ACTIVE = "active"
    SUPERSEDED = "superseded"  # replaced by newer evidence
    CONTRADICTED = "contradicted"  # conflicting evidence found
    RETRACTED = "retracted"  # explicitly retracted


class EvidenceWeight(Enum):
    PRIMARY = 1.0  # first-hand data (MCP, API)
    SECONDARY = 0.7  # structured knowledge (wiki, docs)
    TERTIARY = 0.4  # web search, external
    PRIOR = 0.2  # prior knowledge, estimation


@dataclass
class Evidence:
    """A piece of evidence supporting or contradicting a claim."""

    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    source_id: str = ""  # which source page
    text: str = ""  # the actual evidence text
    weight: float = 0.7
    timestamp: float = field(default_factory=time.time)
    path: Optional[str] = None  # file path / URL
    lines: Optional[str] = None  # line range in source

    def to_dict(self) -> dict:
        d = {
            "id": self.id,
            "source_id": self.source_id,
            "text": self.text,
            "weight": self.weight,
            "timestamp": self.timestamp,
        }
        if self.path:
            d["path"] = self.path
        if self.lines:
            d["lines"] = self.lines
        return d

    @classmethod
    def from_dict(cls, d: dict) -> Evidence:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class Claim:
    """A structured assertion with confidence and evidence trail."""

    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    text: str = ""
    confidence: float = 0.5  # 0.0 - 1.0
    status: ClaimStatus = ClaimStatus.ACTIVE
    polarity: ClaimPolarity = ClaimPolarity.NEUTRAL
    evidence: list[Evidence] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    entity_id: Optional[str] = None  # linked entity
    tags: list[str] = field(default_factory=list)
    durability: ClaimDurability = ClaimDurability.UNKNOWN

    def add_evidence(self, ev: Evidence) -> None:
        self.evidence.append(ev)
        self._recalc_confidence()
        self.updated_at = time.time()

    def _recalc_confidence(self) -> None:
        """Recalculate confidence from evidence weights."""
        if not self.evidence:
            self.confidence = 0.0
            return
        total_weight = sum(e.weight for e in self.evidence)
        # More evidence + higher weight = higher confidence, capped at 0.95
        self.confidence = min(0.95, total_weight / (total_weight + 1.0))

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "text": self.text,
            "confidence": round(self.confidence, 3),
            "status": self.status.value,
            "polarity": self.polarity.value,
            "evidence": [e.to_dict() for e in self.evidence],
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "entity_id": self.entity_id,
            "tags": self.tags,
            "durability": self.durability.value,
        }

    @classmethod
    def from_dict(cls, d: dict) -> Claim:
        evidence = [Evidence.from_dict(e) for e in d.get("evidence", [])]
        status = ClaimStatus(d.get("status", "active"))
        polarity = ClaimPolarity(d.get("polarity", "neutral"))
        durability = ClaimDurability(d.get("durability", "unknown"))
        return cls(
            id=d.get("id", str(uuid.uuid4())[:8]),
            text=d.get("text", ""),
            confidence=d.get("confidence", 0.5),
            status=status,
            polarity=polarity,
            evidence=evidence,
            created_at=d.get("created_at", time.time()),
            updated_at=d.get("updated_at", time.time()),
            entity_id=d.get("entity_id"),
            tags=d.get("tags", []),
            durability=durability,
        )


@dataclass
class TimelineEvent:
    """An append-only timeline entry."""

    date: str  # YYYY-MM-DD
    title: str
    body: str = ""
    source_id: Optional[str] = None
    timestamp: float = field(default_factory=time.time)


class ForesightStatus(Enum):
    PENDING = "pending"
    COMPLETED = "completed"
    EXPIRED = "expired"


@dataclass
class ForesightItem:
    """A forward-looking item (plan, todo, intention) with expiry."""

    text: str = ""
    valid_until: float = 0.0  # timestamp
    source_id: str = ""
    status: ForesightStatus = ForesightStatus.PENDING

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "valid_until": self.valid_until,
            "source_id": self.source_id,
            "status": self.status.value,
        }

    @classmethod
    def from_dict(cls, d: dict) -> ForesightItem:
        return cls(
            text=d.get("text", ""),
            valid_until=d.get("valid_until", 0.0),
            source_id=d.get("source_id", ""),
            status=ForesightStatus(d.get("status", "pending")),
        )


@dataclass
class CompiledTruth:
    """The current best understanding of an entity/concept.

    Compiled Truth is rewritten when new evidence arrives.
    Timeline is append-only.
    """

    summary: str = ""  # current compiled understanding
    claims: list[Claim] = field(default_factory=list)
    timeline: list[TimelineEvent] = field(default_factory=list)
    foresight: list[ForesightItem] = field(default_factory=list)
    last_compiled_at: float = field(default_factory=time.time)

    @property
    def active_claims(self) -> list[Claim]:
        return [c for c in self.claims if c.status == ClaimStatus.ACTIVE]

    @property
    def contradictions(self) -> list[tuple[Claim, Claim]]:
        """Find pairs of active claims that contradict each other.

        A contradiction requires:
        1. Same entity_id (same topic)
        2. Both have confidence > 0.5
        3. Opposite polarity signals detected
        """
        pairs = []
        active = self.active_claims
        for i, a in enumerate(active):
            for b in active[i + 1 :]:
                if not (a.entity_id and a.entity_id == b.entity_id):
                    continue
                if a.confidence <= 0.5 or b.confidence <= 0.5:
                    continue
                if _claims_contradict(a, b):
                    pairs.append((a, b))
        return pairs


# ── Polarity detection helpers ──

# Pairs of opposing keywords (each tuple = positive_signal, negative_signal)
_OPPOSING_PAIRS: list[tuple[str, str]] = [
    ("增", "减"), ("选", "弃"), ("启用", "禁用"), ("是", "不是"),
    ("采用", "放弃"), ("开启", "关闭"), ("成功", "失败"),
    ("升级", "降级"), ("提升", "下降"), ("支持", "不支持"),
    ("increase", "decrease"), ("enable", "disable"),
    ("select", "reject"), ("adopt", "abandon"),
    ("start", "stop"), ("success", "failure"),
    ("upgrade", "downgrade"), ("approve", "deny"),
]


def _detect_polarity(text: str) -> ClaimPolarity:
    """Detect polarity from claim text using keyword signals."""
    text_lower = text.lower()
    pos_score = 0
    neg_score = 0
    for pos_kw, neg_kw in _OPPOSING_PAIRS:
        if pos_kw in text_lower:
            pos_score += 1
        if neg_kw in text_lower:
            neg_score += 1
    if pos_score > neg_score:
        return ClaimPolarity.POSITIVE
    elif neg_score > pos_score:
        return ClaimPolarity.NEGATIVE
    return ClaimPolarity.NEUTRAL


def _claims_contradict(a: Claim, b: Claim) -> bool:
    """Check if two claims have opposing polarity on the same topic."""
    pol_a = a.polarity if a.polarity != ClaimPolarity.NEUTRAL else _detect_polarity(a.text)
    pol_b = b.polarity if b.polarity != ClaimPolarity.NEUTRAL else _detect_polarity(b.text)

    # Only contradict if explicitly opposite
    if pol_a == ClaimPolarity.NEUTRAL or pol_b == ClaimPolarity.NEUTRAL:
        return False
    return pol_a != pol_b


@dataclass
class Entity:
    """A named entity (person, project, tech, org)."""

    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    name: str = ""
    aliases: list[str] = field(default_factory=list)
    entity_type: str = ""  # person / project / tech / org / concept
    compiled_truth: CompiledTruth = field(default_factory=CompiledTruth)
    backlinks: list[str] = field(default_factory=list)  # source IDs that mention this
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)


@dataclass
class Source:
    """A raw source document / text / conversation.

    The optional ``umsf`` field preserves the original UMSFDocument that
    produced this source (when ingested via :func:`ingest_umsf`). Sources
    created from raw text via :meth:`Compiler.ingest` directly leave it
    None. The audit trail enables future reprocessing — re-extracting
    claims with a newer compiler, replaying events into a different
    index, or debugging extraction quality — without re-fetching from
    the original adapter.
    """

    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    title: str = ""
    content: str = ""
    source_type: str = ""  # file / url / text / conversation / feishu_doc
    path: Optional[str] = None
    url: Optional[str] = None
    ingested_at: float = field(default_factory=time.time)
    claims_extracted: list[str] = field(default_factory=list)  # claim IDs
    entities_extracted: list[str] = field(default_factory=list)  # entity IDs
    umsf: Optional[dict] = None  # original UMSFDocument as dict, if any
