"""Vault: local filesystem-based knowledge store.

Directory layout:
    <vault_root>/
    ├── .ak-schema.yaml          # vault metadata
    ├── sources/                  # raw source pages
    │   └── <source_id>.yaml
    ├── entities/                 # entity pages with compiled truth
    │   └── <entity_id>.yaml
    ├── concepts/                 # concept/topic pages
    │   └── <concept_id>.yaml
    ├── syntheses/                # cross-source synthesis
    │   └── <synthesis_id>.yaml
    └── reports/                  # lint, contradictions, dashboard
        └── lint.yaml
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import yaml

from .models import Claim, Entity, Evidence, Source


class Vault:
    """Local filesystem knowledge vault."""

    SCHEMA_FILE = ".ak-schema.yaml"
    DIRS = ("sources", "entities", "concepts", "syntheses", "reports")
    _DEFAULT_CACHE_MAXSIZE = 256

    def __init__(self, root: str | Path, cache_maxsize: int = _DEFAULT_CACHE_MAXSIZE):
        self.root = Path(root).expanduser().resolve()
        self._cache_maxsize = cache_maxsize
        self._source_cache: dict[str, Source] = {}
        self._entity_cache: dict[str, Entity] = {}
        self._cache_order_source: list[str] = []  # LRU order (oldest first)
        self._cache_order_entity: list[str] = []
        self._event_index = None  # lazy-initialized; see event_index property

    @property
    def event_index(self):
        """Companion structured-event store (SQLite). Lazy-initialized.

        Populated automatically by :func:`agent_knowledge.core.umsf.ingest_umsf`,
        but callers can also write to it directly (e.g. backfill jobs).
        """
        if self._event_index is None:
            from .event_index import EventIndex

            self._event_index = EventIndex(self.root)
        return self._event_index

    @property
    def schema_path(self) -> Path:
        return self.root / self.SCHEMA_FILE

    @property
    def is_initialized(self) -> bool:
        return self.schema_path.exists()

    def init(self, lang: str = "zh") -> None:
        """Initialize a new vault."""
        if self.is_initialized:
            return

        self.root.mkdir(parents=True, exist_ok=True)
        for d in self.DIRS:
            (self.root / d).mkdir(exist_ok=True)

        schema = {
            "version": "0.1.0",
            "language": lang,
            "created_by": "agent-knowledge",
        }
        self.schema_path.write_text(
            yaml.dump(schema, allow_unicode=True, default_flow_style=False),
            encoding="utf-8",
        )

    def _ensure_init(self) -> None:
        if not self.is_initialized:
            raise FileNotFoundError(
                f"Vault not initialized at {self.root}. Run `ak init` first."
            )

    # ── Sources ──

    def clear_cache(self) -> None:
        """Clear all in-memory caches."""
        self._source_cache.clear()
        self._entity_cache.clear()
        self._cache_order_source.clear()
        self._cache_order_entity.clear()

    def _cache_put_source(self, source_id: str, source: Source) -> None:
        """Add/update source in LRU cache, evicting oldest if over maxsize."""
        if source_id in self._source_cache:
            self._cache_order_source.remove(source_id)
        elif len(self._source_cache) >= self._cache_maxsize:
            evict = self._cache_order_source.pop(0)
            self._source_cache.pop(evict, None)
        self._source_cache[source_id] = source
        self._cache_order_source.append(source_id)

    def _cache_put_entity(self, entity_id: str, entity: Entity) -> None:
        """Add/update entity in LRU cache, evicting oldest if over maxsize."""
        if entity_id in self._entity_cache:
            self._cache_order_entity.remove(entity_id)
        elif len(self._entity_cache) >= self._cache_maxsize:
            evict = self._cache_order_entity.pop(0)
            self._entity_cache.pop(evict, None)
        self._entity_cache[entity_id] = entity
        self._cache_order_entity.append(entity_id)

    def save_source(self, source: Source) -> Path:
        self._ensure_init()
        p = self.root / "sources" / f"{source.id}.yaml"
        data = {
            "id": source.id,
            "title": source.title,
            "source_type": source.source_type,
            "path": source.path,
            "url": source.url,
            "ingested_at": source.ingested_at,
            "claims_extracted": source.claims_extracted,
            "entities_extracted": source.entities_extracted,
            "content": source.content[:10000],  # truncate for storage (10K chars)
        }
        # Audit trail: original UMSFDocument that produced this source.
        # Only persisted when present — keeps existing source files unchanged.
        if source.umsf is not None:
            data["umsf"] = source.umsf
        p.write_text(
            yaml.dump(data, allow_unicode=True, default_flow_style=False),
            encoding="utf-8",
        )
        self._cache_put_source(source.id, source)
        return p

    def load_source(self, source_id: str) -> Optional[Source]:
        # Check cache first
        if source_id in self._source_cache:
            return self._source_cache[source_id]
        p = self.root / "sources" / f"{source_id}.yaml"
        if not p.exists():
            return None
        data = yaml.safe_load(p.read_text(encoding="utf-8"))
        source = Source(**{k: v for k, v in data.items() if k in Source.__dataclass_fields__})
        self._cache_put_source(source_id, source)
        return source

    def list_sources(self) -> list[str]:
        d = self.root / "sources"
        if not d.exists():
            return []
        return [f.stem for f in d.glob("*.yaml")]

    # ── Entities ──

    def save_entity(self, entity: Entity) -> Path:
        self._ensure_init()
        self._cache_put_entity(entity.id, entity)
        p = self.root / "entities" / f"{entity.id}.yaml"
        data = {
            "id": entity.id,
            "name": entity.name,
            "aliases": entity.aliases,
            "entity_type": entity.entity_type,
            "backlinks": entity.backlinks,
            "created_at": entity.created_at,
            "updated_at": entity.updated_at,
            "compiled_truth": {
                "summary": entity.compiled_truth.summary,
                "claims": [c.to_dict() for c in entity.compiled_truth.claims],
                "foresight": [f.to_dict() for f in entity.compiled_truth.foresight],
                "last_compiled_at": entity.compiled_truth.last_compiled_at,
            },
            "timeline": [
                {
                    "date": e.date,
                    "title": e.title,
                    "body": e.body,
                    "source_id": e.source_id,
                }
                for e in entity.compiled_truth.timeline
            ],
        }
        p.write_text(
            yaml.dump(data, allow_unicode=True, default_flow_style=False),
            encoding="utf-8",
        )
        return p

    def load_entity(self, entity_id: str) -> Optional[Entity]:
        # Check cache first
        if entity_id in self._entity_cache:
            return self._entity_cache[entity_id]
        p = self.root / "entities" / f"{entity_id}.yaml"
        if not p.exists():
            return None
        data = yaml.safe_load(p.read_text(encoding="utf-8"))
        # Reconstruct entity (simplified)
        from .models import CompiledTruth, ForesightItem, TimelineEvent

        ct_data = data.get("compiled_truth", {})
        claims = [Claim.from_dict(c) for c in ct_data.get("claims", [])]
        foresight = [ForesightItem.from_dict(f) for f in ct_data.get("foresight", [])]
        timeline = [
            TimelineEvent(**t) for t in data.get("timeline", [])
        ]
        ct = CompiledTruth(
            summary=ct_data.get("summary", ""),
            claims=claims,
            timeline=timeline,
            foresight=foresight,
            last_compiled_at=ct_data.get("last_compiled_at", 0),
        )
        entity = Entity(
            id=data["id"],
            name=data.get("name", ""),
            aliases=data.get("aliases", []),
            entity_type=data.get("entity_type", ""),
            compiled_truth=ct,
            backlinks=data.get("backlinks", []),
            created_at=data.get("created_at", 0),
            updated_at=data.get("updated_at", 0),
        )
        self._cache_put_entity(entity_id, entity)
        return entity

    def list_entities(self) -> list[str]:
        d = self.root / "entities"
        if not d.exists():
            return []
        return [f.stem for f in d.glob("*.yaml")]

    # ── Stats ──

    def stats(self) -> dict:
        """Return vault statistics."""
        self._ensure_init()
        return {
            "sources": len(self.list_sources()),
            "entities": len(self.list_entities()),
            "concepts": len(list((self.root / "concepts").glob("*.yaml"))) if (self.root / "concepts").exists() else 0,
            "syntheses": len(list((self.root / "syntheses").glob("*.yaml"))) if (self.root / "syntheses").exists() else 0,
            "reports": len(list((self.root / "reports").glob("*.yaml"))) if (self.root / "reports").exists() else 0,
        }
