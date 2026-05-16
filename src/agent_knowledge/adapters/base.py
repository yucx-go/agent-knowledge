"""Base classes for pull adapters.

A pull adapter bridges an external system (a curated MEMORY.md file,
Claude Code JSONL transcripts, Codex sessions, IM messages, …) into
agent-knowledge.
The contract is small:

    1. discover(**opts) -> list[DiscoveredItem]
       Enumerate what's available to pull.

    2. fetch(item) -> Optional[UMSFDocument]
       Convert one discovered item into a UMSFDocument.
       Return None to skip without recording an error.

The base class provides everything else: orchestration, fingerprint-based
dedup, dry-run, error collection, state persistence, and ingestion via the
UMSF boundary (so all adapters take the same path through Compiler).

State persistence
-----------------
Every adapter shares a single state file at ``vault/.ak-pull-state.json``,
namespaced by adapter ``name``. This means many adapters can write to the
same vault without colliding, and inspecting state for a single adapter is
just one dict lookup.

    {
      "markdown-memory": {
        "/home/user/.claude/CLAUDE.md": {
          "fingerprint": "1234:1700000000.5",
          "label": "[claude-code] CLAUDE (workspace_instructions)",
          "last_source_id": "abc123",
          "last_pulled_at": 1700000123.0,
          "metadata": {...}
        }
      },
      "claude-code": {...}   # future adapter writes here
    }

Why fingerprint instead of mtime
--------------------------------
File-based adapters use ``f"{size}:{mtime}"`` as fingerprint. Session-based
adapters might use ``last_event_id`` or a content hash. The base class
doesn't care — any string that changes when content changes works.
"""

from __future__ import annotations

import json
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from ..core.umsf import UMSFDocument
    from ..core.vault import Vault


@dataclass
class DiscoveredItem:
    """A unit of work surfaced by :meth:`BasePullAdapter.discover`.

    Subclasses (e.g. :class:`DiscoveredFile` in markdown_memory) may add
    adapter-specific fields. The base only requires ``id`` and
    ``fingerprint`` — everything else is metadata.
    """
    id: str = ""              # stable identifier for dedup
    fingerprint: str = ""     # content-changes-when-this-changes
    label: str = ""           # human-readable label for reports
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class PullReport:
    """Outcome of a :meth:`BasePullAdapter.pull` invocation."""
    discovered: list[DiscoveredItem] = field(default_factory=list)
    ingested: list[str] = field(default_factory=list)            # source_ids
    skipped: list[str] = field(default_factory=list)             # item ids
    errors: list[tuple[str, str]] = field(default_factory=list)  # (id, message)
    duration_seconds: float = 0.0


class BasePullAdapter(ABC):
    """Abstract base for all pull adapters.

    Subclasses must:
        - set the class attribute ``name`` (used as the state-file key)
        - implement :meth:`discover` and :meth:`fetch`

    Subclasses may override :meth:`pull` for fully custom orchestration,
    but the default implementation handles UMSF ingestion + dedup +
    error collection — almost everyone should keep it.
    """

    name: str = "generic"
    STATE_FILE = ".ak-pull-state.json"

    def __init__(self, vault: "Vault"):
        # Lazy import avoids a circular dependency: compiler imports from
        # search, which we don't need here.
        from ..core.compiler import Compiler

        self.vault = vault
        self.compiler = Compiler(vault)

    # ── Subclass contract ──

    @abstractmethod
    def discover(self, **opts: Any) -> list[DiscoveredItem]:
        """Enumerate items available for ingestion.

        Implementations may take adapter-specific keyword arguments (e.g.
        ``agents=...`` for markdown_memory, ``since_days=...`` for
        session adapters). These flow through from :meth:`pull` via
        ``**opts``.
        """

    @abstractmethod
    def fetch(self, item: DiscoveredItem) -> Optional["UMSFDocument"]:
        """Convert one discovered item into a UMSFDocument.

        Return ``None`` to skip (e.g. file is empty) without recording an
        error. Raise an exception to record an error and continue with
        the next item.
        """

    # ── Default orchestration ──

    def pull(
        self,
        force: bool = False,
        dry_run: bool = False,
        **opts: Any,
    ) -> PullReport:
        """Discover, fetch, and ingest items via the UMSF boundary.

        Args:
            force: Re-ingest even if the fingerprint matches a previous pull.
            dry_run: Discover only — don't fetch, ingest, or update state.
            **opts: Forwarded to :meth:`discover`.
        """
        from ..core.umsf import ingest_umsf

        t0 = time.time()
        report = PullReport()

        items = self.discover(**opts)
        report.discovered = list(items)

        if dry_run:
            report.duration_seconds = time.time() - t0
            return report

        state = self._load_state()
        adapter_state: dict = state.setdefault(self.name, {})

        for item in items:
            prev = adapter_state.get(item.id)
            if not force and prev and prev.get("fingerprint") == item.fingerprint:
                report.skipped.append(item.id)
                continue

            try:
                doc = self.fetch(item)
            except Exception as e:
                report.errors.append((item.id, f"fetch_error: {e}"))
                continue

            if doc is None:
                report.skipped.append(item.id)
                continue

            try:
                source = ingest_umsf(self.compiler, doc)
            except Exception as e:
                report.errors.append((item.id, f"ingest_error: {e}"))
                continue

            report.ingested.append(source.id)
            adapter_state[item.id] = {
                "fingerprint": item.fingerprint,
                "label": item.label,
                "last_source_id": source.id,
                "last_pulled_at": time.time(),
                "metadata": dict(item.metadata),
            }

        self._save_state(state)
        report.duration_seconds = time.time() - t0
        return report

    # ── State persistence ──

    def _state_path(self) -> Path:
        return self.vault.root / self.STATE_FILE

    def _load_state(self) -> dict:
        p = self._state_path()
        if not p.exists():
            return {}
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return {}

    def _save_state(self, state: dict) -> None:
        p = self._state_path()
        try:
            p.write_text(
                json.dumps(state, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        except Exception:
            pass
