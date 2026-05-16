"""Base classes for the auto-capture hook system (UMSF-native).

A Hook subscribes to a subset of UMSFEvent types and reacts when one of
those events flows through HookManager. Hooks share three primitives:

    Hook       — abstract base; subclasses set ``event_types`` (a list of
                 UMSFEvent.type values) and implement should_fire/process.
    HookManager — registry + dispatcher; accepts UMSFDocument inputs,
                 ingests them via the standard Compiler pipeline, and
                 fans out each event to every matching hook.
    HookResult  — what a hook returns after processing one event.

Why UMSF-native (no HookEvent)
------------------------------
HookEvent was a separate envelope from UMSFEvent during the migration
period. Now that all adapters and the MCP server emit UMSF directly,
the bridge layer is pure overhead. Operating on UMSFEvent end-to-end
means: fewer types, no lossy translations, structured tool args
preserved, and one clear path through the system.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..core.compiler import Compiler
    from ..core.models import Claim
    from ..core.umsf import UMSFDocument, UMSFEvent
    from ..core.vault import Vault


@dataclass
class HookResult:
    """Output produced by a single hook invocation."""

    claims: list["Claim"] = field(default_factory=list)
    source_id: str = ""
    entities_found: list[str] = field(default_factory=list)


class Hook(ABC):
    """Abstract base class for auto-capture hooks.

    Subclasses set:
        - ``name``       : stable identifier (used for stats/logging)
        - ``event_types``: list of UMSFEvent.type values to subscribe to

    Subclasses implement:
        - ``should_fire(event)`` — return True iff this hook handles ``event``.
          The default implementation matches ``event.type`` against
          ``event_types``; override to add content-based filtering.
        - ``process(event, session_id, vault, compiler)`` — extract
          knowledge from ``event``, persist via ``compiler``, return a
          :class:`HookResult`.
    """

    name: str = ""
    event_types: list[str] = []

    def should_fire(self, event: "UMSFEvent") -> bool:
        """Return True if this hook should process ``event``.

        Default: match on ``event.type`` membership. Subclasses may add
        content-based filtering (e.g. role checks for message hooks).
        """
        return event.type in self.event_types

    @abstractmethod
    def process(
        self,
        event: "UMSFEvent",
        session_id: str,
        vault: "Vault",
        compiler: "Compiler",
    ) -> HookResult:
        """Process an event and return extracted knowledge.

        Implementations MUST persist results through *compiler* (e.g.
        ``compiler.ingest``), not by writing to the vault directly.
        ``session_id`` comes from the parent UMSFDocument and is passed
        explicitly so hooks don't need to walk back up to the document.
        """
        ...


class HookManager:
    """Registry + dispatcher for hooks. Accepts UMSFDocument inputs.

    The dispatcher runs every UMSFDocument through two parallel tracks:

      1. **Full ingestion** — :func:`ingest_umsf` renders the document
         and creates a single Source with extracted claims/entities.
         This is the authoritative knowledge-layer path.

      2. **Per-event hook fan-out** — each event in the document is
         offered to every registered hook. Hooks that match produce
         additional, more specialized claims (e.g. high-confidence
         decision claims with stable durability).

    Both tracks are best-effort: failures in one don't suppress the
    other. The combined HookResult list lets callers see what each
    track produced.
    """

    def __init__(self, vault: "Vault"):
        from ..core.compiler import Compiler

        self.vault = vault
        self.compiler = Compiler(vault)
        self._hooks: list[Hook] = []
        self._fire_counts: dict[str, int] = {}

    # ── Registration ──

    def register(self, hook: Hook) -> None:
        self._hooks.append(hook)
        self._fire_counts.setdefault(hook.name, 0)

    def register_defaults(self) -> None:
        """Register all built-in hooks."""
        from .decision_hook import DecisionHook
        from .error_hook import ErrorHook
        from .file_hook import FileHook
        from .message_hook import MessageHook
        from .response_hook import ResponseHook
        from .session_hook import SessionHook
        from .tool_hook import ToolHook

        for cls in (
            MessageHook,
            ToolHook,
            ResponseHook,
            ErrorHook,
            SessionHook,
            DecisionHook,
            FileHook,
        ):
            self.register(cls())

    # ── Dispatch ──

    def fire(self, doc: "UMSFDocument") -> list[HookResult]:
        """Ingest a UMSFDocument and fan out hook notifications.

        Returns the aggregated list of HookResult from every track:
        one for the full UMSF ingestion (with the persisted source_id),
        plus one for each fired hook.
        """
        from ..core.umsf import ingest_umsf, validate

        validate(doc)
        results: list[HookResult] = []

        # Track 1: full ingestion via UMSF renderer + Compiler.
        # Surfaces source_id + auto-extracted entities; claims are
        # already persisted so we don't re-list them here.
        try:
            source = ingest_umsf(self.compiler, doc)
            results.append(HookResult(
                source_id=source.id,
                entities_found=source.entities_extracted,
            ))
        except Exception:
            # Track 1 failed; let track 2 try anyway — better partial
            # than total failure when hooks could still extract something.
            pass

        # Track 2: per-event hook fan-out for specialized extraction.
        for ev in doc.events:
            for hook in self._hooks:
                if hook.should_fire(ev):
                    res = hook.process(ev, doc.session_id, self.vault, self.compiler)
                    results.append(res)
                    self._fire_counts[hook.name] = (
                        self._fire_counts.get(hook.name, 0) + 1
                    )

        return results

    # ── Introspection ──

    def list_hooks(self) -> list[dict]:
        return [
            {"name": h.name, "event_types": list(h.event_types)}
            for h in self._hooks
        ]

    def stats(self) -> dict:
        """Return per-hook fire counts."""
        return dict(self._fire_counts)
