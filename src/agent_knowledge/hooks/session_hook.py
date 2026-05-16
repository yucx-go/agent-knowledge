"""SessionHook: compress session into summary and ingest on session_end."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .base import Hook, HookResult

if TYPE_CHECKING:
    from ..core.compiler import Compiler
    from ..core.umsf import UMSFEvent
    from ..core.vault import Vault


class SessionHook(Hook):
    """Fires on UMSF ``session_end`` events. Looks up the message list in
    ``event.metadata["messages"]`` (a list of ``{role, text}`` dicts)
    and ingests it as a compressed conversation summary.

    Why metadata-based: UMSFEvent intentionally has no native "messages"
    field — that data lives in *separate* message events on the parent
    document. But session_end is often emitted by a callback that has
    already collected the prior messages in-memory; passing them on
    metadata avoids requiring callers to re-emit each message just to
    get end-of-session compression.
    """

    name = "session"
    event_types = ["session_end"]

    def should_fire(self, event: "UMSFEvent") -> bool:
        if event.type not in self.event_types:
            return False
        messages = event.metadata.get("messages", [])
        return len(messages) >= 2  # need at least a back-and-forth

    def process(
        self,
        event: "UMSFEvent",
        session_id: str,
        vault: "Vault",
        compiler: "Compiler",
    ) -> HookResult:
        messages = event.metadata.get("messages", [])

        # Compress messages into a summary
        summary_lines: list[str] = []
        for msg in messages:
            role = msg.get("role", "unknown")
            text = msg.get("text", "")[:200]
            if text:
                summary_lines.append(f"[{role}] {text}")

        # Keep summary under 5000 chars
        summary = "\n".join(summary_lines)[:5000]

        source = compiler.ingest(
            summary,
            title=f"session:{session_id or 'unknown'}",
            source_type="conversation",
        )

        return HookResult(
            claims=[],  # claims are extracted by compiler.ingest
            source_id=source.id,
            entities_found=source.entities_extracted,
        )
