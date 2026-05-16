"""ErrorHook: capture error patterns as stable pitfall claims."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..core.models import Claim, ClaimDurability, Evidence, EvidenceWeight
from .base import Hook, HookResult

if TYPE_CHECKING:
    from ..core.compiler import Compiler
    from ..core.umsf import UMSFEvent
    from ..core.vault import Vault


class ErrorHook(Hook):
    """Fires on UMSF ``error`` events. UMSFEvent fields used:

    - ``error``    — error message (preferred); falls back to ``content``
    - ``name``     — name of the failing tool, when applicable
    - ``metadata`` — adapter-specific extras; ``context`` lives here
    """

    name = "error"
    event_types = ["error"]

    def should_fire(self, event: "UMSFEvent") -> bool:
        if event.type not in self.event_types:
            return False
        return bool(event.error or event.content)

    def process(
        self,
        event: "UMSFEvent",
        session_id: str,
        vault: "Vault",
        compiler: "Compiler",
    ) -> HookResult:
        error_msg = event.error or event.content
        tool_name = event.name
        context = event.metadata.get("context", "")

        parts = [f"Error: {error_msg}"]
        if tool_name:
            parts.append(f"Tool: {tool_name}")
        if context:
            parts.append(f"Context: {context}")
        summary = " | ".join(parts)

        source = compiler.ingest(
            summary,
            title=f"error:{tool_name or 'unknown'}",
            source_type="error",
        )

        evidence = Evidence(
            source_id=source.id,
            text=summary[:500],
            weight=EvidenceWeight.PRIMARY.value,
            timestamp=event.ts,
        )
        claim = Claim(
            text=summary[:500],
            evidence=[evidence],
            tags=["error", "pitfall"],
            durability=ClaimDurability.STABLE,
        )
        claim._recalc_confidence()

        return HookResult(
            claims=[claim],
            source_id=source.id,
            entities_found=source.entities_extracted,
        )
