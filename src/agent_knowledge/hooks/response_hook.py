"""ResponseHook: detect agent commitments and self-corrections."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from ..core.models import Claim, ClaimDurability, Evidence, EvidenceWeight
from .base import Hook, HookResult

if TYPE_CHECKING:
    from ..core.compiler import Compiler
    from ..core.umsf import UMSFEvent
    from ..core.vault import Vault

_COMMITMENT_PATTERNS = re.compile(
    r"(?:我会|已经完成|已完成|将会|我已经|马上|立即"
    r"|i will|i've completed|done|completed|will do|i have)",
    re.IGNORECASE,
)

_CORRECTION_PATTERNS = re.compile(
    r"(?:之前说错了|更正|纠正|修正|实际上|抱歉.*?搞错"
    r"|i was wrong|correction|actually|my mistake|let me correct)",
    re.IGNORECASE,
)


class ResponseHook(Hook):
    """Fires on UMSF ``message`` events from the assistant.

    Pairs with MessageHook (user-side) — together they cover both halves
    of a conversation, distinguished by ``event.role``.
    """

    name = "response"
    event_types = ["message"]

    def should_fire(self, event: "UMSFEvent") -> bool:
        if event.type not in self.event_types:
            return False
        # Assistant messages only — MessageHook handles user-side
        if event.role != "assistant":
            return False
        text = event.content
        if len(text) < 5:
            return False
        return bool(
            _COMMITMENT_PATTERNS.search(text)
            or _CORRECTION_PATTERNS.search(text)
        )

    def process(
        self,
        event: "UMSFEvent",
        session_id: str,
        vault: "Vault",
        compiler: "Compiler",
    ) -> HookResult:
        text = event.content
        tags: list[str] = []
        durability = ClaimDurability.TEMPORARY

        if _CORRECTION_PATTERNS.search(text):
            tags.append("correction")
            durability = ClaimDurability.STABLE  # corrections are important to remember
        if _COMMITMENT_PATTERNS.search(text):
            tags.append("commitment")

        source = compiler.ingest(
            text,
            title=f"response:{session_id or 'unknown'}",
            source_type="conversation",
        )

        evidence = Evidence(
            source_id=source.id,
            text=text[:500],
            weight=EvidenceWeight.SECONDARY.value,
            timestamp=event.ts,
        )
        claim = Claim(
            text=text[:500],
            evidence=[evidence],
            tags=tags,
            durability=durability,
        )
        claim._recalc_confidence()

        return HookResult(
            claims=[claim],
            source_id=source.id,
            entities_found=source.entities_extracted,
        )
