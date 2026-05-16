"""DecisionHook: detect decision keywords and create high-confidence stable claims."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from ..core.models import Claim, ClaimDurability, Evidence, EvidenceWeight
from .base import Hook, HookResult

if TYPE_CHECKING:
    from ..core.compiler import Compiler
    from ..core.umsf import UMSFEvent
    from ..core.vault import Vault

_DECISION_PATTERNS = re.compile(
    r"(?:决定|选择|采用|放弃|切换|确定|敲定|最终|拍板"
    r"|decided|chose|selected|adopted|abandoned|switched|finalized|going with)",
    re.IGNORECASE,
)


class DecisionHook(Hook):
    """Fires on UMSF ``decision`` events and on ``message`` events that
    mention decision-marking verbs. Either way, the extracted text gets
    a high-confidence stable claim tagged ``decision``.
    """

    name = "decision"
    event_types = ["decision", "message"]

    def should_fire(self, event: "UMSFEvent") -> bool:
        if event.type not in self.event_types:
            return False
        text = event.content
        if len(text) < 5:
            return False
        return bool(_DECISION_PATTERNS.search(text))

    def process(
        self,
        event: "UMSFEvent",
        session_id: str,
        vault: "Vault",
        compiler: "Compiler",
    ) -> HookResult:
        text = event.content

        source = compiler.ingest(
            text,
            title=f"decision:{session_id or 'unknown'}",
            source_type="decision",
        )

        evidence = Evidence(
            source_id=source.id,
            text=text[:500],
            weight=EvidenceWeight.PRIMARY.value,
            timestamp=event.ts,
        )
        claim = Claim(
            text=text[:500],
            confidence=0.85,  # high confidence for explicit decisions
            evidence=[evidence],
            tags=["decision"],
            durability=ClaimDurability.STABLE,
        )
        # Don't recalc — keep the high base confidence
        # (recalc would set it to weight/(weight+1) ≈ 0.5)

        return HookResult(
            claims=[claim],
            source_id=source.id,
            entities_found=source.entities_extracted,
        )
