"""MessageHook: extract preferences, directives, and foresight from user messages."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from ..core.models import Claim, ClaimDurability, Evidence, EvidenceWeight
from .base import Hook, HookResult

if TYPE_CHECKING:
    from ..core.compiler import Compiler
    from ..core.umsf import UMSFEvent
    from ..core.vault import Vault

# ── Pattern sets ──

_PREFERENCE_PATTERNS = re.compile(
    r"(?:我喜欢|我不喜欢|我偏好|我讨厌|我习惯|我倾向"
    r"|i like|i dislike|i prefer|i hate|i love|my preference)",
    re.IGNORECASE,
)

_DIRECTIVE_PATTERNS = re.compile(
    r"(?:记住|以后|从现在起|别再|永远|一直|始终"
    r"|remember|from now on|always|never|don't ever|going forward)",
    re.IGNORECASE,
)

_FORESIGHT_PATTERNS = re.compile(
    r"(?:下周要|计划|打算|准备|即将|明天要|下个月"
    r"|plan to|going to|will need|schedule|next week|next month|tomorrow)",
    re.IGNORECASE,
)


class MessageHook(Hook):
    """Fires on UMSF ``message`` events from the user.

    Unlike ResponseHook (which targets ``role="assistant"``), this hook
    only handles user-side messages. Empty role is treated as "user" by
    default for back-compat with adapters that don't set role explicitly.
    """

    name = "message"
    event_types = ["message"]

    def should_fire(self, event: "UMSFEvent") -> bool:
        if event.type not in self.event_types:
            return False
        # User messages only — let ResponseHook handle assistant messages
        if event.role and event.role != "user":
            return False
        text = event.content
        if len(text) < 5:
            return False
        return bool(
            _PREFERENCE_PATTERNS.search(text)
            or _DIRECTIVE_PATTERNS.search(text)
            or _FORESIGHT_PATTERNS.search(text)
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
        durability = ClaimDurability.UNKNOWN

        if _PREFERENCE_PATTERNS.search(text):
            tags.append("preference")
            durability = ClaimDurability.STABLE
        if _DIRECTIVE_PATTERNS.search(text):
            tags.append("directive")
            durability = ClaimDurability.STABLE
        if _FORESIGHT_PATTERNS.search(text):
            tags.append("foresight")
            durability = ClaimDurability.TEMPORARY

        source = compiler.ingest(
            text,
            title=f"message:{session_id or 'unknown'}",
            source_type="conversation",
        )

        evidence = Evidence(
            source_id=source.id,
            text=text[:500],
            weight=EvidenceWeight.PRIMARY.value,
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
