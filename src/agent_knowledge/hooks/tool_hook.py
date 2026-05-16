"""ToolHook: capture tool usage patterns and key return values."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from ..core.models import Claim, ClaimDurability, Evidence, EvidenceWeight
from .base import Hook, HookResult

if TYPE_CHECKING:
    from ..core.compiler import Compiler
    from ..core.umsf import UMSFEvent
    from ..core.vault import Vault

# Patterns to extract key identifiers from tool results
_KEY_PATTERNS = re.compile(
    r"(?:token|id|url|key|path)\s*[=:]\s*(\S+)",
    re.IGNORECASE,
)


class ToolHook(Hook):
    """Fires on UMSF ``tool_use`` events. Reads:

    - ``event.name``    — tool name (required)
    - ``event.args``    — structured tool arguments (preserved as-is)
    - ``event.result``  — stringified tool output, scanned for key/value
                          identifiers (tokens, ids, paths)
    """

    name = "tool_use"
    event_types = ["tool_use"]

    def should_fire(self, event: "UMSFEvent") -> bool:
        if event.type not in self.event_types:
            return False
        return bool(event.name)

    def process(
        self,
        event: "UMSFEvent",
        session_id: str,
        vault: "Vault",
        compiler: "Compiler",
    ) -> HookResult:
        tool_name = event.name
        args = event.args or {}
        result = event.result or ""

        # Build summary text — keeps the original "Tool: X | Args: ..." shape
        # so existing search/index expectations don't shift.
        summary_parts = [f"Tool: {tool_name}"]
        if args:
            arg_str = ", ".join(f"{k}={v}" for k, v in list(args.items())[:5])
            summary_parts.append(f"Args: {arg_str}")

        # Extract key values from result
        result_str = str(result)[:2000]
        keys_found = _KEY_PATTERNS.findall(result_str)
        if keys_found:
            summary_parts.append(f"Keys: {', '.join(keys_found[:5])}")

        summary = " | ".join(summary_parts)

        source = compiler.ingest(
            summary,
            title=f"tool:{tool_name}",
            source_type="tool_use",
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
            tags=["tool_use", tool_name],
            durability=ClaimDurability.TEMPORARY,
        )
        claim._recalc_confidence()

        return HookResult(
            claims=[claim],
            source_id=source.id,
            entities_found=source.entities_extracted,
        )
