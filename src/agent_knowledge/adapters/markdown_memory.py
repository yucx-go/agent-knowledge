"""Markdown memory adapter — pulls MEMORY.md / USER.md style curated files.

Many agent workflows converge on a similar convention: keep a small set
of human-readable markdown files that the agent itself curates during
use. Examples this adapter scans:

    ~/.claude/CLAUDE.md       — Claude Code's project/user context
    ~/.codex/AGENTS.md        — Codex CLI workspace instructions
    ~/.hermes/MEMORY.md       — personal "hermes"-style curated memory
    ~/.openclaw/MEMORY.md     — personal "openclaw"-style curated memory

The "hermes" and "openclaw" keys are kept as generic slot names for any
home-grown agent that follows the same `~/.<agent>/MEMORY.md` convention;
override `_AGENT_PATHS` (or write a new adapter) to point at your own
ecosystem.

These files are NOT raw conversation logs — they are pre-curated
summaries that the agent has already extracted from its experiences.
Treating them as high-priority sources gives agent-knowledge a fast
warm-start when bootstrapping a vault from existing agent installations.

Implementation
--------------
This adapter is a thin specialization of :class:`BasePullAdapter`:

    discover() — scan the conventional dotfile locations for known agents
    fetch()    — read a file, wrap as a curated_memory UMSFDocument

Everything else (UMSF-based ingestion, fingerprint dedup, state
persistence, dry-run, error collection) comes from the base class.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from ..core.umsf import UMSFDocument
from ..core.vault import Vault
from .base import BasePullAdapter, DiscoveredItem, PullReport  # noqa: F401  (re-exported)


# Per-agent file conventions: (relative_filename, role_label)
# Role labels surface in the source title and state metadata so users can
# tell at a glance what each ingested file represents.
_AGENT_PATHS: dict[str, list[tuple[str, str]]] = {
    "hermes": [
        ("MEMORY.md", "memory"),
        ("USER.md", "user_profile"),
        ("SOUL.md", "persona"),
        ("AGENTS.md", "workspace_instructions"),
    ],
    "openclaw": [
        ("MEMORY.md", "memory"),
        ("USER.md", "user_profile"),
        ("SOUL.md", "persona"),
        ("AGENTS.md", "workspace_instructions"),
    ],
    "claude-code": [
        ("CLAUDE.md", "workspace_instructions"),
        ("MEMORY.md", "memory"),
    ],
    "codex": [
        ("AGENTS.md", "workspace_instructions"),
    ],
}


KNOWN_AGENTS = tuple(_AGENT_PATHS.keys())


def _agent_root(agent: str, home: Optional[Path] = None) -> Path:
    """Return the conventional root directory for an agent.

    For "hermes" the adapter additionally checks ``%LOCALAPPDATA%\\hermes``
    on Windows before falling back to the dotfile path; other agents
    follow the dotfile convention under the user's home directory.
    """
    home = home or Path.home()

    if agent == "hermes":
        if os.name == "nt":
            local = os.environ.get("LOCALAPPDATA")
            if local:
                p = Path(local) / "hermes"
                if p.exists():
                    return p
        return home / ".hermes"

    if agent == "openclaw":
        return home / ".openclaw"
    if agent == "claude-code":
        return home / ".claude"
    if agent == "codex":
        return home / ".codex"

    raise ValueError(f"Unknown agent: {agent}")


@dataclass
class DiscoveredFile(DiscoveredItem):
    """A markdown memory file. Specialization of :class:`DiscoveredItem`."""
    path: Optional[Path] = None
    agent: str = ""
    role: str = ""
    size_bytes: int = 0
    mtime: float = 0.0

    def __post_init__(self) -> None:
        # Auto-derive base-class fields from path/agent/role/mtime so
        # callers don't have to set them manually.
        if self.path is None:
            return
        if not self.id:
            self.id = str(self.path)
        if not self.fingerprint:
            # mtime alone misses content-without-time-change edits, but in
            # practice agent dotfiles get rewritten with new mtimes; size
            # gives a cheap secondary signal.
            self.fingerprint = f"{self.size_bytes}:{self.mtime}"
        if not self.label:
            self.label = f"[{self.agent}] {self.path.stem} ({self.role})"
        # Surface adapter-specific fields in metadata for state inspection
        self.metadata.setdefault("agent", self.agent)
        self.metadata.setdefault("role", self.role)
        self.metadata.setdefault("size_bytes", self.size_bytes)
        self.metadata.setdefault("mtime", self.mtime)

    @property
    def title(self) -> str:
        """Backward-compatible alias for :attr:`label`."""
        return self.label


class MarkdownMemoryAdapter(BasePullAdapter):
    """Pull adapter for MEMORY.md-style curated markdown files.

    Usage::

        adapter = MarkdownMemoryAdapter(vault)
        report = adapter.pull()                       # all known agents
        report = adapter.pull(agents=["hermes"])      # filter
        report = adapter.pull(custom_paths=[...])     # extras
        report = adapter.pull(dry_run=True)           # discover only
    """

    name = "markdown-memory"

    def __init__(self, vault: Vault, home_override: Optional[Path] = None):
        super().__init__(vault)
        self._home = home_override

    # ── BasePullAdapter contract ──

    def discover(
        self,
        agents: Optional[list[str]] = None,
        custom_paths: Optional[list[str]] = None,
        **_: Any,
    ) -> list[DiscoveredFile]:
        """Find available markdown memory files.

        Args:
            agents: Filter to specific agent names. ``None`` = all known.
                Pass an empty list to skip all known-agent paths and only
                scan ``custom_paths``.
            custom_paths: Extra paths to include (treated as agent="custom").
        """
        results: list[DiscoveredFile] = []
        target_agents = list(KNOWN_AGENTS) if agents is None else list(agents)

        for agent in target_agents:
            if agent not in _AGENT_PATHS:
                continue
            try:
                root = _agent_root(agent, home=self._home)
            except ValueError:
                continue
            if not root.exists():
                continue
            for rel, role in _AGENT_PATHS[agent]:
                p = root / rel
                if p.is_file():
                    st = p.stat()
                    results.append(DiscoveredFile(
                        path=p,
                        agent=agent,
                        role=role,
                        size_bytes=st.st_size,
                        mtime=st.st_mtime,
                    ))

        if custom_paths:
            for cp in custom_paths:
                p = Path(cp).expanduser().resolve()
                if p.is_file():
                    st = p.stat()
                    results.append(DiscoveredFile(
                        path=p,
                        agent="custom",
                        role="memory",
                        size_bytes=st.st_size,
                        mtime=st.st_mtime,
                    ))

        return results

    def fetch(self, item: DiscoveredItem) -> Optional[UMSFDocument]:
        """Wrap a discovered file as a ``curated_memory`` UMSFDocument."""
        if not isinstance(item, DiscoveredFile) or item.path is None:
            return None
        text = item.path.read_text(encoding="utf-8")  # base catches exceptions
        if not text.strip():
            return None
        return UMSFDocument(
            agent=item.agent,
            source_type="curated_memory",
            title=item.label,
            context={
                "body": text,
                "path": str(item.path),
                "role": item.role,
            },
        )
