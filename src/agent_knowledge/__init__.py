"""agent-knowledge: AI Agent Knowledge Management System.

Public API. Stable across 0.3.x; internal modules may change.

Quick start (Python):

    from agent_knowledge import Vault, Compiler, SearchEngine

    vault = Vault("/path/to/vault")
    vault.init()
    compiler = Compiler(vault)
    compiler.ingest("We picked React over Vue.", title="Frontend decision")

    engine = SearchEngine(vault)
    for hit in engine.search("why React?", top_k=5):
        print(hit.title, hit.score)

For MCP server / CLI usage see README.md and docs/mcp-integration.md.
"""

from agent_knowledge.core.vault import Vault
from agent_knowledge.core.compiler import Compiler
from agent_knowledge.core.models import (
    Claim,
    ClaimDurability,
    ClaimPolarity,
    ClaimStatus,
    CompiledTruth,
    Entity,
    Evidence,
    EvidenceWeight,
    Source,
    TimelineEvent,
)
from agent_knowledge.core.umsf import (
    Actor,
    UMSFDocument,
    UMSFEvent,
    UMSFEventType,
    UMSFValidationError,
)
from agent_knowledge.search.engine import SearchEngine, SearchResult

__version__ = "0.3.0"

__all__ = [
    "__version__",
    "Vault",
    "Compiler",
    "SearchEngine",
    "SearchResult",
    "Claim",
    "ClaimDurability",
    "ClaimPolarity",
    "ClaimStatus",
    "CompiledTruth",
    "Entity",
    "Evidence",
    "EvidenceWeight",
    "Source",
    "TimelineEvent",
    "Actor",
    "UMSFDocument",
    "UMSFEvent",
    "UMSFEventType",
    "UMSFValidationError",
]
