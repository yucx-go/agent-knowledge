"""MCP resource templates + read/list handlers.

Resources expose vault contents as browsable URIs:

    ak://entities         — all entities (summary list)
    ak://entities/{id}    — full compiled truth + claims + timeline
    ak://sources/{id}     — source content + metadata

Clients that support resource browsing (e.g., Claude Code) can navigate
the knowledge graph directly without going through tool calls.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .server import MCPServer


RESOURCE_TEMPLATES: list[dict] = [
    {
        "uriTemplate": "ak://entities",
        "name": "All Entities",
        "description": "List all entities in the knowledge vault with their compiled truth summaries.",
        "mimeType": "text/plain",
    },
    {
        "uriTemplate": "ak://entities/{entity_id}",
        "name": "Entity Detail",
        "description": "Read a specific entity's compiled truth, claims, and timeline.",
        "mimeType": "text/plain",
    },
    {
        "uriTemplate": "ak://sources/{source_id}",
        "name": "Source Content",
        "description": "Read a source document's content and metadata.",
        "mimeType": "text/plain",
    },
]


def handle_list(server: "MCPServer") -> dict:
    """MCP ``resources/list`` — concrete resources currently in the vault."""
    resources = []
    for eid in server.vault.list_entities():
        entity = server.vault.load_entity(eid)
        if entity:
            summary = (
                entity.compiled_truth.summary[:200]
                if entity.compiled_truth.summary
                else f"Entity: {entity.name}"
            )
            resources.append({
                "uri": f"ak://entities/{eid}",
                "name": entity.name,
                "description": summary,
                "mimeType": "text/plain",
            })
    for sid in server.vault.list_sources():
        source = server.vault.load_source(sid)
        if source:
            resources.append({
                "uri": f"ak://sources/{sid}",
                "name": source.title,
                "description": f"Source ({source.source_type}): {source.title}",
                "mimeType": "text/plain",
            })
    return {"resources": resources}


def handle_templates_list() -> dict:
    """MCP ``resources/templates/list`` — URI template catalog."""
    return {"resourceTemplates": RESOURCE_TEMPLATES}


def handle_read(server: "MCPServer", params: dict) -> dict:
    """MCP ``resources/read`` — fetch one resource by URI."""
    from .server import _RPCError

    uri = params.get("uri", "")

    if uri == "ak://entities":
        lines = []
        for eid in server.vault.list_entities():
            entity = server.vault.load_entity(eid)
            if entity:
                summary = entity.compiled_truth.summary[:100] or "(no summary)"
                lines.append(f"- {entity.name} [{eid}]: {summary}")
        text = "\n".join(lines) if lines else "No entities in vault."
        return {"contents": [{"uri": uri, "mimeType": "text/plain", "text": text}]}

    if uri.startswith("ak://entities/"):
        entity_id = uri[len("ak://entities/"):]
        entity = server.vault.load_entity(entity_id)
        if not entity:
            raise _RPCError(-32602, f"Entity not found: {entity_id}")
        parts = [
            f"# {entity.name}",
            f"Type: {entity.entity_type}",
            f"Aliases: {', '.join(entity.aliases) or 'none'}",
            f"Backlinks: {len(entity.backlinks)} sources",
            "",
            "## Compiled Truth",
            entity.compiled_truth.summary or "(empty)",
            "",
            "## Active Claims",
        ]
        for c in entity.compiled_truth.active_claims:
            parts.append(f"- [{c.confidence:.2f}] {c.text}")
        parts.append("")
        parts.append("## Timeline")
        for t in entity.compiled_truth.timeline[-10:]:
            parts.append(f"- {t.date}: {t.title}")
        text = "\n".join(parts)
        return {"contents": [{"uri": uri, "mimeType": "text/plain", "text": text}]}

    if uri.startswith("ak://sources/"):
        source_id = uri[len("ak://sources/"):]
        source = server.vault.load_source(source_id)
        if not source:
            raise _RPCError(-32602, f"Source not found: {source_id}")
        parts = [
            f"# {source.title}",
            f"Type: {source.source_type}",
            f"Ingested: {source.ingested_at}",
            f"Claims: {len(source.claims_extracted)}",
            f"Entities: {len(source.entities_extracted)}",
            "",
            "## Content",
            source.content,
        ]
        text = "\n".join(parts)
        return {"contents": [{"uri": uri, "mimeType": "text/plain", "text": text}]}

    raise _RPCError(-32602, f"Unknown resource URI: {uri}")
