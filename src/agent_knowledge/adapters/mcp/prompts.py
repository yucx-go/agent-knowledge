"""MCP prompt templates + render handlers.

A "prompt" in MCP terminology is a server-side template that, when
requested with arguments, returns a fully-formed user message ready to
send to a model. Useful for canned analyses where the server can pull
the relevant data and format it consistently.

Available prompts
-----------------
    knowledge_summary    — summarize what we know about an entity
    contradiction_check  — analyze contradictions detected by the compiler
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .server import MCPServer


PROMPTS: list[dict] = [
    {
        "name": "knowledge_summary",
        "description": "Generate a structured knowledge summary for a given entity.",
        "arguments": [
            {
                "name": "entity_name",
                "description": "Name of the entity to summarize.",
                "required": True,
            },
        ],
    },
    {
        "name": "contradiction_check",
        "description": "Analyze the vault for contradictions and generate a report.",
        "arguments": [],
    },
]


def handle_list() -> dict:
    """MCP ``prompts/list`` response."""
    return {"prompts": PROMPTS}


def handle_get(server: "MCPServer", params: dict) -> dict:
    """MCP ``prompts/get`` — render the named prompt with arguments."""
    from .server import _RPCError

    name = params.get("name", "")
    arguments = params.get("arguments", {})

    if name == "knowledge_summary":
        return _knowledge_summary(server, arguments)
    if name == "contradiction_check":
        return _contradiction_check(server, arguments)
    raise _RPCError(-32602, f"Unknown prompt: {name}")


# ── Renderers ──


def _knowledge_summary(server: "MCPServer", args: dict) -> dict:
    from .server import _RPCError

    entity_name = args.get("entity_name", "")
    if not entity_name:
        raise _RPCError(-32602, "entity_name is required")

    entity = None
    for eid in server.vault.list_entities():
        e = server.vault.load_entity(eid)
        if e and e.name.lower() == entity_name.lower():
            entity = e
            break

    if not entity:
        context = f"No entity named '{entity_name}' found in the knowledge vault."
    else:
        claims_text = "\n".join(
            f"- [{c.confidence:.2f}] {c.text}"
            for c in entity.compiled_truth.active_claims
        )
        timeline_text = "\n".join(
            f"- {t.date}: {t.title}"
            for t in entity.compiled_truth.timeline[-5:]
        )
        context = (
            f"Entity: {entity.name}\n"
            f"Type: {entity.entity_type}\n"
            f"Current Summary: {entity.compiled_truth.summary}\n\n"
            f"Active Claims:\n{claims_text}\n\n"
            f"Recent Timeline:\n{timeline_text}"
        )

    return {
        "description": f"Summarize knowledge about {entity_name}",
        "messages": [
            {
                "role": "user",
                "content": {
                    "type": "text",
                    "text": (
                        f"Based on the following knowledge base data, provide a structured summary "
                        f"of what we know about '{entity_name}'. Include key decisions, current state, "
                        f"and any notable changes over time.\n\n{context}"
                    ),
                },
            }
        ],
    }


def _contradiction_check(server: "MCPServer", args: dict) -> dict:
    from ...core.compiler import Compiler

    compiler = Compiler(server.vault)
    contradictions = compiler.detect_contradictions()

    if not contradictions:
        context = "No contradictions detected in the vault."
    else:
        lines = []
        for c in contradictions:
            lines.append(
                f"Entity: {c['entity']}\n"
                f"  Claim A [{c['claim_a']['confidence']:.2f}]: {c['claim_a']['text']}\n"
                f"  Claim B [{c['claim_b']['confidence']:.2f}]: {c['claim_b']['text']}"
            )
        context = (
            f"Found {len(contradictions)} contradiction(s):\n\n"
            + "\n\n".join(lines)
        )

    return {
        "description": "Analyze contradictions in the knowledge vault",
        "messages": [
            {
                "role": "user",
                "content": {
                    "type": "text",
                    "text": (
                        f"The following contradictions were detected in the knowledge vault. "
                        f"For each, analyze which claim is more likely correct based on confidence "
                        f"scores and evidence, and suggest a resolution.\n\n{context}"
                    ),
                },
            }
        ],
    }
