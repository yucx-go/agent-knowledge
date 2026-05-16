"""MCP Server adapter — JSON-RPC over stdio.

Exposes agent-knowledge as an MCP (Model Context Protocol) server so that
AI agents can query, ingest, and manage knowledge vaults via the standard
MCP tool-calling interface.

Module layout
-------------
    server.py     — MCPServer class, dispatch, run_stdio entry point
    tools.py      — tool schemas + handler functions (ak_query / ak_ingest / …)
    resources.py  — resource templates + read/list handlers (ak://entities/…)
    prompts.py    — prompt templates + render handlers (knowledge_summary, …)

Usage
-----
    ak mcp /path/to/vault          # start MCP server (CLI)
    echo '{...}' | ak mcp /vault   # one-shot JSON-RPC request

Public API (all re-exported here for convenience)::

    from agent_knowledge.adapters.mcp import (
        MCPServer, run_stdio, MCP_PROTOCOL_VERSION,
        TOOLS, RESOURCE_TEMPLATES, PROMPTS,
    )
"""

from .prompts import PROMPTS
from .resources import RESOURCE_TEMPLATES
from .server import MCP_PROTOCOL_VERSION, MCPServer, run_stdio
from .tools import TOOLS

__all__ = [
    "MCPServer",
    "run_stdio",
    "MCP_PROTOCOL_VERSION",
    "TOOLS",
    "RESOURCE_TEMPLATES",
    "PROMPTS",
]
