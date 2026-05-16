"""MCP Server — JSON-RPC 2.0 dispatcher over stdio.

Owns the lifecycle (initialize / notifications / ping) and routes
feature methods to the focused submodules:

    tools/list, tools/call                  → :mod:`.tools`
    resources/list, resources/read,         → :mod:`.resources`
    resources/templates/list
    prompts/list, prompts/get               → :mod:`.prompts`

Why this is small
-----------------
Tool / resource / prompt logic lives in their own files; the server
keeps only the protocol skeleton and a handful of compatibility shims
(``_tool_*`` methods that delegate to :mod:`.tools` for tests that
poke private methods directly).
"""

from __future__ import annotations

import json
import sys
from typing import Any

from ...core.vault import Vault
from . import prompts as _prompts
from . import resources as _resources
from . import tools as _tools

# MCP protocol version we advertise during initialize handshake
MCP_PROTOCOL_VERSION = "2024-11-05"


class MCPServer:
    """JSON-RPC 2.0 dispatcher over stdin/stdout."""

    def __init__(self, vault: Vault):
        self.vault = vault
        self._initialized = False
        self._hook_manager = None  # lazy

    # ── Hook manager (lazy) ──

    def _get_hook_manager(self):
        if self._hook_manager is None:
            from ...hooks.base import HookManager
            self._hook_manager = HookManager(self.vault)
            self._hook_manager.register_defaults()
        return self._hook_manager

    # ── Message handling ──

    def handle_message(self, msg: dict) -> dict | None:
        """Handle a single JSON-RPC message.

        Returns response dict for requests, None for notifications.
        """
        method = msg.get("method", "")
        msg_id = msg.get("id")
        params = msg.get("params", {})
        is_notification = msg_id is None

        try:
            result = self._dispatch(method, params)
        except _RPCError as e:
            if is_notification:
                return None
            return _error_response(msg_id, e.code, str(e))
        except Exception as e:
            if is_notification:
                return None
            return _error_response(msg_id, -32603, str(e))

        if is_notification:
            return None
        return _success_response(msg_id, result)

    def _dispatch(self, method: str, params: dict) -> Any:
        if method == "initialize":
            return self._handle_initialize(params)
        if method == "notifications/initialized":
            self._initialized = True
            return None
        if method == "tools/list":
            return _tools.handle_list()
        if method == "tools/call":
            return _tools.handle_call(self, params)
        if method == "resources/list":
            return _resources.handle_list(self)
        if method == "resources/templates/list":
            return _resources.handle_templates_list()
        if method == "resources/read":
            return _resources.handle_read(self, params)
        if method == "prompts/list":
            return _prompts.handle_list()
        if method == "prompts/get":
            return _prompts.handle_get(self, params)
        if method == "ping":
            return {}
        raise _RPCError(-32601, f"Method not found: {method}")

    def _handle_initialize(self, params: dict) -> dict:
        return {
            "protocolVersion": MCP_PROTOCOL_VERSION,
            "capabilities": {
                "tools": {"listChanged": False},
                "resources": {"subscribe": False, "listChanged": False},
                "prompts": {"listChanged": False},
            },
            "serverInfo": {
                "name": "agent-knowledge",
                "version": "0.3.1",
            },
        }

    # ── Compatibility shims ──
    #
    # Older tests (and any external callers) poke these private methods
    # directly. They delegate to the appropriate submodule handler so
    # there is exactly one source of truth per tool / prompt.

    def _tool_query(self, args: dict) -> dict:
        return _tools.tool_query(self, args)

    def _tool_ingest(self, args: dict) -> dict:
        return _tools.tool_ingest(self, args)

    def _tool_ingest_umsf(self, args: dict) -> dict:
        return _tools.tool_ingest_umsf(self, args)

    def _tool_stats(self, args: dict) -> dict:
        return _tools.tool_stats(self, args)

    def _tool_dream(self, args: dict) -> dict:
        return _tools.tool_dream(self, args)

    def _tool_hook_fire(self, args: dict) -> dict:
        return _tools.tool_hook_fire(self, args)

    def _tool_hook_list(self, args: dict) -> dict:
        return _tools.tool_hook_list(self, args)

    def _tool_hook_stats(self, args: dict) -> dict:
        return _tools.tool_hook_stats(self, args)


# ── JSON-RPC error helpers ──


class _RPCError(Exception):
    """JSON-RPC error with an error code (-32601, -32602, -32603, …)."""

    def __init__(self, code: int, message: str):
        super().__init__(message)
        self.code = code


def _success_response(msg_id: Any, result: Any) -> dict:
    return {"jsonrpc": "2.0", "id": msg_id, "result": result}


def _error_response(msg_id: Any, code: int, message: str) -> dict:
    return {"jsonrpc": "2.0", "id": msg_id, "error": {"code": code, "message": message}}


# ── Stdio entry point ──


def run_stdio(vault_path: str) -> None:
    """Run MCP server reading JSON-RPC from stdin, writing to stdout."""
    vault = Vault(vault_path)
    if not vault.is_initialized:
        vault.init()

    server = MCPServer(vault)

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue

        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            resp = _error_response(None, -32700, "Parse error")
            sys.stdout.write(json.dumps(resp) + "\n")
            sys.stdout.flush()
            continue

        resp = server.handle_message(msg)
        if resp is not None:
            sys.stdout.write(json.dumps(resp) + "\n")
            sys.stdout.flush()
