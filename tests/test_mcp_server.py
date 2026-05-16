"""Tests for MCP Server adapter."""

import json
import os
import shutil
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from agent_knowledge.adapters.mcp import MCPServer, TOOLS, PROMPTS, RESOURCE_TEMPLATES, MCP_PROTOCOL_VERSION
from agent_knowledge.core.vault import Vault
from agent_knowledge.core.compiler import Compiler

TEST_DIR = "/tmp/ak-test-mcp"


@pytest.fixture(autouse=True)
def clean_vault():
    if os.path.exists(TEST_DIR):
        shutil.rmtree(TEST_DIR)
    vault = Vault(TEST_DIR)
    vault.init()
    yield vault
    if os.path.exists(TEST_DIR):
        shutil.rmtree(TEST_DIR)


def _call(server, method, params=None, msg_id=1):
    msg = {"jsonrpc": "2.0", "method": method, "id": msg_id}
    if params:
        msg["params"] = params
    return server.handle_message(msg)


class TestMCPInitialize:
    def test_initialize(self, clean_vault):
        server = MCPServer(clean_vault)
        resp = _call(server, "initialize", {"protocolVersion": MCP_PROTOCOL_VERSION})
        assert resp["result"]["protocolVersion"] == MCP_PROTOCOL_VERSION
        assert "tools" in resp["result"]["capabilities"]
        assert resp["result"]["serverInfo"]["name"] == "agent-knowledge"

    def test_initialized_notification(self, clean_vault):
        server = MCPServer(clean_vault)
        # Notification (no id) should return None
        msg = {"jsonrpc": "2.0", "method": "notifications/initialized"}
        resp = server.handle_message(msg)
        assert resp is None


class TestToolsList:
    def test_list_tools(self, clean_vault):
        server = MCPServer(clean_vault)
        resp = _call(server, "tools/list")
        tools = resp["result"]["tools"]
        names = {t["name"] for t in tools}
        assert names == {
            "ak_query", "ak_ingest", "ak_ingest_umsf", "ak_stats", "ak_dream",
            "ak_hook_fire", "ak_hook_list", "ak_hook_stats",
        }

    def test_tools_have_schema(self, clean_vault):
        server = MCPServer(clean_vault)
        resp = _call(server, "tools/list")
        for tool in resp["result"]["tools"]:
            assert "inputSchema" in tool
            assert "description" in tool


class TestToolQuery:
    def test_query_empty_vault(self, clean_vault):
        server = MCPServer(clean_vault)
        resp = _call(server, "tools/call", {
            "name": "ak_query",
            "arguments": {"query": "test"},
        })
        assert "No results" in resp["result"]["content"][0]["text"]

    def test_query_with_data(self, clean_vault):
        compiler = Compiler(clean_vault)
        compiler.ingest("## 决策\n- MCP 协议是通信标准方案", title="MCP Doc")

        server = MCPServer(clean_vault)
        resp = _call(server, "tools/call", {
            "name": "ak_query",
            "arguments": {"query": "MCP", "top_k": 3},
        })
        text = resp["result"]["content"][0]["text"]
        assert "MCP" in text

    def test_query_missing_param(self, clean_vault):
        server = MCPServer(clean_vault)
        resp = _call(server, "tools/call", {
            "name": "ak_query",
            "arguments": {},
        })
        assert resp["result"].get("isError", False)


class TestToolIngest:
    def test_ingest_text(self, clean_vault):
        server = MCPServer(clean_vault)
        resp = _call(server, "tools/call", {
            "name": "ak_ingest",
            "arguments": {
                "text": "## 决策\n- 选择 React 框架构建前端",
                "title": "React Decision",
            },
        })
        text = resp["result"]["content"][0]["text"]
        assert "Ingested" in text
        assert "React Decision" in text

    def test_ingest_empty_text(self, clean_vault):
        server = MCPServer(clean_vault)
        resp = _call(server, "tools/call", {
            "name": "ak_ingest",
            "arguments": {"text": ""},
        })
        assert resp["result"].get("isError", False)


class TestToolStats:
    def test_stats_empty(self, clean_vault):
        server = MCPServer(clean_vault)
        resp = _call(server, "tools/call", {
            "name": "ak_stats",
            "arguments": {},
        })
        text = resp["result"]["content"][0]["text"]
        assert "sources: 0" in text

    def test_stats_after_ingest(self, clean_vault):
        compiler = Compiler(clean_vault)
        compiler.ingest("## 决策\n- 重要决策", title="Test")

        server = MCPServer(clean_vault)
        resp = _call(server, "tools/call", {
            "name": "ak_stats",
            "arguments": {},
        })
        text = resp["result"]["content"][0]["text"]
        assert "sources: 1" in text


class TestToolDream:
    def test_dream_empty(self, clean_vault):
        server = MCPServer(clean_vault)
        resp = _call(server, "tools/call", {
            "name": "ak_dream",
            "arguments": {"since_hours": 9999},
        })
        text = resp["result"]["content"][0]["text"]
        assert "Dream Report" in text
        assert "0 candidates" in text

    def test_dream_with_data(self, clean_vault):
        compiler = Compiler(clean_vault)
        compiler.ingest("## 决策\n- 架构决策：使用编译式知识图谱", title="Arch")

        server = MCPServer(clean_vault)
        resp = _call(server, "tools/call", {
            "name": "ak_dream",
            "arguments": {"since_hours": 9999},
        })
        text = resp["result"]["content"][0]["text"]
        assert "Dream Report" in text


class TestErrorHandling:
    def test_unknown_method(self, clean_vault):
        server = MCPServer(clean_vault)
        resp = _call(server, "nonexistent/method")
        assert "error" in resp
        assert resp["error"]["code"] == -32601

    def test_unknown_tool(self, clean_vault):
        server = MCPServer(clean_vault)
        resp = _call(server, "tools/call", {
            "name": "nonexistent_tool",
            "arguments": {},
        })
        assert "error" in resp

    def test_ping(self, clean_vault):
        server = MCPServer(clean_vault)
        resp = _call(server, "ping")
        assert resp["result"] == {}

    def test_jsonrpc_format(self, clean_vault):
        server = MCPServer(clean_vault)
        resp = _call(server, "tools/list", msg_id=42)
        assert resp["jsonrpc"] == "2.0"
        assert resp["id"] == 42


class TestResources:
    def test_resources_list_empty(self, clean_vault):
        server = MCPServer(clean_vault)
        resp = _call(server, "resources/list")
        assert resp["result"]["resources"] == []

    def test_resources_templates_list(self, clean_vault):
        server = MCPServer(clean_vault)
        resp = _call(server, "resources/templates/list")
        templates = resp["result"]["resourceTemplates"]
        assert len(templates) == 3
        uris = [t["uriTemplate"] for t in templates]
        assert "ak://entities" in uris
        assert "ak://entities/{entity_id}" in uris
        assert "ak://sources/{source_id}" in uris

    def test_resources_list_after_ingest(self, clean_vault):
        compiler = Compiler(clean_vault)
        compiler.ingest("## 决策\n- 选择 React 框架", title="React Note")

        server = MCPServer(clean_vault)
        resp = _call(server, "resources/list")
        resources = resp["result"]["resources"]
        assert len(resources) > 0
        uris = [r["uri"] for r in resources]
        assert any("sources/" in u for u in uris)

    def test_resources_read_entities_list(self, clean_vault):
        compiler = Compiler(clean_vault)
        compiler.ingest("## 决策\n- 选择 React 框架", title="React Note")

        server = MCPServer(clean_vault)
        resp = _call(server, "resources/read", {"uri": "ak://entities"})
        text = resp["result"]["contents"][0]["text"]
        assert "React" in text or len(text) > 0

    def test_resources_read_source(self, clean_vault):
        compiler = Compiler(clean_vault)
        source = compiler.ingest("## 决策\n- 选择 React 框架", title="React Note")

        server = MCPServer(clean_vault)
        resp = _call(server, "resources/read", {"uri": f"ak://sources/{source.id}"})
        contents = resp["result"]["contents"]
        assert len(contents) == 1
        assert "React" in contents[0]["text"]

    def test_resources_read_unknown(self, clean_vault):
        server = MCPServer(clean_vault)
        resp = _call(server, "resources/read", {"uri": "ak://unknown/xyz"})
        assert "error" in resp


class TestPrompts:
    def test_prompts_list(self, clean_vault):
        server = MCPServer(clean_vault)
        resp = _call(server, "prompts/list")
        prompts = resp["result"]["prompts"]
        names = [p["name"] for p in prompts]
        assert "knowledge_summary" in names
        assert "contradiction_check" in names

    def test_prompt_contradiction_check(self, clean_vault):
        server = MCPServer(clean_vault)
        resp = _call(server, "prompts/get", {
            "name": "contradiction_check",
            "arguments": {},
        })
        result = resp["result"]
        assert "messages" in result
        assert len(result["messages"]) == 1
        assert "contradiction" in result["messages"][0]["content"]["text"].lower()

    def test_prompt_knowledge_summary(self, clean_vault):
        compiler = Compiler(clean_vault)
        compiler.ingest("## 决策\n- React is the chosen framework", title="Tech")

        server = MCPServer(clean_vault)
        resp = _call(server, "prompts/get", {
            "name": "knowledge_summary",
            "arguments": {"entity_name": "React"},
        })
        result = resp["result"]
        assert "messages" in result
        assert "React" in result["messages"][0]["content"]["text"]

    def test_prompt_knowledge_summary_not_found(self, clean_vault):
        server = MCPServer(clean_vault)
        resp = _call(server, "prompts/get", {
            "name": "knowledge_summary",
            "arguments": {"entity_name": "NonexistentEntity"},
        })
        result = resp["result"]
        assert "No entity named" in result["messages"][0]["content"]["text"]

    def test_prompt_unknown(self, clean_vault):
        server = MCPServer(clean_vault)
        resp = _call(server, "prompts/get", {
            "name": "nonexistent_prompt",
            "arguments": {},
        })
        assert "error" in resp


class TestCapabilities:
    def test_capabilities_include_resources(self, clean_vault):
        server = MCPServer(clean_vault)
        resp = _call(server, "initialize", {"protocolVersion": MCP_PROTOCOL_VERSION})
        caps = resp["result"]["capabilities"]
        assert "resources" in caps
        assert "prompts" in caps
        assert "tools" in caps

    def test_version_matches_package(self, clean_vault):
        from agent_knowledge import __version__

        server = MCPServer(clean_vault)
        resp = _call(server, "initialize", {"protocolVersion": MCP_PROTOCOL_VERSION})
        assert resp["result"]["serverInfo"]["version"] == __version__
