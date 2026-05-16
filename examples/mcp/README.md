# MCP Integration Examples

Drop-in config snippets for popular MCP-aware clients.

| Client | File | Location |
|--------|------|----------|
| Claude Code | [`claude-code.json`](claude-code.json) | `~/.claude/settings.json` or `.claude/settings.json` |
| Cursor | [`cursor.json`](cursor.json) | `~/.cursor/mcp.json` or `.cursor/mcp.json` |
| Codex | [`codex.toml`](codex.toml) | `~/.codex/config.toml` |

Steps:

1. `pip install agent-knowledge`
2. `ak init /absolute/path/to/vault`
3. Copy the snippet above into your client's config, replace the vault path
4. Restart the client

The MCP server exposes 8 tools (`ak_query`, `ak_ingest`, `ak_ingest_umsf`,
`ak_hook_fire`, `ak_stats`, `ak_dream`, `ak_hook_list`, `ak_hook_stats`).
See [`../../docs/mcp-integration.md`](../../docs/mcp-integration.md) for the
full schema.

For a non-MCP smoke test of the library itself, run [`../demo.py`](../demo.py).
