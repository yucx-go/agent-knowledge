# MCP Integration

agent-knowledge ships a full MCP server (JSON-RPC 2.0 over stdio).
Any MCP-aware client can call it.

## Setup

For Claude Code (`.claude/settings.json` or `~/.claude/settings.json`)
and Cursor (`.cursor/mcp.json`) — same payload:

```json
{
  "mcpServers": {
    "agent-knowledge": {
      "command": "ak",
      "args": ["mcp", "/path/to/your/vault"]
    }
  }
}
```

For other clients, start the server directly and pipe JSON-RPC over
stdin/stdout:

```bash
ak mcp /path/to/vault
```

If the vault doesn't exist, the server auto-initializes it.

## Tools

| Tool | Purpose | Required args |
|------|---------|---------------|
| `ak_query` | Search the vault | `query` (+ `top_k`, default 5) |
| `ak_ingest` | Ingest one prose document | `text` (+ `title`, `source_type`) |
| `ak_ingest_umsf` | Ingest a full `UMSFDocument` (multi-event) | `doc` (UMSFDocument dict) |
| `ak_stats` | Vault counts | — |
| `ak_dream` | Run a dream cycle | (`since_hours`, default 24) |
| `ak_hook_fire` | Fire one UMSF event for real-time capture | `type` (+ event fields) |
| `ak_hook_list` | List registered hooks | — |
| `ak_hook_stats` | Per-hook fire counts | — |

`ak_hook_fire` takes a flat event payload (`type`, `content`, `role`,
`name`, `args`, `result`, `error`, `path`, `metadata`, `session_id`,
`agent`). See [`adapters/mcp/tools.py`](../src/agent_knowledge/adapters/mcp/tools.py)
for the full schema.

## Resources

Browse the vault as MCP resources (e.g. via Claude Code's resource UI):

| URI | What |
|-----|------|
| `ak://entities` | List all entities with summaries |
| `ak://entities/{id}` | Full entity: compiled truth, claims, timeline |
| `ak://sources/{id}` | Source content + metadata |

## Prompts

| Prompt | What | Args |
|--------|------|------|
| `knowledge_summary` | Structured summary of an entity | `entity_name` |
| `contradiction_check` | Analyze and resolve contradictions | — |
