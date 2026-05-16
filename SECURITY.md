# Security Policy

## Supported versions

Only the latest minor release on `main` receives security fixes.

## Reporting a vulnerability

Please **do not** open public GitHub issues for security problems.

Use [GitHub Security Advisories](https://github.com/yucx-go/agent-knowledge/security/advisories/new)
to report privately. You will get an initial response within 7 days.

## Scope

In scope:

- Path traversal / arbitrary file write via vault operations
- YAML / SQLite injection through ingest payloads
- MCP server: command injection, untrusted tool argument handling
- Adapter code that reads untrusted files

Out of scope:

- Issues that require the attacker to already control the vault directory
- LLM hallucination or downstream agent misuse of returned content
- Denial of service from intentionally large ingest payloads on a local single-user vault
