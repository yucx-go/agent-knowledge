# syntax=docker/dockerfile:1
#
# compiled-memory MCP server
# --------------------------
# A minimal image that starts the MCP server (JSON-RPC over stdio).
# Intended for MCP discovery platforms (Glama, Smithery, etc.) that run
# the container with `docker run -i <image>` and pipe JSON-RPC messages.
#
# Vault data is mounted/persisted at /data/vault — override with the
# AGENT_KNOWLEDGE_VAULT env var or by passing a path as the first arg.

FROM python:3.11-slim AS builder

WORKDIR /build
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN pip install --no-cache-dir --upgrade pip build && \
    python -m build --wheel --outdir /build/dist


FROM python:3.11-slim

LABEL org.opencontainers.image.title="compiled-memory" \
      org.opencontainers.image.description="AI agent knowledge & long-term memory — MCP server" \
      org.opencontainers.image.source="https://github.com/yucx-go/agent-knowledge" \
      org.opencontainers.image.licenses="MIT"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    AGENT_KNOWLEDGE_VAULT=/data/vault

RUN useradd --create-home --uid 1000 mcp && \
    mkdir -p /data/vault && chown -R mcp:mcp /data

COPY --from=builder /build/dist/*.whl /tmp/
RUN pip install --no-cache-dir /tmp/*.whl && rm /tmp/*.whl

USER mcp
WORKDIR /home/mcp
VOLUME ["/data/vault"]

ENTRYPOINT ["compiled-memory-mcp"]
