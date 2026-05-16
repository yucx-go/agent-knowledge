# agent-knowledge

[![PyPI](https://img.shields.io/pypi/v/compiled-memory.svg)](https://pypi.org/project/compiled-memory/)
[![Python](https://img.shields.io/pypi/pyversions/compiled-memory.svg)](https://pypi.org/project/compiled-memory/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![CI](https://github.com/yucx-go/agent-knowledge/actions/workflows/ci.yml/badge.svg)](https://github.com/yucx-go/agent-knowledge/actions/workflows/ci.yml)
[![MCP](https://img.shields.io/badge/MCP-compatible-blue.svg)](docs/mcp-integration.md)

**[English](README.md)** | 简体中文

为 AI Agent 设计的知识管理系统。把会话记录、文档、决策自动编译成带来源追溯、置信度追踪、矛盾检测的结构化知识库。MCP 即插即用，Claude Code / Cursor / Codex 等 agent 一行配置即可接入。

## Why

LLM 的会话记忆要么是扁平的 RAG 片段，要么是 key-value 偏好缓存——两者都回答不了「这件事的来龙去脉」。

agent-knowledge 在记忆之上增加一层**知识编译**：原始素材抽取出主张（Claim）和证据（Evidence），同一实体的多条主张合并为 Compiled Truth，时间线 append-only，新证据进来时整体重写。每条事实都能追溯到原始来源、时间戳、置信度。

## Before / After

**场景**：团队前端框架 10 周内反复决策——Week 1 选 Vue，Week 6 评估 React，Week 10 切到 React。三份会议纪要都进了 agent 的会话历史。

新同事问 *"我们前端用什么？为什么？"*

**Without** — RAG over chat history

```
团队决定使用 Vue 作为前端框架。
```

向量检索命中相似度最高的片段（最早那份纪要），**没有时间感知，给出过时答案**。

**With agent-knowledge** — `ak_query` 返回 entity（vault 里真实的 YAML）

```yaml
name: 前端技术栈
entity_type: concept
compiled_truth:
  claims:
    - text: 使用 React 作为 dashboard 前端
      status: active
      confidence: 0.85
      evidence:
        - source_id: c3a9d1f2
          weight: 1.0
    - text: 使用 Vue 作为 dashboard 前端
      status: superseded
      confidence: 0.65
      evidence:
        - source_id: a1b2c3d4
          weight: 0.7
timeline:
  - date: 2026-02-01
    title: 决定使用 Vue
    source_id: a1b2c3d4
  - date: 2026-03-12
    title: 评估 React 体验
    source_id: b5e6f7a8
  - date: 2026-04-09
    title: 切换至 React，Vue 生态限制
    source_id: c3a9d1f2
```

Agent 一次拿到**当前事实 + 时间线 + 来源 + 被 supersede 的旧主张**，回答自然包含「我们用 React，4 月从 Vue 切过来，因为生态限制」，且每条事实都可追溯到原始纪要。

## Quick Start

```bash
pip install compiled-memory   # PyPI 包名；Python 模块名是 `agent_knowledge`

ak init ~/my-knowledge
ak ingest ~/my-knowledge --file ./meeting-notes.md
ak query ~/my-knowledge "为什么选了 React？"
ak dream ~/my-knowledge       # 后台整合
ak lint  ~/my-knowledge       # 健康检查
```

也可以作为 MCP server 接入任意 MCP client：

```bash
ak mcp ~/my-knowledge
```

复制粘贴版的客户端配置在 [`examples/mcp/`](examples/mcp/)（Claude Code / Cursor / Codex），完整工具列表见 [`docs/mcp-integration.md`](docs/mcp-integration.md)。Agent 接手仓库后请先读 [`AGENTS.md`](AGENTS.md)。

## Architecture

```
┌─────────────────────────────────┐
│  Adapter Layer                  │  CLI / MCP / Pull adapters
├─────────────────────────────────┤
│  UMSF Boundary                  │  统一数据契约
├─────────────────────────────────┤
│  Knowledge Layer (核心)          │  Compiler · Compiled Truth · Hooks · Dream
├─────────────────────────────────┤
│  Storage                        │  Vault (YAML) + EventIndex (SQLite)
├─────────────────────────────────┤
│  Search Layer                   │  Exact + BM25 + Graph + RRF + Reranker
└─────────────────────────────────┘
```

- **Knowledge Layer** 纯 Python + 本地文件，无外部服务依赖
- **Search Layer** 默认零依赖；可选接入向量模型提升语义召回
- **Adapter Layer** 通过 UMSF 统一边界，新增一个 agent 适配器 ~80 行

详见 [`docs/architecture.md`](docs/architecture.md)。

## Benchmark

[LongMemEval-S (ICLR 2025)](https://arxiv.org/abs/2410.10813) — 500 questions, ~48 sessions/question, ~115K tokens/question：

| Metric | Score |
|:-------|:-----:|
| R@5    | **96.6%** |
| R@10   | 98.2% |
| MRR    | **0.9031** |
| NDCG@10 | 0.9218 |

零向量依赖，纯 BM25 + Exact Match + RRF 融合。完整 per-type 分解见 [`BENCHMARK.md`](BENCHMARK.md)。

## Documentation

- [`AGENTS.md`](AGENTS.md) — 给 AI agent 看的项目使用指南
- [`docs/architecture.md`](docs/architecture.md) — 五层架构与数据流
- [`docs/adapters.md`](docs/adapters.md) — 写一个新 adapter
- [`docs/mcp-integration.md`](docs/mcp-integration.md) — MCP server 接入
- [`BENCHMARK.md`](BENCHMARK.md) — benchmark 复现
- [`examples/mcp/`](examples/mcp/) — 各 MCP client 的即用配置

## License

MIT
