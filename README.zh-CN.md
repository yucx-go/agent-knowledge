# agent-knowledge

> **AI Agent 的长期记忆——零向量依赖。**

[![PyPI](https://img.shields.io/pypi/v/compiled-memory.svg)](https://pypi.org/project/compiled-memory/)
[![Downloads](https://static.pepy.tech/badge/compiled-memory)](https://pepy.tech/project/compiled-memory)
[![Python](https://img.shields.io/pypi/pyversions/compiled-memory.svg)](https://pypi.org/project/compiled-memory/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![CI](https://github.com/yucx-go/agent-knowledge/actions/workflows/ci.yml/badge.svg)](https://github.com/yucx-go/agent-knowledge/actions/workflows/ci.yml)
[![MCP](https://img.shields.io/badge/MCP-compatible-blue.svg)](docs/mcp-integration.md)
[![GitHub stars](https://img.shields.io/github/stars/yucx-go/agent-knowledge?style=social)](https://github.com/yucx-go/agent-knowledge/stargazers)

**[English](README.md)** | 简体中文

为 AI Agent 设计的**长期记忆与知识库**系统。把会话记录、文档、决策自动编译成结构化知识——带主张/证据追溯、append-only 时间线、矛盾检测。纯 Python，本地优先，MIT 许可。

作为 **MCP server**（stdio JSON-RPC 2.0）即插即用，[Claude Code](examples/mcp/claude-code.json)、[Cursor](examples/mcp/cursor.json)、[Codex](examples/mcp/codex.toml) 一行配置接入。**LongMemEval-S 上 R@5 96.6%**，零向量依赖——BM25 + 知识图谱 + RRF。

## Why

LLM 的会话记忆要么是扁平的 RAG 片段，要么是 key-value 偏好缓存——两者都回答不了「这件事的来龙去脉」。

agent-knowledge 在记忆之上增加一层**知识编译**：原始素材抽取出主张（Claim）和证据（Evidence），同一实体的多条主张合并为 Compiled Truth，时间线 append-only，新证据进来时整体重写。每条事实都能追溯到原始来源、时间戳、置信度。

## Features

- 🧠 **可编译的长期记忆**——Claim / Evidence / Compiled Truth / append-only Timeline
- 🔍 **多路检索**——Exact + BM25 + Knowledge Graph + 加权 RRF + TF-IDF 重排
- 🚫 **零向量依赖**——无 embedding、无向量数据库、无外部服务
- 🔌 **MCP server**——stdio JSON-RPC 2.0，8 个 tool + 2 个 resource URI，兼容 Claude Code / Cursor / Codex 等任意 MCP 客户端
- 🪶 **本地优先存储**——人类可读的 YAML vault + SQLite 事件索引，git 友好
- 🪝 **自动捕获 Hooks**——内置 7 种 hook，覆盖消息、工具调用、决策、文件变更、错误
- ⚔️ **矛盾检测**——基于极性识别，浮出「我们之前说 X，现在说非 X」
- 💤 **Dream cycle**——离线整合、去重、supersession
- 📊 **已 benchmark**——LongMemEval-S（ICLR 2025）R@5 96.6%, MRR 0.9031
- 🪪 **MIT 许可**，Python 3.10–3.13，运行时仅依赖 PyYAML

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

## FAQ

<details>
<summary><strong>和向量数据库 / RAG 有什么本质区别？</strong></summary>

向量 RAG 按 embedding 相似度召回文本片段，**无法判断一条事实是当前有效、已经被推翻、还是与其他事实冲突**。agent-knowledge 把原始输入编译成结构化的 Claim，按 entity 合并成 Compiled Truth，每条主张显式标注 `active` / `superseded` / `disputed`，并保留 append-only 时间线——agent 一次拿到「当前答案 + 它的历史脉络」。

你仍然可以把向量模型当作可选的重排信号引入，但不是必需。

</details>

<details>
<summary><strong>必须装向量数据库（Chroma / Qdrant / Pinecone ...）吗？</strong></summary>

不需要。默认检索是 Exact Match + BM25 + 知识图谱 + 加权 RRF + TF-IDF 重排，全部纯 Python、全部本地。运行时**只有一个依赖** `PyYAML`。LongMemEval-S 上零向量路径达到 **R@5 96.6%**，对得起主流 embedding 基线。

</details>

<details>
<summary><strong>和 mem0 / Letta / Zep / LangChain memory 比有什么差别？</strong></summary>

主流记忆框架存的是**片段**（消息、摘要、embedding），按相似度召回。agent-knowledge 围绕**知识编译**设计：每条 Claim 有来源、每个 entity 有 Compiled Truth、时间线在新证据进来时整体重写。输出是**可追溯的结构化知识**，不是被记住的对话集合。

优化目标不同，都有效，按需选。

</details>

<details>
<summary><strong>能接入 Claude Code / Cursor / Codex 吗？</strong></summary>

可以——它本身就是 MCP server（stdio JSON-RPC 2.0）。复制粘贴配置在 [`examples/mcp/`](examples/mcp/)：
- Claude Code: [`examples/mcp/claude-code.json`](examples/mcp/claude-code.json)
- Cursor: [`examples/mcp/cursor.json`](examples/mcp/cursor.json)
- Codex: [`examples/mcp/codex.toml`](examples/mcp/codex.toml)

启动命令：`compiled-memory-mcp`（默认 vault 路径 `~/.agent-knowledge/vault`），或 `ak mcp /your/vault/path`。

</details>

<details>
<summary><strong>数据存哪？会不会上传云端？</strong></summary>

默认什么都不上传。Vault 是一个目录里的人类可读 YAML 文件 + SQLite 事件索引——git 友好。无任何 telemetry，无任何 phone-home，无任何强制 API key。

</details>

<details>
<summary><strong>UMSF 是什么？</strong></summary>

Universal Memory Source Format——一个小型 JSON schema，统一会话、工具轨迹、决策、文件变更等数据如何提交到 vault。8 种 event 类型，7 种 source 类型。这是让不同 agent 的 adapter（Claude Code、Codex、自定义 hermes / openclaw 槽位）能共享同一套 ingest 管线的关键。详见 [`docs/architecture.md`](docs/architecture.md)。

</details>

## Citation

如果在研究或论文中使用 agent-knowledge，请引用：

```bibtex
@software{agent_knowledge_2026,
  author  = {Yu, Chengxin},
  title   = {agent-knowledge: long-term memory and knowledge compilation for AI agents},
  year    = {2026},
  url     = {https://github.com/yucx-go/agent-knowledge},
  version = {0.3.1}
}
```

仓库已附带 [`CITATION.cff`](CITATION.cff)，GitHub 引用 widget 可自动识别。

## License

MIT — 见 [`LICENSE`](LICENSE)。
