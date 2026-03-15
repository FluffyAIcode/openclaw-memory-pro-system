# Memora + OpenClaw 集成手册

## 概述

Memora 是 OpenClaw 的增强型记忆管理系统，作为 **OpenClaw Skill** 部署。它不替换原生 `memory/` 机制，而是在其上叠加向量语义搜索、双写、自动提炼能力。

## 架构

```
用户对话 / Heartbeat
       ↓
  OpenClaw Agent (LLM)
       ↓ (通过 bash tool 调用)
  python3 -m memora <command>
       ↓
  ┌─────────────────────────────┐
  │  memora.bridge              │
  │  ├→ collector (daily .md)   │
  │  ├→ vectorstore (JSONL)     │
  │  └→ memory/ (OpenClaw 原生) │
  └─────────────────────────────┘
       ↓
  memory/daily/       → 每日记录
  memory/vector_db/   → 向量持久化
  memory/long_term/   → 提炼摘要
  memory/YYYY-MM-DD.md → OpenClaw 原生日文件
```

## 集成点

| 集成方式 | 文件 | 作用 |
|---------|------|------|
| **Skill** | `skills/memora/SKILL.md` | 指导 LLM 何时及如何调用 memora |
| **AGENTS.md** | `AGENTS.md` Memory 章节 | 启动时提示 agent 可用 memora 搜索 |
| **HEARTBEAT.md** | `HEARTBEAT.md` | 定期触发 `memora digest` |
| **Python 包** | `pip install -e .` | `python3 -m memora` 全局可用 |

## 快速使用

```bash
# 保存记忆（双写到向量库 + OpenClaw memory/）
python3 -m memora add "重要发现" -i 0.9

# 语义搜索
python3 -m memora search "关键词"

# 提炼近 7 天记忆
python3 -m memora digest --days 7

# 查看状态
python3 -m memora status

# 初始化目录
python3 -m memora init
```

## 环境变量

| 变量 | 作用 | 默认值 |
|------|------|--------|
| `MEMORA_BASE_DIR` | 记忆存储根目录 | `~/.openclaw/workspace/memory` |
| `MEMORA_VLLM_URL` | vLLM API 地址 | `http://localhost:8000/v1/chat/completions` |
| `MEMORA_EMBEDDING_MODEL` | 嵌入模型名 | `nomic-ai/nomic-embed-text-v1.5` |

## 数据目录

- `memory/daily/` — Memora daily markdown
- `memory/vector_db/entries.jsonl` — 向量持久化
- `memory/long_term/digest_*.md` — 提炼摘要
- `memory/long_term/lora_training/` — LoRA 训练数据
- `memora/config.yaml` — 配置文件

---

**最后更新**: 2026-03-15
**状态**: 已部署为 OpenClaw Skill，生产可用
