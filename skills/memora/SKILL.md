---
name: memora
description: "统一语料库 — 向量存储与语义搜索"
metadata:
  openclaw:
    emoji: "🧠"
    requires:
      anyBins: ["python3"]
---

# Memora — 向量存储

所有内容首先进入 Memora（nomic-embed-text-v1.5, JSONL），提供语义搜索和去重。

## 何时使用

| 场景 | 命令 |
|------|------|
| 记住重要内容 | `memory-cli remember "内容" --tag thought -i 0.8` |
| 语义搜索 | `memory-cli search "关键词"` |
| 合并搜索 (Memora+MSA) | `memory-cli recall "关键词"` |
| 直接添加 | `memory-cli add "内容" -i 0.9` |
| 记忆总结 | `memory-cli digest --days 7` |

## 何时不用

- 长文档推理 → `memory-cli deep-recall`（走 MSA）
- KG 推理 → `memory-cli contradictions` / `blindspots`（走 Second Brain）

## 数据

- 向量库: `memory/vectorstore.jsonl`
- 日志: `memory/YYYY-MM-DD.md`
