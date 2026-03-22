---
name: msa
description: "长文档存储 — 稀疏路由 + 多跳推理"
metadata:
  openclaw:
    emoji: "🔀"
    requires:
      anyBins: ["python3"]
---

# MSA — 长文档存储

文档级稀疏路由 + LLM 驱动多跳 Memory Interleave。>500 字内容自动路由到 MSA。

## 何时使用

| 场景 | 命令 |
|------|------|
| 长文档自动摄入 | `memory-cli remember "长文本..." -i 0.8 -t "标题"` |
| 合并搜索 | `memory-cli recall "问题"` |
| 多跳推理 | `memory-cli deep-recall "复杂问题"` |

## 与 Memora 的区别

- **Memora**: 扁平向量搜索，返回孤立片段
- **MSA**: 文档级路由，返回整篇文档所有 chunk + LLM 多跳推理

## 何时不用

- 短记忆搜索 → `memory-cli search`（仅 Memora，更快）
- 不涉及长文档的问题 → `recall` 已足够

## 数据

- 路由索引: `memory/msa/routing_index.jsonl`
- 文档内容: `memory/msa/content/{doc_id}.jsonl`
