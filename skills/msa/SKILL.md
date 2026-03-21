---
name: msa
description: "Memory Sparse Attention 记忆系统（文档级稀疏路由 + LLM 多跳推理）。使用场景: (1) 大文档摄入并建立路由索引, (2) 基于稀疏 Top-k 的文档级检索, (3) 通过 Memory Interleave 进行 LLM 驱动的多跳推理。NOT for: 日常笔记(用文件操作), 简单语义搜索(用 memora), 深度知识内化(用 chronos)。"
metadata:
  openclaw:
    emoji: "🔀"
    requires:
      anyBins: ["python3"]
---

# MSA — Memory Sparse Attention 记忆系统

MSA 是 OpenClaw 的第四套记忆系统，基于 [EverMind-AI/MSA](https://github.com/EverMind-AI/MSA) 论文。核心特点：**文档级稀疏路由** + **LLM 驱动的多跳 Memory Interleave**。

## v2.0 更新

- **Memory Server 加速**: 嵌入模型常驻内存，查询 <100ms
- **LLM 多跳推理**: Memory Interleave 现在接入 xAI Grok API 进行中间生成
- **共享嵌入模型**: 与 Memora 共用 SentenceTransformer，节省 ~500MB 内存

## 与 Memora 的关键区别

- **Memora**: 扁平向量搜索，每条记忆一个嵌入，返回孤立片段
- **MSA**: 文档级路由（chunk-mean pooling），Top-k 选择后返回**整篇文档的所有 chunk**

## 使用方法

### 推荐：通过 Memory Server

```bash
# 智能路由（长文本自动走 MSA）
memory-cli remember "一篇很长的文档..." -i 0.8 -t "文档标题"

# 跨系统合并搜索
memory-cli recall "量子纠缠如何用于通信？"

# 多跳推理（LLM 驱动，每轮检索新文档并生成中间答案）
memory-cli deep-recall "量子纠缠如何影响量子计算的容错能力？"
```

### 直接 MSA CLI

```bash
# 摄入文档
python3 -m msa ingest "文档内容..." -t "标题"
python3 -m msa ingest -f /path/to/document.txt -t "标题"

# 单轮检索
python3 -m msa query "问题" -k 3

# 多跳推理
python3 -m msa interleave "复杂问题" -r 5

# 管理
python3 -m msa remove doc_id
python3 -m msa status
python3 -m msa report
```

## 何时使用 MSA

**用 MSA / deep-recall 当：**
- 需要摄入长文档并保留完整上下文
- 问题需要跨多个段落或多篇文档的推理
- 需要 LLM 驱动的多跳推理

**用 Memora / search 当：**
- 搜索简短的对话记忆或事实片段
- 需要快速的单条语义检索

## 数据位置

- 路由索引：`~/.openclaw/workspace/memory/msa/routing_index.jsonl`
- 文档内容：`~/.openclaw/workspace/memory/msa/content/{doc_id}.jsonl`
- 系统状态：`~/.openclaw/workspace/memory/msa/state.json`
- 配置文件：`~/.openclaw/workspace/msa/config.yaml`
